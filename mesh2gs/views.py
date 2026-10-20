"""Training views for M2 / M3: Cycles renders of the source objects from cameras on a sphere.

Everything happens in a temporary scene with world-space copies of the objects (modifiers
applied), so the user's scene, objects and materials stay as they are. Albedo views route
Base Color into Emission, the same target the albedo Gaussians carry; lit views use the
lit bake's stand-in materials (metals as diffuse, like the lit M1 start) and bring along
the scene's world and lights, while the other objects of the scene go into an "indirect
only" collection: they still shadow and reflect, but the camera does not see them.

Output, as the trainer reads it: <out_dir>/train/view_XXX.png (RGBA, sRGB, "Standard" view
transform) and <out_dir>/cameras_train.json (OpenCV intrinsics and world-to-camera matrices
in Blender world space).
"""

import json
import math
import os

import bpy
import numpy as np
from mathutils import Vector

from . import bake, preview

FOV = math.radians(40)
INDEX = "cameras_train.json"
ALBEDO_SAMPLES = 16  # emission only: a few samples anti-alias the edges


def fibonacci_directions(n, offset=0.0):
    """n near-uniform unit vectors on a sphere."""
    i = np.arange(n) + offset
    z = 1 - 2 * i / n
    r = np.sqrt(np.maximum(0.0, 1 - z * z))
    phi = i * math.pi * (3 - math.sqrt(5))
    return np.stack([r * np.cos(phi), r * np.sin(phi), z], axis=1)


def camera_record(cam_obj, width, height):
    """Intrinsics and OpenCV world-to-camera matrix (x right, y down, z forward), as gsplat wants."""
    f = (width / 2) / math.tan(cam_obj.data.angle / 2)
    c2w = np.array(cam_obj.matrix_world, dtype=np.float64)
    c2w_cv = c2w @ np.diag([1.0, -1.0, -1.0, 1.0])  # Blender camera looks down -Z with +Y up
    return {
        "width": width, "height": height,
        "K": [[f, 0, width / 2], [0, f, height / 2], [0, 0, 1]],
        "viewmat": np.linalg.inv(c2w_cv).tolist(),
    }


class ViewRenderer:
    """Renders view i of `count` with render(i); close() removes everything it created."""

    def __init__(self, objects, depsgraph, out_dir, count=64, size=512, lit=False, samples=64, emission=True):
        source = bpy.context.scene
        self.out_dir, self.count, self.size = out_dir, count, size
        self.records = []
        self._objects, self._meshes, self._materials, self._collections = [], [], [], []
        self.scene = bpy.data.scenes.new("mesh2gs_views")
        try:
            self._copy_objects(objects, depsgraph, lit, emission)
            if lit:
                self._bring_context(source, objects)
            self._setup_render(source, lit, samples)
            self.center, radius = self._bounding_sphere()
            self.distance = radius / math.sin(FOV / 2) * 1.05
            self._add_camera()
            self.directions = fibonacci_directions(count)
            os.makedirs(os.path.join(out_dir, "train"), exist_ok=True)
        except Exception:
            self.close()
            raise

    def _copy_objects(self, objects, depsgraph, lit, emission):
        for obj in objects:
            mesh = bpy.data.meshes.new_from_object(obj.evaluated_get(depsgraph),
                                                   preserve_all_data_layers=True, depsgraph=depsgraph)
            self._meshes.append(mesh)
            copy = bpy.data.objects.new(f"{obj.name}.mesh2gs_view", mesh)
            self._objects.append(copy)
            self.scene.collection.objects.link(copy)
            copy.matrix_world = obj.matrix_world
            slots = [s.material for s in obj.material_slots] or [None]
            mesh.materials.clear()
            for mat in slots:
                # The same stand-ins as the bakes, so M3 refines toward what M1 started from.
                mat = bake._lit_copy(mat) if lit else bake._emission_copy(mat, glow=emission)
                self._materials.append(mat)
                mesh.materials.append(mat)

    def _bring_context(self, source, objects):
        """Lights as they are; every other renderable object only for shadows and reflections."""
        self.scene.world = source.world
        lights = bpy.data.collections.new("mesh2gs_views_lights")
        others = bpy.data.collections.new("mesh2gs_views_context")
        self._collections += [lights, others]
        self.scene.collection.children.link(lights)
        self.scene.collection.children.link(others)
        names = {o.name for o in objects}
        for o in source.objects:
            if o.name in names or preview.is_gaussian_object(o) or o.hide_render or not o.visible_get():
                continue
            if o.type == 'LIGHT':
                lights.objects.link(o)
            elif o.type in {'MESH', 'CURVE', 'SURFACE', 'META', 'FONT', 'VOLUME', 'CURVES', 'POINTCLOUD'}:
                others.objects.link(o)
        self.scene.view_layers[0].layer_collection.children[others.name].indirect_only = True

    def _setup_render(self, source, lit, samples):
        scene = self.scene
        scene.render.engine = 'CYCLES'
        scene.cycles.device = source.cycles.device
        scene.cycles.samples = samples if lit else ALBEDO_SAMPLES
        scene.cycles.use_denoising = False
        scene.render.film_transparent = True
        scene.view_settings.view_transform = 'Standard'
        scene.view_settings.look = 'None'
        scene.view_settings.exposure = 0.0
        scene.view_settings.gamma = 1.0
        scene.render.resolution_x = scene.render.resolution_y = self.size
        scene.render.resolution_percentage = 100
        scene.render.image_settings.file_format = 'PNG'
        scene.render.image_settings.color_mode = 'RGBA'
        scene.render.image_settings.color_depth = '8'
        if not lit:
            scene.world = None  # emission only; the background is transparent anyway

    def _bounding_sphere(self):
        points = []
        for obj in self._objects:
            co = np.empty(len(obj.data.vertices) * 3)
            obj.data.vertices.foreach_get("co", co)
            world = np.array(obj.matrix_world, dtype=np.float64)
            points.append(co.reshape(-1, 3) @ world[:3, :3].T + world[:3, 3])
        pts = np.concatenate(points)
        center = (pts.min(0) + pts.max(0)) / 2
        return center, max(float(np.linalg.norm(pts - center, axis=1).max()), 1e-6)

    def _add_camera(self):
        data = bpy.data.cameras.new("mesh2gs_view")
        data.lens_unit = 'FOV'
        data.angle = FOV
        data.sensor_fit = 'HORIZONTAL'
        data.clip_start, data.clip_end = self.distance * 0.01, self.distance * 10
        self.camera = bpy.data.objects.new("mesh2gs_view", data)
        self._objects.append(self.camera)
        self.scene.collection.objects.link(self.camera)
        self.scene.camera = self.camera

    def render(self, i):
        cam = self.camera
        cam.location = Vector(self.center) + Vector(self.directions[i]) * self.distance
        cam.rotation_euler = (Vector(self.center) - cam.location).to_track_quat('-Z', 'Y').to_euler()
        self.scene.view_layers[0].update()
        name = f"view_{i:03d}.png"
        self.scene.render.filepath = os.path.join(self.out_dir, "train", name)
        bpy.ops.render.render(write_still=True, scene=self.scene.name)
        self.records.append({"image": f"train/{name}", **camera_record(cam, self.size, self.size)})

    def write_index(self):
        path = os.path.join(self.out_dir, INDEX)
        with open(path, "w") as f:
            json.dump({"views": self.records}, f)
        return path

    def close(self):
        cameras = [o.data for o in self._objects if o.type == 'CAMERA']
        for o in self._objects:
            bpy.data.objects.remove(o)
        for c in cameras:
            bpy.data.cameras.remove(c)
        for m in self._meshes:
            bpy.data.meshes.remove(m)
        for m in self._materials:
            bpy.data.materials.remove(m)
        for c in self._collections:
            bpy.data.collections.remove(c)
        if self.scene is not None:
            bpy.data.scenes.remove(self.scene)
            self.scene = None
        self._objects, self._meshes, self._materials, self._collections = [], [], [], []
