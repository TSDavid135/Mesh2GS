"""Operator tests on a real scene; skipped outside Blender."""

import os
import tempfile
import unittest

import numpy as np

try:
    import bpy
except ModuleNotFoundError:
    bpy = None

from mesh2gs.core.color import linear_to_srgb
from mesh2gs.io import ply
from test_core import quaternion_to_matrix

BUDGET = 20_000


def setUpModule():
    if bpy is not None:
        import mesh2gs
        mesh2gs.register()


def tearDownModule():
    if bpy is not None:
        import mesh2gs
        mesh2gs.unregister()


def material(name, rgb):
    mat = bpy.data.materials.new(name)
    mat.use_nodes = True
    bsdf = next(n for n in mat.node_tree.nodes if n.type == 'BSDF_PRINCIPLED')
    bsdf.inputs["Base Color"].default_value = (*rgb, 1.0)
    return mat


QUADRANTS = [(1.0, 0.0, 0.0), (0.0, 1.0, 0.0), (0.0, 0.0, 1.0), (1.0, 1.0, 1.0)]  # sRGB, row 0 at bottom


def image_material(name, uv_map=""):
    """2x2 image (red green / blue white), Closest, optionally on a named UV map."""
    img = bpy.data.images.new(name, 2, 2)
    img.pixels[:] = [c for rgb in QUADRANTS for c in (*rgb, 1.0)]
    mat = bpy.data.materials.new(name)
    mat.use_nodes = True
    nodes, links = mat.node_tree.nodes, mat.node_tree.links
    bsdf = next(n for n in nodes if n.type == 'BSDF_PRINCIPLED')
    tex = nodes.new('ShaderNodeTexImage')
    tex.image, tex.interpolation = img, 'Closest'
    links.new(tex.outputs["Color"], bsdf.inputs["Base Color"])
    if uv_map:
        uv = nodes.new('ShaderNodeUVMap')
        uv.uv_map = uv_map
        links.new(uv.outputs["UV"], tex.inputs["Vector"])
    return mat


def gradient_material(name, metallic=0.0, coords="Generated"):
    """Base Color = linear gradient along x of the given texture coordinates (procedural)."""
    mat = material(name, (0.5, 0.5, 0.5))
    nodes, links = mat.node_tree.nodes, mat.node_tree.links
    bsdf = next(n for n in nodes if n.type == 'BSDF_PRINCIPLED')
    bsdf.inputs["Metallic"].default_value = metallic
    grad = nodes.new('ShaderNodeTexGradient')
    texco = nodes.new('ShaderNodeTexCoord')
    links.new(texco.outputs[coords], grad.inputs["Vector"])
    links.new(grad.outputs["Color"], bsdf.inputs["Base Color"])
    return mat


def quadrant_colors(xy):
    """Expected sRGB for points on a size-2 plane whose UVs span 0..1."""
    q = (xy[:, 1] >= 0) * 2 + (xy[:, 0] >= 0)
    return np.array(QUADRANTS)[q]


def world_normals(g):
    return quaternion_to_matrix(g.rotations / np.linalg.norm(g.rotations, axis=1, keepdims=True))[:, :, 2]


@unittest.skipIf(bpy is None, "needs Blender")
class OperatorTest(unittest.TestCase):
    def setUp(self):
        if bpy.context.object and bpy.context.object.mode != 'OBJECT':
            bpy.ops.object.mode_set(mode='OBJECT')
        for obj in list(bpy.data.objects):
            bpy.data.objects.remove(obj)
        self.tmp = tempfile.TemporaryDirectory()
        self.path = os.path.join(self.tmp.name, "out.ply")
        s = bpy.context.scene.mesh2gs
        s.method, s.budget_mode, s.budget, s.hide_source = 'M1', 'COUNT', BUDGET, True

    def tearDown(self):
        self.tmp.cleanup()

    def convert(self):
        """Run the converter, export the result and read the PLY back."""
        self.assertEqual(bpy.ops.mesh2gs.convert(), {'FINISHED'})
        obj = bpy.context.active_object
        # Blender axes, so positions compare directly with the scene
        self.assertEqual(bpy.ops.mesh2gs.export_ply(filepath=self.path, axis_convention='RFU'), {'FINISHED'})
        return obj, ply.read(self.path)

    def test_import_round_trip(self):
        bpy.ops.mesh.primitive_cube_add(size=2, location=(1, 2, 3))
        obj, original = self.convert()
        bpy.ops.mesh2gs.export_ply(filepath=self.path)  # RDF
        assert bpy.ops.mesh2gs.import_ply(filepath=self.path, axis_convention='RDF') == {'FINISHED'}
        imported = bpy.context.active_object
        self.assertEqual(imported.name, "out")
        from mesh2gs import preview
        back = preview.read_object(imported)
        np.testing.assert_allclose(back.positions, original.positions, atol=1e-5)
        np.testing.assert_allclose(back.colors, original.colors, atol=1e-5)

    def test_export_axis_conventions(self):
        bpy.ops.mesh.primitive_plane_add(size=1, location=(0.3, 0.5, 2.0))  # above the origin
        self.convert()
        bpy.ops.mesh2gs.export_ply(filepath=self.path)  # default: RDF, as 3DGS / SuperSplat
        rdf = ply.read(self.path)
        np.testing.assert_allclose(rdf.positions[:, 1], -2.0, atol=1e-5)   # up -> -y
        np.testing.assert_allclose(rdf.positions[:, 2].mean(), 0.5, atol=0.02)  # forward -> +z
        bpy.ops.mesh2gs.export_ply(filepath=self.path, axis_convention='RUF')  # Unity
        ruf = ply.read(self.path)
        np.testing.assert_allclose(ruf.positions[:, 1], 2.0, atol=1e-5)
        np.testing.assert_allclose(ruf.positions[:, 0].mean(), 0.3, atol=0.02)

    def test_export_spz(self):
        bpy.ops.mesh.primitive_plane_add(size=1, location=(0.3, 0.5, 2.0))
        self.convert()
        path = os.path.splitext(self.path)[0] + ".spz"
        self.assertEqual(bpy.ops.mesh2gs.export_spz(filepath=path), {'FINISHED'})
        from mesh2gs.io import spz
        rub = spz.read(path)  # SPZ is RUB: up -> +y, forward -> -z
        np.testing.assert_allclose(rub.positions[:, 1], 2.0, atol=1e-3)
        np.testing.assert_allclose(rub.positions[:, 2].mean(), -0.5, atol=0.02)
        # Imported back, it lands where it was, in Blender axes.
        original = bpy.context.active_object
        self.assertEqual(bpy.ops.mesh2gs.import_spz(filepath=path), {'FINISHED'})
        from mesh2gs import preview
        back, ref = preview.read_object(bpy.context.active_object), preview.read_object(original)
        np.testing.assert_allclose(back.positions, ref.positions, atol=1e-3)
        np.testing.assert_allclose(back.colors, ref.colors, atol=0.02)
        # Scaled on export: a 1 m plane written 10x larger.
        bpy.context.view_layer.objects.active = original
        self.assertEqual(bpy.ops.mesh2gs.export_spz(filepath=path, scale=10.0), {'FINISHED'})
        np.testing.assert_allclose(spz.read(path).positions[:, 1], 20.0, atol=1e-2)

    def test_spz_scale_notes(self):
        from mesh2gs.operators import spz_scale_notes
        self.assertEqual(spz_scale_notes(np.array([[0, 0, 0], [1.0, 1, 1]])), [])
        self.assertIn("5.0 cm", spz_scale_notes(np.array([[0, 0, 0], [0.05, 0, 0]]))[0])  # 5 cm model
        self.assertIn("±2048", spz_scale_notes(np.array([[0, 0, 0], [3000.0, 0, 0]]))[0])  # too far out

    def test_transformed_cube(self):
        bpy.ops.mesh.primitive_cube_add(size=2, location=(1, 2, 3), scale=(1, 1, 2))
        cube = bpy.context.active_object
        cube.data.materials.append(material("red", (1.0, 0.0, 0.0)))
        obj, g = self.convert()

        self.assertEqual(obj.name, "Cube.M1")
        self.assertIn(obj.name, bpy.data.collections["Gaussians"].objects)
        self.assertTrue(cube.hide_get())
        self.assertEqual(len(g), BUDGET)
        local = (g.positions - [1, 2, 3]) / [1, 1, 2]
        np.testing.assert_allclose(np.abs(local).max(axis=1), 1.0, atol=1e-5)
        # scale (1,1,2): side faces have twice the area of top/bottom -> 8 of 10 area units
        self.assertAlmostEqual(np.mean(np.abs(local[:, 2]) < 1 - 1e-4), 0.8, delta=0.02)
        np.testing.assert_allclose(np.abs(world_normals(g)).max(axis=1), 1.0, atol=1e-5)
        np.testing.assert_allclose(g.colors, np.tile(linear_to_srgb([1.0, 0.0, 0.0]), (BUDGET, 1)), atol=1e-6)

    def test_modifiers_are_applied(self):
        bpy.ops.mesh.primitive_plane_add(size=1)
        bpy.context.active_object.modifiers.new("array", 'ARRAY').count = 3
        _, g = self.convert()
        self.assertAlmostEqual(g.positions[:, 0].max() - g.positions[:, 0].min(), 3.0, places=2)

    def test_several_objects_and_materials(self):
        bpy.ops.mesh.primitive_plane_add(size=1, location=(0, 0, 0))
        bpy.context.active_object.data.materials.append(material("green", (0.0, 1.0, 0.0)))
        bpy.ops.mesh.primitive_plane_add(size=2, location=(5, 0, 0))
        bpy.context.active_object.data.materials.append(material("blue", (0.0, 0.0, 1.0)))
        for obj in bpy.data.objects:
            obj.select_set(True)
        obj, g = self.convert()
        self.assertEqual(obj.name, "Mesh2GS.M1")
        on_big = g.positions[:, 0] > 2.5
        self.assertAlmostEqual(on_big.mean(), 0.8, delta=0.02)
        np.testing.assert_allclose(g.colors[on_big], np.tile([0, 0, 1], (on_big.sum(), 1)), atol=1e-6)
        np.testing.assert_allclose(g.colors[~on_big], np.tile([0, 1, 0], ((~on_big).sum(), 1)), atol=1e-6)

    def test_image_texture_per_point(self):
        bpy.ops.mesh.primitive_plane_add(size=2)
        bpy.context.active_object.data.materials.append(image_material("quad"))
        _, g = self.convert()
        away = np.abs(g.positions[:, :2]).min(axis=1) > 0.01  # skip texel borders
        np.testing.assert_allclose(g.colors[away], quadrant_colors(g.positions[away, :2]), atol=1e-3)

    def test_named_uv_map(self):
        bpy.ops.mesh.primitive_plane_add(size=2)
        plane = bpy.context.active_object
        second = plane.data.uv_layers.new(name="flipped")
        uv = np.empty(len(plane.data.loops) * 2, dtype=np.float32)
        second.data.foreach_get("uv", uv)
        second.data.foreach_set("uv", (1.0 - uv))  # both axes mirrored
        plane.data.uv_layers.active = plane.data.uv_layers[0]
        plane.data.materials.append(image_material("quad_flipped", uv_map="flipped"))
        _, g = self.convert()
        away = np.abs(g.positions[:, :2]).min(axis=1) > 0.01
        np.testing.assert_allclose(g.colors[away], quadrant_colors(-g.positions[away, :2]), atol=1e-3)

    def test_procedural_color_falls_back_without_bake(self):
        bpy.context.scene.mesh2gs.bake_procedural = False
        self.addCleanup(setattr, bpy.context.scene.mesh2gs, "bake_procedural", True)
        bpy.ops.mesh.primitive_plane_add(size=2)
        mat = material("noise", (0.2, 0.2, 0.2))
        mat.diffuse_color = (0.0, 1.0, 0.0, 1.0)
        noise = mat.node_tree.nodes.new('ShaderNodeTexNoise')
        bsdf = next(n for n in mat.node_tree.nodes if n.type == 'BSDF_PRINCIPLED')
        mat.node_tree.links.new(noise.outputs["Color"], bsdf.inputs["Base Color"])
        bpy.context.active_object.data.materials.append(mat)
        from mesh2gs import materials
        self.assertEqual(materials.slot_color(mat).kind, 'UNSUPPORTED')
        _, g = self.convert()
        np.testing.assert_allclose(g.colors, np.tile([0, 1, 0], (len(g), 1)), atol=1e-6)

    def test_bake_gradient_ignores_metallic(self):
        bpy.context.scene.mesh2gs.bake_resolution = '1024'
        bpy.ops.mesh.primitive_plane_add(size=2)
        bpy.context.active_object.data.materials.append(gradient_material("grad", metallic=1.0))
        _, g = self.convert()
        fac = (g.positions[:, 0] + 1) / 2  # Generated coordinates span the plane's box
        np.testing.assert_allclose(g.colors[:, 0], linear_to_srgb(fac), atol=0.03)

    def test_bake_leaves_scene_untouched(self):
        bpy.context.scene.mesh2gs.bake_resolution = '1024'
        bpy.ops.mesh.primitive_plane_add(size=2)
        plane = bpy.context.active_object
        mat = gradient_material("grad")
        plane.data.materials.append(mat)
        before = {k: set(getattr(bpy.data, k).keys()) for k in ("objects", "meshes", "materials", "images")}
        node_count, uv_names = len(mat.node_tree.nodes), [l.name for l in plane.data.uv_layers]
        obj, _ = self.convert()
        after = {k: set(getattr(bpy.data, k).keys()) for k in before}
        self.assertEqual(after["objects"] - before["objects"], {obj.name})
        self.assertEqual(after["meshes"] - before["meshes"], {obj.data.name})
        self.assertLessEqual(after["materials"] - before["materials"], {"Mesh2GS Splat"})
        self.assertEqual(after["images"], before["images"])
        self.assertEqual(len(mat.node_tree.nodes), node_count)
        self.assertEqual([l.name for l in plane.data.uv_layers], uv_names)

    def test_bake_mesh_without_uv(self):
        bpy.context.scene.mesh2gs.bake_resolution = '1024'
        bpy.ops.mesh.primitive_plane_add(size=2)
        plane = bpy.context.active_object
        while plane.data.uv_layers:
            plane.data.uv_layers.remove(plane.data.uv_layers[0])
        plane.data.materials.append(gradient_material("grad"))
        _, g = self.convert()
        fac = (g.positions[:, 0] + 1) / 2
        np.testing.assert_allclose(g.colors[:, 0], linear_to_srgb(fac), atol=0.03)

    def test_bake_array_copies_get_their_own_texels(self):
        bpy.context.scene.mesh2gs.bake_resolution = '1024'
        bpy.ops.mesh.primitive_plane_add(size=2)
        plane = bpy.context.active_object
        plane.modifiers.new("array", 'ARRAY').count = 2  # second copy at x in [1, 3]
        plane.data.materials.append(gradient_material("grad", coords="Object"))
        _, g = self.convert()
        x = g.positions[:, 0]
        right = x > 1.05                   # object x > 1: gradient saturates at 1
        mid = (x > 0.05) & (x < 0.95)      # first copy: gradient = x
        # A handful of points on UV island borders blend with bake margins; 99% must match.
        self.assertGreater(np.mean(np.abs(g.colors[right, 0] - 1.0) < 0.03), 0.99)
        self.assertGreater(np.mean(np.abs(g.colors[mid, 0] - linear_to_srgb(x[mid])) < 0.03), 0.99)

    def test_lit_mode_bakes_scene_light(self):
        s = bpy.context.scene.mesh2gs
        s.color_mode, s.bake_resolution, s.lit_samples = 'LIT', '1024', 16
        self.addCleanup(setattr, s, "color_mode", 'ALBEDO')
        bpy.context.scene.world = bpy.data.worlds.new("dark")
        bpy.context.scene.world.use_nodes = False
        bpy.context.scene.world.color = (0, 0, 0)
        sun = bpy.data.objects.new("sun", bpy.data.lights.new("sun", 'SUN'))
        sun.data.energy = 3.0
        bpy.context.scene.collection.objects.link(sun)   # shines straight down (-Z)
        bpy.ops.mesh.primitive_plane_add(size=2, location=(0, 0, 0))                     # faces the sun
        bpy.ops.mesh.primitive_plane_add(size=2, location=(3, 0, 0), rotation=(3.14159, 0, 0))  # faces away
        blocker_before = len(bpy.data.objects)
        for o in bpy.data.objects:
            o.select_set(o.type == 'MESH')
        mat = material("white", (0.8, 0.8, 0.8))
        # A metal: black without glossy, so the bake treats it as diffuse of the same color.
        next(n for n in mat.node_tree.nodes if n.type == 'BSDF_PRINCIPLED').inputs["Metallic"].default_value = 1.0
        for o in bpy.data.objects:
            if o.type == 'MESH':
                o.data.materials.append(mat)
        _, g = self.convert()
        up = g.positions[:, 0] < 1.5
        self.assertGreater(g.colors[up].mean(), 0.5)       # lit: white albedo x sun
        self.assertLess(g.colors[~up].mean(), 0.05)        # facing away, black world: dark
        self.assertEqual(len([o for o in bpy.data.objects if ".mesh2gs_bake" in o.name]), 0)
        self.assertTrue(all(not o.hide_render for o in bpy.data.objects))  # restored
        self.assertGreaterEqual(len(bpy.data.objects), blocker_before)

    def test_albedo_adds_emission(self):
        bpy.ops.mesh.primitive_plane_add(size=2)
        plane = bpy.context.active_object
        mat = material("glow", (1.0, 0.0, 0.0))
        bsdf = next(n for n in mat.node_tree.nodes if n.type == 'BSDF_PRINCIPLED')
        color = bsdf.inputs.get("Emission Color") or bsdf.inputs["Emission"]
        color.default_value = (0.0, 1.0, 0.0, 1.0)
        bsdf.inputs["Emission Strength"].default_value = 1.0
        plane.data.materials.append(mat)
        s = bpy.context.scene.mesh2gs
        s.bake_resolution = '1024'
        _, g = self.convert()
        np.testing.assert_allclose(np.median(g.colors, axis=0), (1, 1, 0), atol=0.02)  # red + green glow
        s.use_emission = False
        self.addCleanup(setattr, s, "use_emission", True)
        for o in bpy.data.objects:
            o.hide_set(False)
            o.select_set(o == plane)
        _, g = self.convert()
        np.testing.assert_allclose(g.colors, np.tile((1, 0, 0), (len(g), 1)), atol=1e-6)  # Base Color alone

    def test_density_budget(self):
        bpy.ops.mesh.primitive_cube_add(size=2)  # area 24 m^2
        s = bpy.context.scene.mesh2gs
        s.budget_mode, s.density = 'DENSITY', 1000.0
        _, g = self.convert()
        self.assertEqual(len(g), 24_000)

    def test_m0_on_flat_cube(self):
        bpy.ops.mesh.primitive_cube_add(size=2)
        bpy.context.scene.mesh2gs.method = 'M0'
        obj, g = self.convert()
        self.assertEqual(obj.name, "Cube.M0")
        self.assertEqual(len(g), 12)  # one per triangle, like KIRI
        # Flat shading: split normals stop the smoothing, so each Gaussian gets its
        # face normal, turned into a quaternion the way KIRI does (tilted on +-Y faces).
        axis = np.argmax(np.abs(g.positions), axis=1)
        face_normals = np.eye(3)[axis] * np.sign(g.positions[np.arange(12), axis])[:, None]
        from mesh2gs.core.baseline import _normal_to_quaternion
        np.testing.assert_allclose(g.rotations, _normal_to_quaternion(face_normals), atol=1e-5)
        bpy.context.scene.mesh2gs.per_triangle = '6'
        for o in bpy.data.objects:
            o.hide_set(False)
        bpy.data.objects["Cube"].select_set(True)
        _, g = self.convert()
        self.assertEqual(len(g), 72)

    def test_selected_faces_only(self):
        bpy.ops.mesh.primitive_cube_add(size=2)
        cube = bpy.context.active_object
        bpy.ops.object.mode_set(mode='EDIT')
        import bmesh
        bm = bmesh.from_edit_mesh(cube.data)
        for elem in (*bm.verts, *bm.edges, *bm.faces):
            elem.select_set(False)
        for face in bm.faces:
            face.select_set(face.normal.z > 0.5)  # top face only
        bmesh.update_edit_mesh(cube.data)
        bpy.context.scene.mesh2gs.selected_faces_only = True
        self.assertEqual(bpy.ops.mesh2gs.convert(), {'FINISHED'})
        bpy.ops.object.mode_set(mode='OBJECT')
        g = __import__("mesh2gs.preview", fromlist=["x"]).read_object(bpy.data.objects["Cube.M1"])
        np.testing.assert_allclose(g.positions[:, 2], 1.0, atol=1e-6)
        bpy.context.scene.mesh2gs.selected_faces_only = False

    def test_toggle_source_and_moved_result(self):
        bpy.ops.mesh.primitive_plane_add(size=1)
        plane = bpy.context.active_object
        obj, _ = self.convert()
        self.assertTrue(plane.hide_get() and not obj.hide_get())
        bpy.ops.mesh2gs.toggle_source()
        self.assertTrue(obj.hide_get() and not plane.hide_get())
        bpy.ops.mesh2gs.toggle_source()
        obj.location.x = 10
        bpy.context.view_layer.update()
        bpy.ops.mesh2gs.export_ply(filepath=self.path)
        self.assertGreater(ply.read(self.path).positions[:, 0].min(), 9.4)

    def test_polls(self):
        bpy.ops.object.empty_add()
        with self.assertRaises(RuntimeError):
            bpy.ops.mesh2gs.convert()           # nothing to convert
        with self.assertRaises(RuntimeError):
            bpy.ops.mesh2gs.export_ply(filepath=self.path)  # active object is not a result
        bpy.ops.mesh.primitive_cube_add()
        bpy.context.scene.mesh2gs.method = 'M2'
        self.addCleanup(setattr, bpy.context.scene.mesh2gs, "method", 'M1')
        from mesh2gs import training
        dev_python, training.DEV_PYTHON = training.DEV_PYTHON, os.path.join(self.tmp.name, "none.exe")
        try:
            with self.assertRaises(RuntimeError):
                bpy.ops.mesh2gs.convert()       # no Python with gsplat to train in
        finally:
            training.DEV_PYTHON = dev_python

    def test_training_views_leave_scene_untouched(self):
        from mesh2gs.views import INDEX, ViewRenderer
        bpy.ops.mesh.primitive_cube_add(size=2, location=(1, 0, 0))
        cube = bpy.context.active_object
        cube.data.materials.append(material("red", (1.0, 0.0, 0.0)))
        before = {k: len(getattr(bpy.data, k)) for k in ("scenes", "objects", "meshes", "materials", "cameras")}
        r = ViewRenderer([cube], bpy.context.evaluated_depsgraph_get(), self.tmp.name, count=4, size=64)
        try:
            for i in range(r.count):
                r.render(i)
            index = r.write_index()
        finally:
            r.close()
        self.assertEqual({k: len(getattr(bpy.data, k)) for k in before}, before)
        import json
        views = json.load(open(index))["views"]
        self.assertEqual(len(views), 4)
        self.assertEqual(os.path.basename(index), INDEX)
        img = bpy.data.images.load(os.path.join(self.tmp.name, views[0]["image"]))
        px = np.array(img.pixels[:]).reshape(64, 64, 4)
        bpy.data.images.remove(img)
        center = px[24:40, 24:40]
        np.testing.assert_allclose(center[..., :3].mean(axis=(0, 1)), (1, 0, 0), atol=0.02)  # albedo, unlit
        self.assertLess(px[0, 0, 3], 0.01)  # transparent background
        # The camera sees the cube's center at the image center.
        vm, k = np.array(views[0]["viewmat"]), np.array(views[0]["K"])
        p = vm @ [1, 0, 0, 1]
        np.testing.assert_allclose((k @ p[:3])[:2] / p[2], (32, 32), atol=0.5)

    def test_cancel_training_cleans_up(self):
        from mesh2gs import training
        bpy.ops.mesh.primitive_cube_add(size=2)
        cube = bpy.context.active_object
        job = training.TrainJob(bpy.context, 'M2', [cube], 100, 10, bbox=(-1, -1, -1, 1, 1, 1),
                                views=4, size=32, python="none.exe")
        job.step()
        self.assertEqual((job.stage, job.views_done), ("VIEWS", 1))
        job.cancel()
        job.cleanup()
        self.assertEqual(job.stage, "CANCELLED")
        self.assertNotIn("mesh2gs_views", bpy.data.scenes)
        self.assertFalse(os.path.exists(job.dir))
        # Without a usable Python the job fails with a message once the views are done.
        job = training.TrainJob(bpy.context, 'M2', [cube], 100, 10, bbox=(-1, -1, -1, 1, 1, 1),
                                views=2, size=32, python=os.path.join(self.tmp.name, "none.exe"))
        self.assertFalse(job.run())
        self.assertIn("Python", job.error)
        job.cleanup()
        self.assertNotIn("mesh2gs_views", bpy.data.scenes)

    def test_every_interface_string_has_russian(self):
        import re
        import mesh2gs
        from mesh2gs import i18n, training
        # Names that stay as they are in Russian too.
        keep = {"", "Seed", "Mesh → Gaussian Splatting", "Gaussian Splatting (.ply, Mesh2GS)",
                "Gaussian Splatting (.spz, Mesh2GS)", "English", "Русский", "Mesh2GS"}
        strings = set(training.SETUP_STEPS)

        for cls in mesh2gs.classes:
            names, tips = i18n.interface_strings(cls)
            strings |= names | tips
        # Literals the code translates itself, and labels and buttons it lets Blender translate.
        folder = os.path.dirname(mesh2gs.__file__)
        for name in os.listdir(folder):
            if name.endswith(".py"):
                source = open(os.path.join(folder, name), encoding="utf-8").read()
                for literal in re.findall(r'tr\(\s*((?:"[^"]*"\s*)+)\)', source):
                    strings.add("".join(re.findall(r'"([^"]*)"', literal)))
                for literal in re.findall(r'(?:label|operator)\([^)]*?text="([^"{]*)"', source):
                    strings.add(literal)
        strings = {s for s in strings if s not in keep and not s.isdigit()
                   and not re.match(r"^[A-Z]{3} — ", s) or s.startswith("RFU")}
        self.assertGreater(len(strings), 150)  # the collection itself works
        missing = sorted(s for s in strings if s not in i18n.RU)
        self.assertEqual(missing, [])

    def test_language_preference(self):
        from mesh2gs import i18n
        self.addCleanup(i18n.apply, 'AUTO')
        if bpy.app.version < (4, 0, 0):
            # 3.6 translates add-ons only with Preferences → Interface → Translation → Interface on.
            view = bpy.context.preferences.view
            self.addCleanup(setattr, view, "use_translate_interface", view.use_translate_interface)
            self.addCleanup(setattr, view, "use_translate_tooltips", view.use_translate_tooltips)
            view.use_translate_interface = view.use_translate_tooltips = True
        i18n.apply('RU')    # Russian even though Blender runs in English here
        self.assertEqual(i18n.tr("Convert to Gaussians"), "Конвертировать в гауссианы")
        self.assertEqual(i18n.tr("Objects: {}").format(3), "Объектов: 3")
        # Tooltips are looked up in Blender's default context; they are translated too.
        self.assertEqual(bpy.app.translations.pgettext_tip("Bake texture size per object"),
                         "Размер текстуры запекания на объект")
        # Blender's own strings stay as they were: names live in the add-on's context only.
        self.assertEqual(bpy.app.translations.pgettext_iface("Scale"), "Scale")
        self.assertEqual(bpy.app.translations.pgettext_iface("Resolution"), "Resolution")
        rna = bpy.types.Scene.bl_rna.properties["mesh2gs"].fixed_type.properties["method"]
        self.assertEqual(rna.translation_context, i18n.CTX)
        i18n.apply('EN')
        self.assertEqual(i18n.tr("Convert to Gaussians"), "Convert to Gaussians")
        i18n.apply('AUTO')  # follows Blender, which is English
        self.assertEqual(i18n.tr("Objects: {}"), "Objects: {}")

    def test_installer_checks_before_downloading(self):
        from mesh2gs import setup_env
        self.assertEqual(setup_env.interpreter("env", "win32"), os.path.join("env", "python", "python.exe"))
        with self.assertRaises(RuntimeError):
            setup_env.EnvInstaller(self.tmp.name, platform="darwin").start()  # no CUDA on macOS
        needed, setup_env.NEEDED_GB = setup_env.NEEDED_GB, 1e9
        try:
            with self.assertRaises(RuntimeError):  # not enough free space: refused up front
                setup_env.EnvInstaller(os.path.join(self.tmp.name, "env")).start()
        finally:
            setup_env.NEEDED_GB = needed
        self.assertFalse(os.path.exists(os.path.join(self.tmp.name, "env")))
        self.assertEqual(setup_env.EnvInstaller(self.tmp.name).progress, 0.0)

    def test_setup_commands(self):
        from mesh2gs import training
        win, linux = training.setup_commands("win32"), training.setup_commands("linux")
        self.assertTrue(win[0].startswith("py -3.10 -m venv mesh2gs-venv"))
        self.assertIn("win_amd64.whl", win[2])
        self.assertIn("linux_x86_64.whl", linux[2])
        self.assertTrue(all("mesh2gs-venv" in c for c in win + linux))

    def test_light_from_splat_scene(self):
        from mesh2gs import training
        from mesh2gs.core.gaussians import GaussianSet
        if not training.check_python(training.python_path())[0]:
            self.skipTest("нет Python с PyTorch, gsplat и CUDA")
        # A "room": a sphere of splats around the origin, red toward +X, blue toward -X.
        n = 20000
        i = np.arange(n) + 0.5
        z = 1 - 2 * i / n
        r = np.sqrt(1 - z * z)
        phi = i * np.pi * (3 - np.sqrt(5))
        p = np.stack([r * np.cos(phi), r * np.sin(phi), z], 1) * 6.0
        colors = np.where(p[:, :1] > 0, [[1.0, 0.1, 0.1]], [[0.1, 0.1, 1.0]])
        room = GaussianSet(positions=p, scales=np.full((n, 3), 0.25), rotations=np.tile([1.0, 0, 0, 0], (n, 1)),
                           opacities=np.full(n, 0.99), colors=colors)
        room_path = os.path.join(self.tmp.name, "room.ply")
        ply.write(room_path, room)
        self.assertEqual(bpy.ops.mesh2gs.import_ply(filepath=room_path, axis_convention='RFU'), {'FINISHED'})
        room_obj = bpy.context.active_object
        bpy.ops.mesh.primitive_cube_add(size=1)
        bpy.context.active_object.data.materials.append(material("white", (0.8, 0.8, 0.8)))
        s = bpy.context.scene.mesh2gs
        s.light_scene, s.lit_samples, s.bake_resolution = room_obj, 16, '1024'
        self.addCleanup(setattr, s, "color_mode", 'ALBEDO')
        self.assertEqual(bpy.ops.mesh2gs.scene_light(), {'FINISHED'})
        self.assertEqual(s.color_mode, 'LIT')
        image = bpy.context.scene.world.node_tree.nodes["Environment Texture"].image
        self.assertTrue(image.packed_file is not None)
        w, h = image.size
        px = np.array(image.pixels[:]).reshape(h, w, 4)
        np.testing.assert_array_less(px[h // 2, w // 2, 2], px[h // 2, w // 2, 0])   # +X: red
        np.testing.assert_array_less(px[h // 2, 2, 0], px[h // 2, 2, 2])             # -X: blue
        _, g = self.convert()
        east, west = g.positions[:, 0] > 0.49, g.positions[:, 0] < -0.49
        self.assertGreater(g.colors[east, 0].mean(), g.colors[east, 2].mean() + 0.2)
        self.assertGreater(g.colors[west, 2].mean(), g.colors[west, 0].mean() + 0.2)

    def train(self, method):
        from mesh2gs import training
        if not training.check_python(training.python_path())[0]:
            self.skipTest("нет Python с PyTorch, gsplat и CUDA")
        bpy.ops.mesh.primitive_cube_add(size=2)
        bpy.context.active_object.data.materials.append(material("red", (1.0, 0.0, 0.0)))
        s = bpy.context.scene.mesh2gs
        s.method, s.budget = method, 2000
        s.m2_iterations = s.m3_iterations = 60
        s.train_views, s.train_size = 8, '256'
        self.addCleanup(setattr, s, "method", 'M1')
        obj, g = self.convert()
        self.assertEqual(obj.name, f"Cube.{method}")
        self.assertEqual(len(g), 2000)
        self.assertIsNone(training.ACTIVE)
        self.assertNotIn("mesh2gs_views", bpy.data.scenes)
        return g

    def test_m3_refines_m1(self):
        g = self.train('M3')
        # M1 starts on the surface; a short refinement keeps the cube's shape and color.
        self.assertLess(np.abs(np.abs(g.positions).max(axis=1) - 1).mean(), 0.05)
        np.testing.assert_allclose(np.median(g.colors, axis=0), (1, 0, 0), atol=0.1)

    def test_m2_trains_from_random_points(self):
        g = self.train('M2')
        self.assertLess(np.abs(g.positions).max(), 2.0)


if __name__ == "__main__":
    unittest.main()
