"""Evaluated Blender meshes -> NumPy arrays in world space."""

from dataclasses import dataclass, field
from typing import Optional

import bpy
import numpy as np

from . import bake, materials
from .i18n import tr
from .core.texture import ColorField


@dataclass
class SceneMesh:
    vertices: np.ndarray                 # (V, 3) world space
    triangles: np.ndarray                # (T, 3) vertex indices
    colors: ColorField                   # color source per triangle
    corner_normals: np.ndarray           # (T, 3, 3) world-space split normals
    corner_uvs: Optional[np.ndarray]     # (T, 3, 2) active UV map, None if no mesh has one
    warnings: list = field(default_factory=list)


def _get(collection, attr, count, dtype, width=1):
    out = np.empty(count * width, dtype=dtype)
    collection.foreach_get(attr, out)
    return out.reshape(count, width) if width > 1 else out


def _uv_layer(mesh, name):
    """Named UV map, or the one Image Texture nodes use by default (render-active)."""
    if name:
        return mesh.uv_layers.get(name)
    return next((l for l in mesh.uv_layers if l.active_render), mesh.uv_layers.active)


def scene_mesh(objects, depsgraph, selected_faces_only=False, bake_resolution=0, lit=False, lit_samples=64,
               emission=True):
    """Merge mesh objects (modifiers and transforms applied) into one triangle soup.

    bake_resolution > 0 bakes procedural Base Colors (see bake.py) at that size, and,
    with `emission`, any glowing material's Emission on top of its Base Color; 0 keeps
    the viewport-color fallback and no glow. lit=True bakes every object with the
    scene's lighting instead (needs bake_resolution > 0).
    """
    parts, sources, warnings = [], [], []
    pixel_cache = {}
    offset = 0
    for obj in objects:
        slots = materials.slot_colors(obj)
        if bake_resolution and (lit or any(s.bakeable or (emission and s.emissive) for s in slots)):
            with bake.baked_copy(obj, depsgraph, bake_resolution, lit=lit, samples=lit_samples,
                                 glow=emission) as (copy, image):
                baked = materials.SlotColor('TEXTURE', materials.DEFAULT_COLOR, image=image,
                                            extension="EXTEND", uv_map=bake.BAKE_UV, note=image.name)
                evaluated = copy.evaluated_get(bpy.context.evaluated_depsgraph_get())
                part = _object_part(obj.name, evaluated, [baked] * len(slots),
                                    selected_faces_only, sources, warnings, pixel_cache)
        else:
            part = _object_part(obj.name, obj.evaluated_get(depsgraph), slots,
                                selected_faces_only, sources, warnings, pixel_cache)
        part["triangles"] += offset
        offset += len(part["vertices"])
        parts.append(part)

    # Parts without triangles stay: their vertices keep the index offsets valid.
    if not any(len(p["triangles"]) for p in parts):
        raise ValueError(tr("nothing to convert: no triangles"))
    has_uv = any(p["weld_uvs"] is not None for p in parts)
    return SceneMesh(
        vertices=np.concatenate([p["vertices"] for p in parts]),
        triangles=np.concatenate([p["triangles"] for p in parts]),
        colors=ColorField(np.concatenate([p["tri_source"] for p in parts]), sources,
                          np.concatenate([p["color_uvs"] for p in parts])),
        corner_normals=np.concatenate([p["corner_normals"] for p in parts]),
        corner_uvs=np.concatenate([
            p["weld_uvs"] if p["weld_uvs"] is not None else np.zeros((len(p["triangles"]), 3, 2))
            for p in parts]) if has_uv else None,
        warnings=list(dict.fromkeys(warnings)),
    )


def _object_part(name, evaluated, slots, selected_faces_only, sources, warnings, pixel_cache):
    """One evaluated object as world-space arrays; appends its color sources to `sources`."""
    mesh = evaluated.to_mesh()
    try:
        if hasattr(mesh, "calc_normals_split"):  # Blender < 4.1 computes them on demand
            mesh.calc_normals_split()
        mesh.calc_loop_triangles()
        tris = mesh.loop_triangles
        count = len(tris)
        co = _get(mesh.vertices, "co", len(mesh.vertices), np.float32, 3)
        tri_vertices = _get(tris, "vertices", count, np.int32, 3)
        normals = _get(tris, "split_normals", count, np.float32, 9).reshape(count, 3, 3)
        mat_index = np.clip(_get(tris, "material_index", count, np.int32), 0, len(slots) - 1)
        loops = _get(tris, "loops", count, np.int32, 3)

        def corner_uv(layer):
            uv = _get(layer.data, "uv", len(mesh.loops), np.float32, 2)
            return uv[loops].astype(np.float64)

        active = mesh.uv_layers.active
        weld_uvs = corner_uv(active) if active is not None else None

        # One color source per slot; textured slots read UVs from their own map.
        tri_source = mat_index + len(sources)
        color_uvs = np.zeros((count, 3, 2))
        for i, slot in enumerate(slots):
            used = mat_index == i
            if slot.kind == 'TEXTURE':
                layer = _uv_layer(mesh, slot.uv_map)
                if layer is None:
                    warnings.append(tr("{}: the mesh has no UV map for “{}”, the viewport color is used")
                                    .format(name, slot.note))
                    sources.append(slot.color)
                    continue
                if used.any():
                    color_uvs[used] = corner_uv(layer)[used]
                sources.append(materials.image_texture(
                    slot.image, slot.interpolation, slot.extension, pixel_cache))
            else:
                if slot.kind == 'UNSUPPORTED' and used.any():
                    warnings.append(tr("{}: {}, the viewport color is used").format(name, slot.note))
                sources.append(slot.color)

        keep = np.ones(count, dtype=bool)
        if selected_faces_only:
            poly_select = _get(mesh.polygons, "select", len(mesh.polygons), bool)
            keep = poly_select[_get(tris, "polygon_index", count, np.int32)]
    finally:
        evaluated.to_mesh_clear()

    world = np.array(evaluated.matrix_world, dtype=np.float64)
    normal_matrix = np.linalg.inv(world[:3, :3]).T
    normals = normals.astype(np.float64) @ normal_matrix.T
    normals /= np.maximum(np.linalg.norm(normals, axis=2, keepdims=True), 1e-12)
    return dict(
        vertices=co.astype(np.float64) @ world[:3, :3].T + world[:3, 3],
        triangles=tri_vertices[keep].astype(np.int64),
        tri_source=tri_source[keep],
        color_uvs=color_uvs[keep],
        corner_normals=normals[keep],
        weld_uvs=None if weld_uvs is None else weld_uvs[keep],
    )
