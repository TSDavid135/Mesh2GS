"""Bake surface color into a texture with Cycles, on a temporary copy.

Two modes:
  albedo  Base Color (for procedural materials), plus the material's own Emission if
          asked: the copy's materials route both into Emission and an EMIT bake
          records them, unaffected by light or metallic.
  lit     the scene's light as it lands on the surface: a COMBINED bake of diffuse,
          emission and transmission. Glossy is left out on purpose: it depends on
          the viewer, and SH degree 0 Gaussians look the same from every side.
          Metals are baked as diffuse surfaces of the same color: a metal has no
          diffuse part, so without glossy it would come out black, while a diffuse
          stand-in keeps its average brightness under the scene's light.

The user's objects, meshes and materials are never modified: the copy has the
modifiers applied (repeated geometry must not share UVs) and gets a fresh
non-overlapping UV map. While a lit bake runs, the original object and existing
Gaussian previews are hidden from render so they neither occlude nor shadow the
copy; every other object still casts its shadows.
"""

import contextlib

import bpy

from . import materials

BAKE_UV = "mesh2gs_bake"
MARGIN = 4
LIT_PASSES = {'EMIT', 'DIRECT', 'INDIRECT', 'DIFFUSE', 'TRANSMISSION'}


@contextlib.contextmanager
def baked_copy(obj, depsgraph, resolution, lit=False, samples=64, glow=True):
    """Yield (copy, image): a world-space twin of `obj` whose colors live in `image`
    on UV map BAKE_UV. Everything created here is removed on exit. `glow` adds the
    materials' Emission to an albedo bake (a lit bake always has it)."""
    created_objects, created_meshes, created_materials, created_images = [], [], [], []
    view_layer = bpy.context.view_layer
    selected = [o for o in view_layer.objects if o.select_get()]
    active = view_layer.objects.active
    hidden = []
    try:
        mesh = bpy.data.meshes.new_from_object(obj.evaluated_get(depsgraph),
                                               preserve_all_data_layers=True, depsgraph=depsgraph)
        created_meshes.append(mesh)
        copy = bpy.data.objects.new(f"{obj.name}.mesh2gs_bake", mesh)
        created_objects.append(copy)
        bpy.context.scene.collection.objects.link(copy)
        copy.matrix_world = obj.matrix_world

        image = bpy.data.images.new(f"{obj.name}.mesh2gs_bake", resolution, resolution,
                                    alpha=False, float_buffer=True)
        created_images.append(image)
        slots = [s.material for s in obj.material_slots] or [None]
        mesh.materials.clear()
        for mat in slots:
            baked = _lit_copy(mat, image) if lit else _emission_copy(mat, image, glow)
            created_materials.append(baked)
            mesh.materials.append(baked)

        if lit:
            for o in bpy.context.scene.objects:
                if (o == obj or "mesh2gs_method" in o) and not o.hide_render:
                    o.hide_render = True
                    hidden.append(o)
        _unwrap(copy, view_layer)
        _bake(copy, view_layer, lit, samples)
        view_layer.update()
        yield copy, image
    finally:
        for o in hidden:
            o.hide_render = False
        # Selection first: iterating the view layer after removals can hit freed objects.
        for o in view_layer.objects:
            o.select_set(o in selected)
        view_layer.objects.active = active
        for o in created_objects:
            bpy.data.objects.remove(o)
        for m in created_meshes:
            bpy.data.meshes.remove(m)
        for m in created_materials:
            bpy.data.materials.remove(m)
        for i in created_images:
            bpy.data.images.remove(i)


def _target(mat, image):
    """Make `image` the active Image Texture node of `mat`: the bake writes there."""
    target = mat.node_tree.nodes.new('ShaderNodeTexImage')
    target.image = image
    mat.node_tree.nodes.active = target


def _lit_copy(material, image=None):
    """Copy of `material` lit by the scene, metals made diffuse, with `image` as the bake target."""
    if material is None:
        mat = bpy.data.materials.new("mesh2gs_bake")
    else:
        mat = material.copy()
    if not mat.use_nodes:  # a node-less material renders its viewport color as a diffuse BSDF
        color = (*mat.diffuse_color[:3], 1.0)
        mat.use_nodes = True
        bsdf = materials.principled(mat)
        if bsdf is not None:
            bsdf.inputs["Base Color"].default_value = color
    for node in mat.node_tree.nodes:
        if node.type == 'BSDF_PRINCIPLED':
            metallic = node.inputs["Metallic"]
            for link in list(metallic.links):
                mat.node_tree.links.remove(link)
            metallic.default_value = 0.0
    if image is not None:
        _target(mat, image)
    return mat


def _emission_copy(material, image=None, glow=True):
    """Copy of `material` that emits its Base Color (plus its own Emission, with `glow`),
    with `image` (if any) as the bake target."""
    had_nodes = material is not None and material.use_nodes and material.node_tree is not None
    mat = material.copy() if material is not None else bpy.data.materials.new("mesh2gs_bake")
    mat.use_nodes = True
    nodes, links = mat.node_tree.nodes, mat.node_tree.links
    emission = nodes.new('ShaderNodeEmission')
    bsdf = materials.principled(mat) if had_nodes else None
    if bsdf is not None and bsdf.inputs["Base Color"].is_linked:
        links.new(bsdf.inputs["Base Color"].links[0].from_socket, emission.inputs["Color"])
    elif bsdf is not None:
        emission.inputs["Color"].default_value = bsdf.inputs["Base Color"].default_value
    else:
        emission.inputs["Color"].default_value = (*material.diffuse_color[:3], 1.0) if material else (0.8, 0.8, 0.8, 1.0)

    surface = emission.outputs["Emission"]
    if glow and bsdf is not None and materials.is_emissive(bsdf):
        # The material's own glow on top of its Base Color, as it shows without light.
        extra = nodes.new('ShaderNodeEmission')
        for source, target in zip(materials.emission_inputs(bsdf), (extra.inputs["Color"], extra.inputs["Strength"])):
            if source is None:
                continue
            if source.is_linked:
                links.new(source.links[0].from_socket, target)
            else:
                target.default_value = source.default_value
        add = nodes.new('ShaderNodeAddShader')
        links.new(surface, add.inputs[0])
        links.new(extra.outputs["Emission"], add.inputs[1])
        surface = add.outputs["Shader"]

    output = next((n for n in nodes if n.type == 'OUTPUT_MATERIAL' and n.is_active_output), None)
    if output is None:
        output = nodes.new('ShaderNodeOutputMaterial')
    links.new(surface, output.inputs["Surface"])
    if image is not None:
        _target(mat, image)
    return mat


def _only(view_layer, obj):
    for o in view_layer.objects:
        o.select_set(o == obj)
    view_layer.objects.active = obj


def _unwrap(obj, view_layer):
    """Non-overlapping UVs in a new map; the mesh's active UV map stays as it was."""
    uv_layers = obj.data.uv_layers
    previous = uv_layers.active.name if uv_layers.active else None
    uv_layers.active = uv_layers.new(name=BAKE_UV)
    _only(view_layer, obj)
    bpy.ops.object.mode_set(mode='EDIT')
    bpy.ops.mesh.select_all(action='SELECT')
    bpy.ops.uv.smart_project(island_margin=0.002)
    bpy.ops.object.mode_set(mode='OBJECT')
    if previous is not None:
        uv_layers.active = uv_layers[previous]


def _bake(obj, view_layer, lit, samples):
    scene = bpy.context.scene
    engine, old_samples = scene.render.engine, scene.cycles.samples
    try:
        scene.render.engine = 'CYCLES'
        _only(view_layer, obj)
        if lit:
            scene.cycles.samples = samples
            bpy.ops.object.bake(type='COMBINED', pass_filter=LIT_PASSES, uv_layer=BAKE_UV,
                                margin=MARGIN, use_clear=True)
        else:
            scene.cycles.samples = 4  # emission only: a few samples anti-alias procedural edges
            bpy.ops.object.bake(type='EMIT', uv_layer=BAKE_UV, margin=MARGIN, use_clear=True)
    finally:
        scene.render.engine, scene.cycles.samples = engine, old_samples
