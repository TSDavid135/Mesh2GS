"""Gaussian set <-> Blender object.

The converted set lives on a point-only mesh as attributes, so it is saved with
the .blend and exported later without recomputing. A Geometry Nodes modifier
draws every Gaussian as a flat colored disc for the viewport.
"""

import bpy
import numpy as np

from .core.color import srgb_to_linear
from .core.gaussians import GaussianSet

COLLECTION = "Gaussians"
NODE_GROUP = "Mesh2GS Preview"
MATERIAL = "Mesh2GS Splat"
TAG = "mesh2gs_method"  # custom property marking a Gaussian object
MODERN_NODES = bpy.app.version >= (4, 0, 0)


def is_gaussian_object(obj):
    return obj is not None and obj.type == 'MESH' and TAG in obj


# ---- storage ---------------------------------------------------------------

def _set_attr(mesh, name, kind, values):
    attr = mesh.attributes.get(name) or mesh.attributes.new(name, kind, 'POINT')
    field = {"FLOAT": "value", "FLOAT_VECTOR": "vector", "FLOAT_COLOR": "color"}[kind]
    attr.data.foreach_set(field, np.ascontiguousarray(values, dtype=np.float32).ravel())


def _get_attr(mesh, name, width):
    attr = mesh.attributes[name]
    field = {1: "value", 3: "vector", 4: "color"}[width]
    out = np.empty(len(attr.data) * width, dtype=np.float32)
    attr.data.foreach_get(field, out)
    return out.reshape(-1, width) if width > 1 else out


def quaternion_to_euler(q):
    """(N, 4) w-first quaternions -> (N, 3) Blender XYZ Euler angles."""
    q = q / np.linalg.norm(q, axis=1, keepdims=True)
    w, x, y, z = q.T
    r00 = 1 - 2 * (y * y + z * z)
    r10 = 2 * (x * y + w * z)
    r20 = 2 * (x * z - w * y)
    r21 = 2 * (y * z + w * x)
    r22 = 1 - 2 * (x * x + y * y)
    return np.stack([np.arctan2(r21, r22), np.arcsin(np.clip(-r20, -1, 1)), np.arctan2(r10, r00)], axis=1)


def write_object(name, gaussians: GaussianSet, method, seconds, preview_limit):
    """Create or replace the Gaussian object `name`; returns it."""
    g = gaussians
    mesh = bpy.data.meshes.new(name)
    mesh.vertices.add(len(g))
    mesh.vertices.foreach_set("co", g.positions.ravel())
    _set_attr(mesh, "gs_scale", "FLOAT_VECTOR", g.scales)
    _set_attr(mesh, "gs_rot_w", "FLOAT", g.rotations[:, 0])
    _set_attr(mesh, "gs_rot_xyz", "FLOAT_VECTOR", g.rotations[:, 1:])
    _set_attr(mesh, "gs_opacity", "FLOAT", g.opacities)
    _set_attr(mesh, "gs_color", "FLOAT_VECTOR", g.colors)  # sRGB, exported as is
    display = np.concatenate([srgb_to_linear(g.colors), np.ones((len(g), 1))], axis=1)
    _set_attr(mesh, "gs_display", "FLOAT_COLOR", display)
    # Active color attribute: Solid mode with "Attribute" coloring shows the splat colors.
    if hasattr(mesh.color_attributes, "active_color_name"):
        mesh.color_attributes.active_color_name = "gs_display"
    else:
        mesh.color_attributes.active_color = mesh.color_attributes["gs_display"]
    _set_attr(mesh, "gs_euler", "FLOAT_VECTOR", quaternion_to_euler(g.rotations.astype(np.float64)))
    mesh.update()

    obj = bpy.data.objects.get(name)
    if obj is not None and obj.type == 'MESH':
        old = obj.data
        obj.data = mesh
        bpy.data.meshes.remove(old)
    else:
        obj = bpy.data.objects.new(name, mesh)
        _collection().objects.link(obj)
    obj[TAG] = method
    obj["mesh2gs_count"] = len(g)
    obj["mesh2gs_seconds"] = float(seconds)
    _attach_preview(obj, preview_limit)
    return obj


def read_object(obj) -> GaussianSet:
    """The object's Gaussians in world space (the object may have been moved)."""
    mesh = obj.data
    n = len(mesh.vertices)
    positions = np.empty(n * 3, dtype=np.float32)
    mesh.vertices.foreach_get("co", positions)
    positions = positions.reshape(n, 3).astype(np.float64)
    scales = _get_attr(mesh, "gs_scale", 3).astype(np.float64)
    rotations = np.concatenate([_get_attr(mesh, "gs_rot_w", 1)[:, None],
                                _get_attr(mesh, "gs_rot_xyz", 3)], axis=1).astype(np.float64)

    loc, rot, scale = obj.matrix_world.decompose()
    if not obj.matrix_world.is_identity:
        world = np.array(obj.matrix_world, dtype=np.float64)
        positions = positions @ world[:3, :3].T + world[:3, 3]
        scales = scales * float(sum(scale) / 3)  # non-uniform object scale is not representable
        rotations = _quat_multiply(np.array([rot.w, rot.x, rot.y, rot.z]), rotations)
    return GaussianSet(
        positions=positions,
        scales=scales,
        rotations=rotations,
        opacities=_get_attr(mesh, "gs_opacity", 1),
        colors=_get_attr(mesh, "gs_color", 3),
    )


def _quat_multiply(a, b):
    aw, ax, ay, az = a
    bw, bx, by, bz = b.T
    return np.stack([
        aw * bw - ax * bx - ay * by - az * bz,
        aw * bx + ax * bw + ay * bz - az * by,
        aw * by - ax * bz + ay * bw + az * bx,
        aw * bz + ax * by - ay * bx + az * bw,
    ], axis=1)


def _collection():
    coll = bpy.data.collections.get(COLLECTION)
    if coll is None:
        coll = bpy.data.collections.new(COLLECTION)
    if coll.name not in bpy.context.scene.collection.children:
        bpy.context.scene.collection.children.link(coll)
    return coll


# ---- viewport preview ------------------------------------------------------

def _attach_preview(obj, preview_limit):
    group = _node_group()
    mod = obj.modifiers.get("Mesh2GS") or obj.modifiers.new("Mesh2GS", 'NODES')
    mod.node_group = group
    fraction = min(1.0, preview_limit / max(len(obj.data.vertices), 1))
    mod[_fraction_identifier(group)] = fraction


def set_preview_fraction(obj, preview_limit):
    mod = obj.modifiers.get("Mesh2GS")
    if mod is not None and mod.node_group is not None:
        mod[_fraction_identifier(mod.node_group)] = min(1.0, preview_limit / max(len(obj.data.vertices), 1))
        obj.update_tag()


def _fraction_identifier(group):
    """The group's float input, the preview fraction; found by type, as groups saved by
    older versions name it in Russian."""
    if MODERN_NODES:
        return next(s.identifier for s in group.interface.items_tree
                    if s.item_type == 'SOCKET' and s.in_out == 'INPUT' and s.socket_type == 'NodeSocketFloat')
    return next(s.identifier for s in group.inputs if s.type == 'VALUE')


def _attribute_output(node):
    return next(o for o in node.outputs if o.name == "Attribute" and o.enabled)


def _node_group():
    group = bpy.data.node_groups.get(NODE_GROUP)
    if group is not None:
        return group
    group = bpy.data.node_groups.new(NODE_GROUP, 'GeometryNodeTree')
    if MODERN_NODES:
        group.interface.new_socket("Geometry", in_out='INPUT', socket_type='NodeSocketGeometry')
        fraction = group.interface.new_socket("Preview fraction", in_out='INPUT', socket_type='NodeSocketFloat')
        group.interface.new_socket("Geometry", in_out='OUTPUT', socket_type='NodeSocketGeometry')
    else:
        group.inputs.new('NodeSocketGeometry', "Geometry")
        fraction = group.inputs.new('NodeSocketFloat', "Preview fraction")
        group.outputs.new('NodeSocketGeometry', "Geometry")
    fraction.default_value, fraction.min_value, fraction.max_value = 1.0, 0.0, 1.0

    nodes, links = group.nodes, group.links
    inp = nodes.new('NodeGroupInput')
    out = nodes.new('NodeGroupOutput')
    keep = nodes.new('FunctionNodeRandomValue')
    keep.data_type = 'BOOLEAN'
    circle = nodes.new('GeometryNodeMeshCircle')
    circle.fill_type = 'NGON'
    circle.inputs["Vertices"].default_value = 8
    # A splat renderer shows a Gaussian out to roughly 1.5 sigma (alpha ~0.3 there),
    # so an opaque disc of that radius approximates its footprint.
    circle.inputs["Radius"].default_value = 1.5
    set_mat = nodes.new('GeometryNodeSetMaterial')
    set_mat.inputs["Material"].default_value = _material()
    rot = nodes.new('GeometryNodeInputNamedAttribute')
    rot.data_type = 'FLOAT_VECTOR'
    rot.inputs["Name"].default_value = "gs_euler"
    scale = nodes.new('GeometryNodeInputNamedAttribute')
    scale.data_type = 'FLOAT_VECTOR'
    scale.inputs["Name"].default_value = "gs_scale"
    instance = nodes.new('GeometryNodeInstanceOnPoints')
    # Realized discs carry the point attributes as plain geometry attributes, which
    # every engine and the Solid mode read reliably (instancer lookups do not).
    realize = nodes.new('GeometryNodeRealizeInstances')

    links.new(inp.outputs[1], keep.inputs["Probability"])
    links.new(inp.outputs[0], instance.inputs["Points"])
    links.new(_bool_output(keep), instance.inputs["Selection"])
    links.new(circle.outputs["Mesh"], set_mat.inputs["Geometry"])
    links.new(set_mat.outputs["Geometry"], instance.inputs["Instance"])
    links.new(_attribute_output(rot), instance.inputs["Rotation"])
    links.new(_attribute_output(scale), instance.inputs["Scale"])
    links.new(instance.outputs["Instances"], realize.inputs["Geometry"])
    links.new(realize.outputs["Geometry"], out.inputs[0])

    for i, node in enumerate((inp, keep, circle, set_mat, rot, scale, instance, realize, out)):
        node.location = (200 * i, 0)
    return group


def _bool_output(node):
    return next(o for o in node.outputs if o.enabled and o.type == 'BOOLEAN')


def _material():
    mat = bpy.data.materials.get(MATERIAL)
    if mat is not None:
        return mat
    mat = bpy.data.materials.new(MATERIAL)
    mat.use_nodes = True
    nodes, links = mat.node_tree.nodes, mat.node_tree.links
    nodes.clear()
    attr = nodes.new('ShaderNodeAttribute')
    attr.attribute_type = 'GEOMETRY'
    attr.attribute_name = "gs_display"
    emit = nodes.new('ShaderNodeEmission')
    out = nodes.new('ShaderNodeOutputMaterial')
    links.new(attr.outputs["Color"], emit.inputs["Color"])
    links.new(emit.outputs["Emission"], out.inputs["Surface"])
    return mat
