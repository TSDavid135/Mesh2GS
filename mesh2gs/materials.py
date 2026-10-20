"""Where a material's Base Color comes from: a constant, an image by UV, or unsupported."""

from dataclasses import dataclass

import numpy as np

from .core.color import srgb_to_linear
from .core.texture import Texture
from .i18n import tr

DEFAULT_COLOR = (0.8, 0.8, 0.8)


@dataclass
class SlotColor:
    kind: str                 # 'CONSTANT', 'TEXTURE' or 'UNSUPPORTED'
    color: tuple              # scene-linear RGB; the fallback when not 'CONSTANT'
    image: object = None      # bpy.types.Image for 'TEXTURE'
    interpolation: str = "Linear"
    extension: str = "REPEAT"
    uv_map: str = ""          # "" = the mesh's render UV map
    note: str = ""            # shown in the UI
    bakeable: bool = False    # 'UNSUPPORTED' whose Base Color a Cycles bake can capture
    emissive: bool = False    # the material glows (Emission); only a bake adds that to the color


def principled(material):
    if material is None or not material.use_nodes or material.node_tree is None:
        return None
    output = next((n for n in material.node_tree.nodes
                   if n.type == 'OUTPUT_MATERIAL' and n.is_active_output), None)
    if output is not None and output.inputs["Surface"].is_linked:
        node = _upstream(output.inputs["Surface"])
        if node is not None and node.type == 'BSDF_PRINCIPLED':
            return node
    return next((n for n in material.node_tree.nodes if n.type == 'BSDF_PRINCIPLED'), None)


def _upstream(socket):
    """Node feeding `socket`, skipping reroutes."""
    node = socket.links[0].from_node if socket.is_linked else None
    while node is not None and node.type == 'REROUTE':
        node = node.inputs[0].links[0].from_node if node.inputs[0].is_linked else None
    return node


def emission_inputs(bsdf):
    """(color, strength) sockets of a Principled BSDF's emission; the color input is
    "Emission Color" since Blender 4.0 and "Emission" before."""
    color = bsdf.inputs.get("Emission Color") or bsdf.inputs.get("Emission")
    return color, bsdf.inputs.get("Emission Strength")


def is_emissive(bsdf):
    color, strength = emission_inputs(bsdf)
    if color is None:
        return False
    glows = color.is_linked or max(color.default_value[:3]) > 0
    return glows and (strength is None or strength.is_linked or strength.default_value > 0)


def slot_color(material):
    """Where the Base Color comes from; `emissive` is set on top of that."""
    info = _base_color(material)
    bsdf = principled(material)
    info.emissive = bsdf is not None and is_emissive(bsdf)
    return info


def _base_color(material):
    if material is None:
        return SlotColor('CONSTANT', DEFAULT_COLOR, note=tr("no material"))
    fallback = tuple(material.diffuse_color[:3])
    bsdf = principled(material)
    if bsdf is None:
        return SlotColor('UNSUPPORTED', fallback, note=tr("no Principled BSDF"))
    base = bsdf.inputs["Base Color"]
    if not base.is_linked:
        return SlotColor('CONSTANT', tuple(base.default_value[:3]))

    node = _upstream(base)
    if node is None or node.type != 'TEX_IMAGE' or node.image is None:
        return SlotColor('UNSUPPORTED', fallback, note=tr("procedural color"), bakeable=True)
    uv_map = ""
    if node.inputs["Vector"].is_linked:
        source = _upstream(node.inputs["Vector"])
        if source is None or source.type != 'UVMAP':
            return SlotColor('UNSUPPORTED', fallback, note=tr("texture through Mapping"), bakeable=True)
        uv_map = source.uv_map
    if node.image.size[0] == 0 or node.image.size[1] == 0:
        return SlotColor('UNSUPPORTED', fallback, note=tr("image “{}” is not loaded").format(node.image.name))
    return SlotColor('TEXTURE', fallback, image=node.image, interpolation=node.interpolation,
                     extension=node.extension, uv_map=uv_map, note=node.image.name)


def slot_colors(obj):
    """One SlotColor per material slot, indexable by a triangle's material_index."""
    colors = [slot_color(slot.material) for slot in obj.material_slots]
    return colors or [slot_color(None)]


def image_texture(image, interpolation, extension, cache):
    """Scene-linear Texture of a Blender image; `cache` dedups pixel reads per image."""
    pixels = cache.get(image.name)
    if pixels is None:
        w, h = image.size
        raw = np.empty(w * h * 4, dtype=np.float32)
        image.pixels.foreach_get(raw)
        rgb = raw.reshape(h, w, 4)[:, :, :3]
        space = image.colorspace_settings.name.lower()
        # Float buffers are already linear; byte buffers hold values in their color space.
        if not image.is_float and "linear" not in space and space != "non-color":
            rgb = srgb_to_linear(rgb).astype(np.float32)
        pixels = cache[image.name] = np.ascontiguousarray(rgb)
    return Texture(pixels, interpolation=interpolation, extension=extension)
