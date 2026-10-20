"""Color space helpers.

Blender stores material colors scene-linear; 3DGS renderers treat the SH DC term as
display sRGB (the original method is trained on sRGB photos), so we convert on output.
Both functions keep float32 input as float32 (textures can be large).
"""

import numpy as np


def _as_float(c):
    c = np.asarray(c)
    return c if np.issubdtype(c.dtype, np.floating) else c.astype(np.float64)


def linear_to_srgb(c):
    c = np.clip(_as_float(c), 0.0, 1.0)
    return np.where(c <= 0.0031308, c * 12.92, 1.055 * np.power(c, 1.0 / 2.4) - 0.055)


def srgb_to_linear(c):
    c = np.clip(_as_float(c), 0.0, 1.0)
    return np.where(c <= 0.04045, c / 12.92, np.power((c + 0.055) / 1.055, 2.4))
