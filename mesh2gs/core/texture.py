"""Surface colors: constant per material or sampled from an image by UV.

Everything here is scene-linear RGB; conversion to sRGB happens once, on output.
Image rows follow Blender: row 0 is the bottom, so UV (0, 0) is the first texel.
"""

from dataclasses import dataclass

import numpy as np


@dataclass
class Texture:
    pixels: np.ndarray            # (H, W, 3) scene-linear
    interpolation: str = "Linear"  # Blender Image Texture node: Linear / Closest / Cubic / Smart
    extension: str = "REPEAT"      # REPEAT / EXTEND / CLIP / MIRROR

    def sample(self, uv, mode="auto"):
        """(N, 2) UVs -> (N, 3) colors.

        mode "auto"     bilinear, or nearest texel if the node says Closest
             "nearest"  nearest texel
             "kiri"     KIRI Mesh 2 3DGS rule: round(uv * (size - 1)), clamped. Unlike the
                        converter (which clamps any UV into the image), a REPEAT texture is
                        wrapped first, so tiled UVs such as glTF's v in [-1, 0] stay correct.
        """
        h, w, _ = self.pixels.shape
        if mode == "kiri":
            if self.extension == "REPEAT":  # only UVs outside [0, 1]: 1.0 must stay the last texel
                uv = np.where((uv < 0) | (uv > 1), uv - np.floor(uv), uv)
            xi = np.clip(np.round(uv[:, 0] * (w - 1)).astype(np.int64), 0, w - 1)
            yi = np.clip(np.round(uv[:, 1] * (h - 1)).astype(np.int64), 0, h - 1)
            return self.pixels[yi, xi]
        x = uv[:, 0] * w - 0.5
        y = uv[:, 1] * h - 0.5
        if mode == "nearest" or self.interpolation == "Closest":
            xi, yi = np.floor(x + 0.5).astype(np.int64), np.floor(y + 0.5).astype(np.int64)
            return self._fetch(xi, yi, w, h)
        x0, y0 = np.floor(x).astype(np.int64), np.floor(y).astype(np.int64)
        fx, fy = (x - x0)[:, None], (y - y0)[:, None]
        c00 = self._fetch(x0, y0, w, h)
        c10 = self._fetch(x0 + 1, y0, w, h)
        c01 = self._fetch(x0, y0 + 1, w, h)
        c11 = self._fetch(x0 + 1, y0 + 1, w, h)
        return (c00 * (1 - fx) + c10 * fx) * (1 - fy) + (c01 * (1 - fx) + c11 * fx) * fy

    def _fetch(self, xi, yi, w, h):
        if self.extension == "REPEAT":
            xi, yi = xi % w, yi % h
        elif self.extension == "MIRROR":
            xi, yi = _mirror(xi, w), _mirror(yi, h)
        else:  # EXTEND; CLIP is treated like EXTEND (no transparency support)
            xi, yi = np.clip(xi, 0, w - 1), np.clip(yi, 0, h - 1)
        return self.pixels[yi, xi]


def _mirror(i, n):
    period = 2 * n
    i = i % period
    return np.where(i < n, i, period - 1 - i)


class ColorField:
    """Color over a triangle mesh: each triangle takes one source, a constant or a texture.

    tri_source   (T,) index into `sources`
    sources      list of 3-tuples (constant linear RGB) or Texture
    corner_uvs   (T, 3, 2) UVs of the map each triangle's texture uses (zeros when unused)
    """

    def __init__(self, tri_source, sources, corner_uvs=None):
        self.tri_source = np.asarray(tri_source, dtype=np.int64)
        self.sources = list(sources)
        t = len(self.tri_source)
        self.corner_uvs = np.zeros((t, 3, 2)) if corner_uvs is None else np.asarray(corner_uvs, dtype=np.float64)

    @classmethod
    def constant(cls, triangle_colors):
        colors = np.asarray(triangle_colors, dtype=np.float64)
        return cls(np.arange(len(colors)), [tuple(c) for c in colors])

    def subset(self, keep):
        return ColorField(self.tri_source[keep], self.sources, self.corner_uvs[keep])

    @property
    def textured(self):
        return any(isinstance(s, Texture) for s in self.sources)

    def sample(self, tri, bary, mode="auto"):
        """Colors at points given by triangle index (N,) and barycentric weights (N, 3)."""
        tri = np.asarray(tri)
        out = np.empty((len(tri), 3))
        source = self.tri_source[tri]
        for i, src in enumerate(self.sources):
            mask = source == i
            if not mask.any():
                continue
            if isinstance(src, Texture):
                uv = np.einsum("nk,nkd->nd", bary[mask], self.corner_uvs[tri[mask]])
                out[mask] = src.sample(uv, mode=mode)
            else:
                out[mask] = src
        return out

    def corner_colors(self, mode="kiri"):
        """(T, 3, 3) colors at the triangle corners (what vertex-based schemes see)."""
        t = len(self.tri_source)
        tri = np.repeat(np.arange(t), 3)
        bary = np.tile(np.eye(3), (t, 1))
        return self.sample(tri, bary, mode=mode).reshape(t, 3, 3)
