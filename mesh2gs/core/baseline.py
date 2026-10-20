"""M0: the scheme KIRI Mesh 2 3DGS ships (GauStudio mesh initializer, SuGaR layout).

Written from the SuGaR paper and from observing the KIRI converter's output, to
compare against on the same inputs and OS. It reproduces the converter's
behavior, quirks included, with two deliberate differences: opacity is finite
(KIRI writes logit(1) = +inf), and UVs outside [0, 1] wrap on repeating textures
(KIRI clamps them to the image border, which paints tiled or glTF-imported models
with one row of texels).
"""

import numpy as np

from .color import linear_to_srgb
from .gaussians import GaussianSet

# Barycentric layouts and disc radii (relative to the shortest edge) per SuGaR.
LAYOUTS = {
    1: ([[1 / 3, 1 / 3, 1 / 3]], 0.5 / np.sqrt(3.0)),
    3: ([[1 / 2, 1 / 4, 1 / 4], [1 / 4, 1 / 2, 1 / 4], [1 / 4, 1 / 4, 1 / 2]],
        0.5 / (np.sqrt(3.0) + 1.0)),
    4: ([[1 / 3, 1 / 3, 1 / 3], [2 / 3, 1 / 6, 1 / 6], [1 / 6, 2 / 3, 1 / 6], [1 / 6, 1 / 6, 2 / 3]],
        1.0 / (4.0 * np.sqrt(3.0))),
    6: ([[2 / 3, 1 / 6, 1 / 6], [1 / 6, 2 / 3, 1 / 6], [1 / 6, 1 / 6, 2 / 3],
         [1 / 6, 5 / 12, 5 / 12], [5 / 12, 1 / 6, 5 / 12], [5 / 12, 5 / 12, 1 / 6]],
        1.0 / (4.0 + 2.0 * np.sqrt(3.0))),
}
KIRI_PER_TRIANGLE = 1  # what the shipped converter actually builds
THICKNESS = 1e-7


def convert(vertices, triangles, corner_colors, per_triangle=KIRI_PER_TRIANGLE, *, opacity=0.99):
    """Fixed Gaussians per triangle.

    corner_colors  (T, 3, 3) scene-linear RGB at each triangle corner
    per_triangle   1, 3, 4 or 6
    """
    if per_triangle not in LAYOUTS:
        raise ValueError(f"per_triangle must be one of {sorted(LAYOUTS)}")
    bary, radius = LAYOUTS[per_triangle]
    bary = np.asarray(bary)                      # (n, 3)
    vertices = np.asarray(vertices, dtype=np.float64)
    triangles = np.asarray(triangles, dtype=np.int64)
    if len(triangles) == 0:
        raise ValueError("mesh has no triangles")

    corners = vertices[triangles]                # (T, 3, 3)
    positions = np.einsum("nk,tkd->tnd", bary, corners).reshape(-1, 3)

    normals = _normalize(np.einsum("nk,tkd->tnd", bary, vertex_normals(vertices, triangles)[triangles]))
    rotations = _normal_to_quaternion(normals.reshape(-1, 3))

    edges = np.linalg.norm(corners - np.roll(corners, -1, axis=1), axis=2).min(axis=1)
    disc = np.repeat(2.0 * radius * np.maximum(edges, 0.0), per_triangle)
    scales = np.stack([disc, disc, np.full_like(disc, THICKNESS)], axis=1)
    scales = np.maximum(scales, THICKNESS)       # KIRI stores log(2s + 1e-7)

    # Colors are averaged in sRGB, as the converter reads texture bytes directly.
    srgb = linear_to_srgb(np.asarray(corner_colors, dtype=np.float64))
    vertex_color = np.zeros_like(vertices)
    np.add.at(vertex_color, triangles.ravel(), srgb.reshape(-1, 3))
    counts = np.bincount(triangles.ravel(), minlength=len(vertices))
    vertex_color /= np.maximum(counts, 1)[:, None]
    colors = np.einsum("nk,tkd->tnd", bary, vertex_color[triangles]).reshape(-1, 3)

    return GaussianSet(
        positions=positions,
        scales=scales,
        rotations=rotations,
        opacities=np.full(len(positions), opacity),
        colors=np.clip(colors, 0.0, 1.0),
    )


def weld(vertices, triangles, corner_normals, corner_uvs=None, decimals=5):
    """Split vertices the way the converter sees an exported OBJ.

    Open3D keeps one vertex per distinct (position, uv, normal) index triple, so
    smoothing stops at UV seams and hard edges. Returns (vertices, triangles).
    """
    keys = [triangles.reshape(-1, 1).astype(np.float64),
            np.round(corner_normals.reshape(-1, 3), decimals)]
    if corner_uvs is not None:
        keys.append(np.round(corner_uvs.reshape(-1, 2), decimals))
    _, first, inverse = np.unique(np.hstack(keys), axis=0, return_index=True, return_inverse=True)
    source = triangles.reshape(-1)[first]
    return vertices[source], inverse.reshape(-1, 3)


def vertex_normals(vertices, triangles):
    """Area-weighted vertex normals over shared vertices (smooth across hard edges)."""
    v0, v1, v2 = (vertices[triangles[:, i]] for i in range(3))
    face = np.cross(v1 - v0, v2 - v0)
    acc = np.zeros_like(vertices)
    for i in range(3):
        np.add.at(acc, triangles[:, i], face)
    return _normalize(acc)


def _normalize(v, eps=1e-12):
    return v / np.maximum(np.linalg.norm(v, axis=-1, keepdims=True), eps)


def _normal_to_quaternion(n):
    """GauStudio's normal2rotation + rotmat2quaternion, sign flips included."""
    r0 = np.array([1.0, 0.0, 0.0]) - (n[:, :1] * n)          # (1,0,0) minus its normal part
    r0 = _normalize(r0 * np.sign(r0[:, :1]))
    r1 = np.cross(n, r0)
    r1 = r1 * (np.sign(r1[:, 1:2]) * np.sign(n[:, 2:3]))
    m = np.stack([r0, r1, n], axis=-1)                         # columns
    r = np.sqrt(1.0 + m[:, 0, 0] + m[:, 1, 1] + m[:, 2, 2] + 1e-6) / 2.0
    q = np.stack([
        r,
        (m[:, 2, 1] - m[:, 1, 2]) / (4 * r),
        (m[:, 0, 2] - m[:, 2, 0]) / (4 * r),
        (m[:, 1, 0] - m[:, 0, 1]) / (4 * r),
    ], axis=1)
    return q  # not normalized, exactly like the converter; renderers normalize
