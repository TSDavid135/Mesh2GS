"""M1: analytic mesh -> Gaussians, no training.

Points are spread over the surface by area, so the result depends on the surface
and the budget, not on how the mesh happens to be triangulated.
"""

import numpy as np

from .color import linear_to_srgb
from .gaussians import GaussianSet
from .texture import ColorField

# Triangles smaller than this (m^2) carry no area and would give undefined tangents.
MIN_AREA = 1e-12


SAMPLINGS = ("stratified", "random")
GOLDEN = (np.sqrt(5.0) - 1.0) / 2.0
# Disc sigma per sample spacing. Tuned on the benchmark (E2): 0.7 beats 0.5 and 1.0 on all
# models tested; larger discs blur silhouettes and overhang, smaller ones leave gaps.
DEFAULT_OVERLAP = 0.7


def convert(vertices, triangles, colors, budget, *,
            overlap=DEFAULT_OVERLAP, flatness=0.05, opacity=0.99, seed=0, sampling="stratified",
            edge_aware=True, sharp_angle=30.0, edge_margin=1.5):
    """Sample `budget` flat Gaussians on a triangle mesh.

    vertices     (V, 3) world coordinates
    triangles    (T, 3) vertex indices
    colors       ColorField, or (T, 3) scene-linear RGB per triangle; sampled at every point
    overlap      disc radius relative to the mean spacing between samples (k)
    flatness     thickness along the normal relative to the disc radius
    opacity      alpha of every Gaussian; kept below 1 so the stored logit is finite
    sampling     "stratified" (even spacing, the default) or "random" (i.i.d., for comparison)
    edge_aware   squash discs near feature edges so they do not overhang them
    sharp_angle  dihedral angle (degrees) above which an edge is a feature edge
    edge_margin  a disc's visible extent in sigmas, kept inside the nearest feature edge
    """
    if sampling not in SAMPLINGS:
        raise ValueError(f"sampling must be one of {SAMPLINGS}")
    vertices = np.asarray(vertices, dtype=np.float64)
    triangles = np.asarray(triangles, dtype=np.int64)
    field = colors if isinstance(colors, ColorField) else ColorField.constant(colors)
    if budget < 1:
        raise ValueError("budget must be positive")

    v0, v1, v2 = (vertices[triangles[:, i]] for i in range(3))
    cross = np.cross(v1 - v0, v2 - v0)
    double_area = np.linalg.norm(cross, axis=1)
    keep = double_area > 2 * MIN_AREA
    if not keep.any():
        raise ValueError("mesh has no triangles with non-zero area")
    v0, v1, v2, cross, double_area = v0[keep], v1[keep], v2[keep], cross[keep], double_area[keep]
    field = field.subset(keep)
    area = 0.5 * double_area

    rng = np.random.default_rng(seed)
    expected = budget * area / area.sum()
    if sampling == "random":
        tri, s, t = _random_samples(expected, budget, rng)
    else:
        tri, s, t = _stratified_samples(expected, (v0 + v1 + v2) / 3.0, rng)

    # Unit square -> triangle, area-preserving (Osada et al. 2002).
    r1 = np.sqrt(s)
    a, b, c = 1.0 - r1, r1 * (1.0 - t), r1 * t
    positions = a[:, None] * v0[tri] + b[:, None] * v1[tri] + c[:, None] * v2[tri]

    bary = np.stack([a, b, c], axis=1)
    normal = cross / double_area[:, None]
    # Disc radius from the expected density, not the realized count: no zero division.
    radius = overlap * np.sqrt(area / expected)
    r = radius[tri]

    # Local frame: tangent along the first edge, normal of the face; isotropic disc.
    tangent = (v1 - v0)[tri]
    along, across = r.copy(), r.copy()
    if edge_aware:
        # Edge i is opposite vertex i. A point's distance to it is bary_i * 2A / |e_i|.
        edges = np.stack([v2 - v1, v0 - v2, v1 - v0], axis=1)                    # (T, 3, 3)
        lengths = np.linalg.norm(edges, axis=2)
        feature = _feature_edges(triangles[keep], normal, sharp_angle)            # (T, 3)
        dist = bary * (double_area[:, None] / lengths)[tri]
        dist = np.where(feature[tri], dist, np.inf)
        order = np.argsort(dist, axis=1)
        nearest = np.take_along_axis(dist, order[:, :1], axis=1)[:, 0]
        second = np.take_along_axis(dist, order[:, 1:2], axis=1)[:, 0]
        near = nearest < r * edge_margin
        # Across the nearest feature edge the visible extent (edge_margin sigmas) stops at
        # the edge; along it the disc stays full unless a second feature edge (a corner) is near.
        floor = 0.05 * r
        across = np.where(near, np.maximum(np.minimum(r, nearest / edge_margin), floor), r)
        along = np.where(near, np.maximum(np.minimum(r, second / edge_margin), floor), r)
        edge_dir = edges[tri, order[:, 0]]
        tangent = np.where(near[:, None], edge_dir, tangent)
    tangent = tangent / np.linalg.norm(tangent, axis=1)[:, None]
    point_normal = normal[tri]
    bitangent = np.cross(point_normal, tangent)
    rotations = matrix_to_quaternion(np.stack([tangent, bitangent, point_normal], axis=2))
    scales = np.stack([along, across, r * flatness], axis=1)

    # Color at each point's own UV, not per triangle or per vertex.
    colors = linear_to_srgb(field.sample(tri, bary))
    return GaussianSet(
        positions=positions,
        scales=scales,
        rotations=rotations,
        opacities=np.full(budget, opacity),
        colors=colors,
    )


def _feature_edges(triangles, normals, sharp_angle):
    """(T, 3) bool: edge i of each triangle (opposite vertex i) is a boundary, non-manifold
    or sharp edge (face normals differ by more than `sharp_angle` degrees)."""
    t = len(triangles)
    ends = np.stack([triangles[:, [1, 2]], triangles[:, [2, 0]], triangles[:, [0, 1]]], axis=1)
    ends = np.sort(ends.reshape(-1, 2), axis=1)
    key = ends[:, 0] * (int(triangles.max()) + 1) + ends[:, 1]
    _, inverse, counts = np.unique(key, return_inverse=True, return_counts=True)
    inverse = inverse.reshape(-1)
    feature = counts[inverse] != 2
    # Manifold edges: compare the normals of their two faces.
    slots = np.argsort(inverse, kind="stable")
    starts = np.cumsum(counts) - counts
    pair = np.flatnonzero(counts == 2)
    face = np.repeat(np.arange(t), 3)
    f1, f2 = face[slots[starts[pair]]], face[slots[starts[pair] + 1]]
    sharp_edge = np.zeros(len(counts), dtype=bool)
    sharp_edge[pair] = (normals[f1] * normals[f2]).sum(axis=1) < np.cos(np.radians(sharp_angle))
    feature |= sharp_edge[inverse]
    return feature.reshape(t, 3)


def _random_samples(expected, budget, rng):
    """i.i.d. points: multinomial counts, uniform (s, t) in the unit square."""
    counts = rng.multinomial(budget, expected / expected.sum())
    tri = np.repeat(np.arange(len(expected)), counts)
    return tri, rng.random(budget), rng.random(budget)


def _stratified_samples(expected, centroids, rng):
    """Evenly spread points with the same expected density as `_random_samples`.

    Triangles are walked along a Morton curve through their centroids, so neighbours
    in the walk are neighbours in space. Systematic sampling along the cumulative
    expected count (one random offset) gives each triangle floor or ceil of its share
    and spaces points by equal area even where points are fewer than triangles.
    Inside a triangle its n points form a jittered Fibonacci lattice.
    """
    order = np.argsort(_morton_codes(centroids), kind="stable")
    cumulative = np.cumsum(expected[order])
    budget = round(cumulative[-1])
    cumulative *= budget / cumulative[-1]
    cumulative[-1] = budget  # exact, so the counts sum to the budget despite rounding
    offset = rng.random()
    counts_sorted = np.diff(np.floor(np.concatenate([[0.0], cumulative]) + offset)).astype(np.int64)
    counts = np.empty_like(counts_sorted)
    counts[order] = counts_sorted
    tri = np.repeat(np.arange(len(expected)), counts)

    # Index of each point within its triangle, and that triangle's count.
    starts = np.repeat(np.cumsum(counts) - counts, counts)
    j = np.arange(len(tri)) - starts
    n = counts[tri]
    s = (j + rng.random(len(tri))) / n                        # one point per stratum
    t = (j * GOLDEN + rng.random(len(expected))[tri]) % 1.0    # Kronecker sequence
    return tri, s, t


def _morton_codes(points, bits=21):
    """64-bit Z-order codes of points quantized over their bounding box."""
    lo, hi = points.min(axis=0), points.max(axis=0)
    q = ((points - lo) / np.maximum(hi - lo, 1e-12) * ((1 << bits) - 1)).astype(np.uint64)
    code = np.zeros(len(points), dtype=np.uint64)
    for axis in range(3):
        code |= _spread_bits(q[:, axis]) << np.uint64(axis)
    return code


def _spread_bits(x):
    """Insert two zero bits between each of the low 21 bits of x."""
    x = x & np.uint64(0x1FFFFF)
    for shift, mask in ((32, 0x1F00000000FFFF), (16, 0x1F0000FF0000FF), (8, 0x100F00F00F00F00F),
                        (4, 0x10C30C30C30C30C3), (2, 0x1249249249249249)):
        x = (x | (x << np.uint64(shift))) & np.uint64(mask)
    return x


def surface_area(vertices, triangles):
    v = np.asarray(vertices, dtype=np.float64)[np.asarray(triangles)]
    return 0.5 * np.linalg.norm(np.cross(v[:, 1] - v[:, 0], v[:, 2] - v[:, 0]), axis=1).sum()


def matrix_to_quaternion(m):
    """(N, 3, 3) rotation matrices -> (N, 4) unit quaternions, w first, w >= 0.

    Shepperd's method: each row is solved from its largest of w, x, y, z, so the result
    stays exact near 180° turns, where w ~ 0 and the off-diagonal differences that
    would carry the signs vanish.
    """
    m = np.asarray(m, dtype=np.float64)
    m00, m11, m22 = m[:, 0, 0], m[:, 1, 1], m[:, 2, 2]
    d21, d02, d10 = m[:, 2, 1] - m[:, 1, 2], m[:, 0, 2] - m[:, 2, 0], m[:, 1, 0] - m[:, 0, 1]
    s01, s02, s12 = m[:, 0, 1] + m[:, 1, 0], m[:, 0, 2] + m[:, 2, 0], m[:, 1, 2] + m[:, 2, 1]
    # 4 * (squared component) for w, x, y, z; the largest is well away from zero
    four = np.stack([1 + m00 + m11 + m22, 1 + m00 - m11 - m22, 1 - m00 + m11 - m22, 1 - m00 - m11 + m22], 1)
    pick = np.argmax(four, axis=1)
    s = 2 * np.sqrt(np.maximum(four[np.arange(len(m)), pick], 1e-300))  # = 4 * that component
    q = np.select(
        [pick[:, None] == 0, pick[:, None] == 1, pick[:, None] == 2],
        [np.stack([s / 4, d21 / s, d02 / s, d10 / s], 1),
         np.stack([d21 / s, s / 4, s01 / s, s02 / s], 1),
         np.stack([d02 / s, s01 / s, s / 4, s12 / s], 1)],
        np.stack([d10 / s, s02 / s, s12 / s, s / 4], 1))
    q = q * np.where(q[:, :1] < 0, -1.0, 1.0)  # q and -q are the same turn
    return q / np.linalg.norm(q, axis=1)[:, None]
