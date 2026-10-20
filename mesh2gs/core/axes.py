"""Axis conventions for export, named by what +X, +Y, +Z point to (as in the SPZ spec).

R/L right/left, U/D up/down, F/B forward/back, as seen by a viewer looking at the
model's front. Blender is RFU: +X right, +Y forward (away from the front viewer), +Z up.
"""

import numpy as np

from .analytic import matrix_to_quaternion
from .gaussians import GaussianSet

BLENDER = "RFU"
PRESETS = {
    "RDF": "3DGS / SuperSplat / PlayCanvas",   # the original PLY convention (COLMAP, y down)
    "RUF": "Unity",                             # left-handed, y up; its importer reads PLY as is
    "RUB": "OpenGL / three.js / SPZ",
    "RFU": "Blender, no rotation",
}
_BLENDER_AXIS = {"R": (0, 1), "L": (0, -1), "F": (1, 1), "B": (1, -1), "U": (2, 1), "D": (2, -1)}


def matrix(code):
    """3x3 matrix taking Blender coordinates to the `code` convention."""
    if len(code) != 3 or {_BLENDER_AXIS[c][0] for c in code} != {0, 1, 2}:
        raise ValueError(f"not an axis convention: {code}")
    m = np.zeros((3, 3))
    for row, letter in enumerate(code):
        axis, sign = _BLENDER_AXIS[letter]
        m[row, axis] = sign
    return m


def convert(gaussians: GaussianSet, code) -> GaussianSet:
    """Re-express Blender-space Gaussians in the `code` convention."""
    return gaussians if code == BLENDER else transform(gaussians, matrix(code))


def to_blender(gaussians: GaussianSet, code) -> GaussianSet:
    """Inverse of `convert`: Gaussians given in `code` back to Blender space."""
    return gaussians if code == BLENDER else transform(gaussians, matrix(code).T)


def transform(gaussians: GaussianSet, m) -> GaussianSet:
    """Apply an axis matrix M (orthogonal, entries 0/±1).

    Positions map by M. A covariance R S^2 R^T maps to M R S^2 R^T M^T, i.e. rotation
    M R; when M is a reflection (handedness change) that is improper, so the local
    z axis is flipped as well (R' = M R diag(1, 1, -1)), which leaves S^2 unchanged.
    """
    q = gaussians.rotations.astype(np.float64)
    q = q / np.linalg.norm(q, axis=1, keepdims=True)
    rot = _quaternion_to_matrix(q)
    new_rot = m @ rot
    if np.linalg.det(m) < 0:
        new_rot[:, :, 2] *= -1
    return GaussianSet(
        positions=gaussians.positions.astype(np.float64) @ m.T,
        scales=gaussians.scales,
        rotations=matrix_to_quaternion(new_rot),
        opacities=gaussians.opacities,
        colors=gaussians.colors,
    )


def _quaternion_to_matrix(q):
    w, x, y, z = q.T
    return np.stack([
        np.stack([1 - 2 * (y * y + z * z), 2 * (x * y - w * z), 2 * (x * z + w * y)], -1),
        np.stack([2 * (x * y + w * z), 1 - 2 * (x * x + z * z), 2 * (y * z - w * x)], -1),
        np.stack([2 * (x * z - w * y), 2 * (y * z + w * x), 1 - 2 * (x * x + y * y)], -1),
    ], axis=1)
