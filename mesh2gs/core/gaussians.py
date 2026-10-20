"""Common Gaussian set that every conversion method produces."""

from dataclasses import dataclass

import numpy as np


@dataclass
class GaussianSet:
    """Gaussians in activated (human-readable) form.

    positions  (N, 3) world coordinates
    scales     (N, 3) standard deviations along the local axes, > 0
    rotations  (N, 4) unit quaternions, w first
    opacities  (N,)   alpha in (0, 1)
    colors     (N, 3) display-referred sRGB in [0, 1], the space 3DGS renderers expect
    """

    positions: np.ndarray
    scales: np.ndarray
    rotations: np.ndarray
    opacities: np.ndarray
    colors: np.ndarray

    def __post_init__(self):
        n = len(self.positions)
        shapes = {
            "positions": (n, 3), "scales": (n, 3), "rotations": (n, 4),
            "opacities": (n,), "colors": (n, 3),
        }
        for name, shape in shapes.items():
            arr = np.asarray(getattr(self, name), dtype=np.float32)
            if arr.shape != shape:
                raise ValueError(f"{name}: expected shape {shape}, got {arr.shape}")
            setattr(self, name, arr)

    def __len__(self):
        return len(self.positions)
