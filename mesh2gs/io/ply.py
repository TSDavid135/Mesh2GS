"""PLY in the layout of the original 3DGS implementation (SH degree 0).

Values are stored pre-activation: f_dc = (rgb - 0.5) / C0, scale = ln(s),
opacity = logit(alpha), rotation = quaternion with w first.
"""

import numpy as np

from ..core.gaussians import GaussianSet

SH_C0 = 0.28209479177387814

FIELDS = (
    "x", "y", "z", "nx", "ny", "nz",
    "f_dc_0", "f_dc_1", "f_dc_2",
    "opacity",
    "scale_0", "scale_1", "scale_2",
    "rot_0", "rot_1", "rot_2", "rot_3",
)
DTYPE = np.dtype([(name, "<f4") for name in FIELDS])


def write(path, gaussians: GaussianSet):
    g = gaussians
    data = np.zeros(len(g), dtype=DTYPE)
    data["x"], data["y"], data["z"] = g.positions.T
    # nx..nz stay zero: the original format carries them but no renderer reads them.
    dc = (g.colors - 0.5) / SH_C0
    data["f_dc_0"], data["f_dc_1"], data["f_dc_2"] = dc.T
    alpha = np.clip(g.opacities.astype(np.float64), 1e-6, 1 - 1e-6)
    data["opacity"] = np.log(alpha / (1 - alpha))
    log_s = np.log(g.scales)
    data["scale_0"], data["scale_1"], data["scale_2"] = log_s.T
    data["rot_0"], data["rot_1"], data["rot_2"], data["rot_3"] = g.rotations.T

    header = ["ply", "format binary_little_endian 1.0", f"element vertex {len(g)}"]
    header += [f"property float {name}" for name in FIELDS]
    header += ["end_header"]
    with open(path, "wb") as f:
        f.write(("\n".join(header) + "\n").encode("ascii"))
        f.write(data.tobytes())


def read(path) -> GaussianSet:
    """Read a binary little-endian 3DGS PLY (ours or another tool's); only the DC color
    is kept, higher SH bands (f_rest_*) are ignored."""
    with open(path, "rb") as f:
        lines = []
        while True:
            line = f.readline().decode("ascii").strip()
            lines.append(line)
            if line == "end_header":
                break
        if "format binary_little_endian 1.0" not in lines:
            raise ValueError("only binary little-endian PLY is supported")
        count = next(int(l.split()[2]) for l in lines if l.startswith("element vertex"))
        props = [l.split() for l in lines if l.startswith("property")]
        if any(p[1] != "float" for p in props):
            raise ValueError("only float properties are supported")
        names = [p[2] for p in props]
        missing = set(FIELDS) - set(names) - {"nx", "ny", "nz"}
        if missing:
            raise ValueError(f"PLY lacks 3DGS fields: {sorted(missing)}")
        dtype = np.dtype([(n, "<f4") for n in names])
        data = np.frombuffer(f.read(count * dtype.itemsize), dtype=dtype, count=count)

    def cols(*names):
        return np.stack([data[n] for n in names], axis=1).astype(np.float64)

    return GaussianSet(
        positions=cols("x", "y", "z"),
        scales=np.exp(cols("scale_0", "scale_1", "scale_2")),
        rotations=cols("rot_0", "rot_1", "rot_2", "rot_3"),
        opacities=1 / (1 + np.exp(-data["opacity"].astype(np.float64))),
        colors=cols("f_dc_0", "f_dc_1", "f_dc_2") * SH_C0 + 0.5,
    )
