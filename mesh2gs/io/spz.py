"""SPZ, Niantic's compressed Gaussian splat format, version 3 (gzip), SH degree 0.

Version 3 is the gzip layout every SPZ reader accepts (the reference library's newer
version 4 switched to ZSTD streams and still reads 1-3). The stream is a 16-byte header,
then per attribute one array over all points:

    positions  3 x 24-bit signed fixed point, 12 fractional bits (~0.25 mm)
    alphas     sigmoid(opacity) * 255
    colors     f_dc * 0.15 * 255 + 127.5
    scales     (ln s + 10) * 16
    rotations  "smallest three": index of the largest component in the top 2 bits,
               the other three as sign + 9-bit magnitude, scaled by 1/sqrt(1/2)

Positions are stored to 1/4096 of a unit (~0.25 mm in metres) within ±2048 units, so
the format suits scenes modelled in metres; a model a few centimetres across loses
visible detail. SPZ files are RUB (the reference library converts PLY's RDF on load); pass the set
already converted, e.g. axes.convert(g, "RUB"). Constants follow nianticlabs/spz
(load-spz.cc, splat-utils.h).
"""

import gzip
import struct

import numpy as np

from ..core.gaussians import GaussianSet

MAGIC = 0x5053474E  # "NGSP"
VERSION = 3
FRACTIONAL_BITS = 12
LIMIT = float(1 << (23 - FRACTIONAL_BITS))  # |coordinate| must stay below 2048 units
SMALL_EXTENT = 0.1  # units; below this the 1/4096 step starts to show (the 6 cm corset lost detail)
COLOR_SCALE = 0.15
SH_C0 = 0.28209479177387814
SQRT1_2 = 0.7071067811865476


def _u8(values):
    return np.clip(np.round(values), 0, 255).astype(np.uint8)


def _pack_rotations(wxyz):
    q = np.asarray(wxyz, dtype=np.float64)
    q = q / np.linalg.norm(q, axis=1, keepdims=True)
    q = q[:, [1, 2, 3, 0]]                      # SPZ orders components x, y, z, w
    largest = np.argmax(np.abs(q), axis=1)
    negate = q[np.arange(len(q)), largest] < 0
    comp = largest.astype(np.uint32)
    for i in range(4):                          # the three others, x first, in the low bits last
        keep = largest != i
        sign = ((q[:, i] < 0) ^ negate).astype(np.uint32)
        mag = ((2 ** 9 - 1) * (np.abs(q[:, i]) / SQRT1_2) + 0.5).astype(np.uint32)
        comp = np.where(keep, (comp << 10) | (sign << 9) | np.minimum(mag, 511), comp)
    return comp.astype("<u4").view(np.uint8).reshape(-1, 4)


def _unpack_rotations(packed):
    comp = packed.reshape(-1, 4).copy().view("<u4").ravel()
    largest = comp >> 30
    q = np.zeros((len(comp), 4))
    for i in range(3, -1, -1):                  # read back from the low bits
        other = largest != i
        mag = (comp & 511).astype(np.float64) / 511 * SQRT1_2
        sign = np.where(comp & 512, -1.0, 1.0)
        q[:, i] = np.where(other, sign * mag, q[:, i])
        comp = np.where(other, comp >> 10, comp)
    rest = np.sqrt(np.maximum(0.0, 1.0 - np.sum(q * q, axis=1)))
    q[np.arange(len(q)), largest] = rest
    return q[:, [3, 0, 1, 2]]                   # back to w, x, y, z


def write(path, gaussians: GaussianSet):
    g = gaussians
    n = len(g)
    fixed = np.round(np.asarray(g.positions, dtype=np.float64) * (1 << FRACTIONAL_BITS)).astype(np.int64)
    if np.abs(fixed).max(initial=0) >= 1 << 23:
        raise ValueError(f"SPZ positions must lie within ±{LIMIT:.0f} units")
    positions = (fixed & 0xFFFFFF).astype("<u4").view(np.uint8).reshape(n, 3, 4)[:, :, :3]
    alphas = _u8(np.clip(g.opacities, 0, 1) * 255)
    dc = (np.asarray(g.colors, dtype=np.float64) - 0.5) / SH_C0
    colors = _u8(dc * (COLOR_SCALE * 255) + 0.5 * 255)
    scales = _u8((np.log(np.maximum(g.scales, 1e-30)) + 10) * 16)
    rotations = _pack_rotations(g.rotations)

    header = struct.pack("<IIIBBBB", MAGIC, VERSION, n, 0, FRACTIONAL_BITS, 0, 0)
    body = b"".join(a.tobytes() for a in (positions, alphas, colors, scales, rotations))
    with gzip.open(path, "wb") as f:
        f.write(header + body)


def read(path) -> GaussianSet:
    """Read an SPZ file of version 2 or 3; higher SH bands are skipped."""
    with gzip.open(path, "rb") as f:
        data = f.read()
    magic, version, n, sh_degree, frac, _flags, _ = struct.unpack_from("<IIIBBBB", data)
    if magic != MAGIC or version not in (2, 3):
        raise ValueError(f"not an SPZ file of version 2 or 3 (magic {magic:#x}, version {version})")
    pos = 16

    def take(count, width):
        nonlocal pos
        chunk = np.frombuffer(data, dtype=np.uint8, count=count * width, offset=pos).reshape(count, width)
        pos += count * width
        return chunk

    raw = take(n * 3, 3).astype(np.int32)
    fixed = raw[:, 0] | (raw[:, 1] << 8) | (raw[:, 2] << 16)
    fixed = np.where(fixed & 0x800000, fixed - (1 << 24), fixed)
    positions = fixed.reshape(n, 3) / (1 << frac)
    alphas = take(n, 1)[:, 0] / 255.0
    colors = ((take(n, 3) / 255.0) - 0.5) / COLOR_SCALE * SH_C0 + 0.5
    scales = np.exp(take(n, 3) / 16.0 - 10)
    if version >= 3:
        rotations = _unpack_rotations(take(n, 4))
    else:
        xyz = take(n, 3) / 127.5 - 1
        w = np.sqrt(np.maximum(0.0, 1 - np.sum(xyz * xyz, axis=1)))
        rotations = np.column_stack([w, xyz])
    return GaussianSet(positions=positions, scales=scales, rotations=rotations,
                       opacities=alphas, colors=colors)
