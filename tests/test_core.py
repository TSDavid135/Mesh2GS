import gzip
import os
import struct
import tempfile
import unittest

import numpy as np

from mesh2gs.core import analytic, axes, baseline
from mesh2gs.core.color import linear_to_srgb, srgb_to_linear
from mesh2gs.core.texture import ColorField, Texture
from mesh2gs.io import ply, spz

SH_C0 = 0.28209479177387814

GREY = (0.5, 0.5, 0.5)
DATA = os.path.join(os.path.dirname(os.path.abspath(__file__)), "data")


def grid_plane(k):
    """Unit square as k x k quads, 2 k^2 triangles (lower-right then upper-left per quad)."""
    xs = np.linspace(0, 1, k + 1)
    x, y = np.meshgrid(xs, xs)
    v = np.stack([x.ravel(), y.ravel(), np.zeros(x.size)], 1)
    t = []
    for i in range(k):
        for j in range(k):
            a = i * (k + 1) + j
            t += [[a, a + 1, a + k + 2], [a, a + k + 2, a + k + 1]]
    return v, np.array(t)


def triangle_of(xy, k):
    """Index into grid_plane(k) triangles for points in the unit square."""
    col = np.minimum((xy[:, 0] * k).astype(int), k - 1)
    row = np.minimum((xy[:, 1] * k).astype(int), k - 1)
    fx, fy = xy[:, 0] * k - col, xy[:, 1] * k - row
    return (row * k + col) * 2 + (fy > fx)


def read_obj(path):
    """Minimal OBJ reader: positions, triangles, per-corner normals and UVs."""
    v, vt, vn, faces = [], [], [], []
    for line in open(path):
        p = line.split()
        if not p:
            continue
        if p[0] in ("v", "vt", "vn"):
            {"v": v, "vt": vt, "vn": vn}[p[0]].append([float(x) for x in p[1:]])
        elif p[0] == "f":
            faces.append([[int(i) - 1 for i in c.split("/")] for c in p[1:4]])
    f = np.array(faces)  # (T, 3, 3): v, vt, vn indices
    return np.array(v), f[:, :, 0], np.array(vn)[f[:, :, 2]], np.array(vt)[f[:, :, 1]]


def read_any_ply(path):
    raw = open(path, "rb").read()
    end = raw.index(b"end_header\n") + len(b"end_header\n")
    names = [l.split()[-1] for l in raw[:end].decode().splitlines() if l.startswith("property")]
    return np.frombuffer(raw[end:], dtype=np.dtype([(n, "<f4") for n in names]))


def quaternion_to_matrix(q):
    w, x, y, z = q.T
    return np.stack([
        np.stack([1 - 2 * (y * y + z * z), 2 * (x * y - w * z), 2 * (x * z + w * y)], -1),
        np.stack([2 * (x * y + w * z), 1 - 2 * (x * x + z * z), 2 * (y * z - w * x)], -1),
        np.stack([2 * (x * z - w * y), 2 * (y * z + w * x), 1 - 2 * (x * x + y * y)], -1),
    ], axis=1)


def square(size=1.0, offset=(0.0, 0.0, 0.0)):
    v = np.array([[0, 0, 0], [1, 0, 0], [1, 1, 0], [0, 1, 0]], dtype=float) * size + offset
    return v, np.array([[0, 1, 2], [0, 2, 3]])


class QuaternionTest(unittest.TestCase):
    def test_round_trip_random_rotations(self):
        rng = np.random.default_rng(1)
        q = rng.normal(size=(500, 4))
        q /= np.linalg.norm(q, axis=1)[:, None]
        m = quaternion_to_matrix(q)
        back = analytic.matrix_to_quaternion(m)
        np.testing.assert_allclose(quaternion_to_matrix(back), m, atol=1e-9)

    def test_half_turns(self):
        # w ~ 0: the signs of x, y, z cannot come from the off-diagonal differences
        rng = np.random.default_rng(2)
        axis = rng.normal(size=(500, 3))
        axis /= np.linalg.norm(axis, axis=1)[:, None]
        axis[:5] = [[1, 0, 0], [0, 1, 0], [0, 0, 1], [0, 0.99, 0.136], [0.6, -0.8, 0]]
        axis /= np.linalg.norm(axis, axis=1)[:, None]
        angle = np.pi - rng.uniform(0, 1e-4, size=(500, 1))
        angle[:250] = np.pi  # exact half turns, as a flipped tangent frame gives
        q = np.hstack([np.cos(angle / 2), axis * np.sin(angle / 2)])
        m = quaternion_to_matrix(q)
        back = analytic.matrix_to_quaternion(m)
        np.testing.assert_allclose(quaternion_to_matrix(back), m, atol=1e-9)


class AnalyticTest(unittest.TestCase):
    def test_square_geometry(self):
        v, t = square()
        g = analytic.convert(v, t, [GREY, GREY], 10_000, seed=3, edge_aware=False, overlap=1.0)
        self.assertEqual(len(g), 10_000)
        self.assertTrue(np.all(g.positions[:, 2] == 0))
        self.assertTrue(np.all((g.positions[:, :2] >= 0) & (g.positions[:, :2] <= 1)))
        normals = quaternion_to_matrix(g.rotations.astype(np.float64))[:, :, 2]
        np.testing.assert_allclose(normals, np.tile([0, 0, 1], (len(g), 1)), atol=1e-6)
        # disc radius = k * sqrt(area per Gaussian), thickness = flatness * radius
        np.testing.assert_allclose(g.scales[:, 0], np.sqrt(1 / 10_000), rtol=1e-5)
        np.testing.assert_allclose(g.scales[:, 2], 0.05 * np.sqrt(1 / 10_000), rtol=1e-5)

    def test_samples_follow_area_not_triangulation(self):
        small, t_small = square(1.0)
        big, t_big = square(2.0, offset=(5, 0, 0))
        v = np.concatenate([small, big])
        t = np.concatenate([t_small, t_big + 4])
        g = analytic.convert(v, t, [GREY] * 4, 50_000, seed=0, edge_aware=False)
        share_big = np.mean(g.positions[:, 0] >= 5)
        self.assertAlmostEqual(share_big, 0.8, delta=0.01)  # area 4 of 5
        radii = np.unique(np.round(g.scales[:, 0], 9))
        self.assertEqual(len(radii), 1)  # uniform density -> one disc size everywhere

    def test_uniform_inside_triangle(self):
        v = np.array([[0, 0, 0], [1, 0, 0], [0, 1, 0]], dtype=float)
        g = analytic.convert(v, [[0, 1, 2]], [GREY], 40_000, seed=5)
        lower_half = np.mean(g.positions[:, 1] < 0.5)
        self.assertAlmostEqual(lower_half, 0.75, delta=0.01)  # area below y=0.5 is 3/4

    def test_degenerate_triangles(self):
        v, t = square()
        v = np.concatenate([v, [[2, 2, 2]]])
        t = np.concatenate([t, [[4, 4, 4]]])
        g = analytic.convert(v, t, [GREY] * 3, 1000)
        self.assertEqual(len(g), 1000)
        with self.assertRaises(ValueError):
            analytic.convert(v, [[4, 4, 4]], [GREY], 10)

    def test_colors_converted_to_srgb(self):
        v, t = square()
        g = analytic.convert(v, t, [(0.5, 0.0, 1.0)] * 2, 10)
        np.testing.assert_allclose(g.colors[0], [0.735357, 0.0, 1.0], atol=1e-5)
        np.testing.assert_allclose(srgb_to_linear(linear_to_srgb(0.2)), 0.2)

    def test_texture_sampled_at_every_point(self):
        v, t = square()
        tex = Texture(np.array([[[1, 0, 0], [0, 1, 0]], [[0, 0, 1], [1, 1, 1]]], dtype=float),
                      interpolation="Closest")
        uvs = v[t][:, :, :2]  # UV = xy on the unit square
        g = analytic.convert(v, t, ColorField([0, 0], [tex], uvs), 20_000, seed=2)
        quadrant = (g.positions[:, 1] >= 0.5) * 2 + (g.positions[:, 0] >= 0.5)
        expected = linear_to_srgb(tex.pixels.reshape(4, 3))[quadrant]
        np.testing.assert_allclose(g.colors, expected, atol=1e-6)

    def test_feature_edges(self):
        v, t = square()
        n = np.tile([0.0, 0.0, 1.0], (2, 1))
        feature = analytic._feature_edges(t, n, 30.0)
        self.assertEqual(feature.sum(), 4)  # the 4 border edges, not the shared diagonal
        cube_v = np.array([[x, y, z] for x in (0, 1) for y in (0, 1) for z in (0, 1)], float)
        quads = [(0, 1, 3, 2), (4, 6, 7, 5), (0, 4, 5, 1), (2, 3, 7, 6), (0, 2, 6, 4), (1, 5, 7, 3)]
        cube_t = np.array([tri for a, b, c, d in quads for tri in ((a, b, c), (a, c, d))])
        e1, e2 = cube_v[cube_t[:, 1]] - cube_v[cube_t[:, 0]], cube_v[cube_t[:, 2]] - cube_v[cube_t[:, 0]]
        normals = np.cross(e1, e2)
        normals /= np.linalg.norm(normals, axis=1, keepdims=True)
        feature = analytic._feature_edges(cube_t, normals, 30.0)
        self.assertEqual(feature.sum(), 24)  # 12 cube edges x 2 triangles; 6 diagonals are not

    def test_discs_stay_inside_feature_edges(self):
        v, t = square()
        margin = 1.5
        g = analytic.convert(v, t, [GREY, GREY], 400, seed=1, edge_margin=margin, overlap=1.0)
        x, y = g.positions[:, 0], g.positions[:, 1]
        border = np.minimum.reduce([x, 1 - x, y, 1 - y])
        r = np.sqrt(1 / 400)
        extent = margin * g.scales[:, :2].min(axis=1)
        self.assertTrue(np.all(extent <= np.maximum(border, margin * 0.05 * r) + 1e-6))
        far = border > r * margin
        np.testing.assert_allclose(g.scales[far, :2], r, rtol=1e-5)  # untouched away from edges
        self.assertTrue(np.any(g.scales[:, 1] < 0.9 * r))              # and squashed near them

    def test_stratified_counts_follow_expected_share(self):
        v, t = grid_plane(20)  # 800 equal triangles
        for budget in (1, 100, 800, 2345):
            g = analytic.convert(v, t, np.full((len(t), 3), 0.5), budget, seed=4)
            self.assertEqual(len(g), budget)
            tri = triangle_of(g.positions[:, :2], 20)
            counts = np.bincount(tri, minlength=len(t))
            expected = budget / len(t)
            self.assertTrue(np.all(np.abs(counts - expected) < 1), budget)  # floor or ceil

    def test_stratified_is_more_even_than_random(self):
        v, t = grid_plane(20)
        cv = {}
        for sampling in ("random", "stratified"):
            runs = []
            for seed in range(3):
                p = analytic.convert(v, t, np.full((len(t), 3), 0.5), 400, seed=seed,
                                     sampling=sampling).positions[:, :2]
                d = np.linalg.norm(p[:, None] - p[None], axis=2)
                np.fill_diagonal(d, np.inf)
                nn = d.min(axis=1)
                runs.append(nn.std() / nn.mean())
            cv[sampling] = np.mean(runs)
        self.assertLess(cv["stratified"], 0.75 * cv["random"])

    def test_seed_is_reproducible(self):
        v, t = square()
        a = analytic.convert(v, t, [GREY] * 2, 500, seed=7)
        b = analytic.convert(v, t, [GREY] * 2, 500, seed=7)
        np.testing.assert_array_equal(a.positions, b.positions)


def checker_texture():
    srgb = np.load(os.path.join(DATA, "checker_texture.npz"))["srgb"] / 255.0
    return Texture(srgb_to_linear(srgb))


class TextureTest(unittest.TestCase):
    def setUp(self):
        # 2x2 texels, row 0 at the bottom: red green / blue white
        self.tex = Texture(np.array([[[1, 0, 0], [0, 1, 0]], [[0, 0, 1], [1, 1, 1]]], dtype=float))

    def test_texel_centers_and_bilinear(self):
        centers = np.array([[0.25, 0.25], [0.75, 0.25], [0.25, 0.75], [0.75, 0.75]])
        np.testing.assert_allclose(self.tex.sample(centers), [[1, 0, 0], [0, 1, 0], [0, 0, 1], [1, 1, 1]])
        np.testing.assert_allclose(self.tex.sample(np.array([[0.5, 0.25]])), [[0.5, 0.5, 0]])

    def test_extension_and_nearest(self):
        wrapped = self.tex.sample(np.array([[1.25, 0.25], [-0.75, 0.25]]))
        np.testing.assert_allclose(wrapped, [[1, 0, 0], [1, 0, 0]])  # REPEAT
        self.tex.extension = "EXTEND"
        np.testing.assert_allclose(self.tex.sample(np.array([[1.25, 0.25]])), [[0, 1, 0]])
        np.testing.assert_allclose(self.tex.sample(np.array([[0.6, 0.9]]), mode="nearest"), [[1, 1, 1]])

    def test_kiri_rule_rounds_to_corner_texels(self):
        # round(uv * (size - 1)): 0.49 -> texel 0, 0.51 -> texel 1
        np.testing.assert_allclose(self.tex.sample(np.array([[0.49, 0.0], [0.51, 0.0]]), mode="kiri"),
                                   [[1, 0, 0], [0, 1, 0]])

    def test_kiri_rule_wraps_tiled_uvs(self):
        # glTF imports give v in [-1, 0]; the converter clamps, M0 wraps a REPEAT texture.
        np.testing.assert_allclose(self.tex.sample(np.array([[0.0, -0.99]]), mode="kiri"), [[1, 0, 0]])
        np.testing.assert_allclose(self.tex.sample(np.array([[1.0, 1.0]]), mode="kiri"), [[1, 1, 1]])
        self.tex.extension = "EXTEND"
        np.testing.assert_allclose(self.tex.sample(np.array([[0.0, -0.99]]), mode="kiri"), [[1, 0, 0]])

    def test_field_mixes_constants_and_textures(self):
        uvs = np.array([[[0.25, 0.25]] * 3, [[0.75, 0.75]] * 3])
        field = ColorField([0, 1], [(0.1, 0.2, 0.3), self.tex], uvs)
        out = field.sample(np.array([0, 1]), np.full((2, 3), 1 / 3))
        np.testing.assert_allclose(out, [[0.1, 0.2, 0.3], [1, 1, 1]])


class BaselineTest(unittest.TestCase):
    # Output of the real KIRI mesh2gs_win.exe (3DGS Render 5.1.0) for checker_cube_s3.obj. Third-party
    # output, so it is kept out of the repository (.gitignore); the test runs where the file exists.
    KIRI_OUTPUT = os.path.join(DATA, "kiri_checker_cube_s3.ply")

    @unittest.skipUnless(os.path.exists(KIRI_OUTPUT), "KIRI converter output not present")
    def test_matches_kiri_converter_output(self):
        v, t, normals, uvs = read_obj(os.path.join(DATA, "checker_cube_s3.obj"))
        kiri = read_any_ply(self.KIRI_OUTPUT)
        wv, wt = baseline.weld(v, t, normals, uvs)
        field = ColorField(np.zeros(len(t), dtype=int), [checker_texture()], uvs)
        g = baseline.convert(wv, wt, field.corner_colors(mode="kiri"))

        self.assertEqual(len(g), len(kiri))  # one Gaussian per triangle
        np.testing.assert_allclose(g.positions, np.stack([kiri["x"], kiri["y"], kiri["z"]], 1), atol=1e-6)
        kiri_scales = np.exp(np.stack([kiri[f"scale_{i}"] for i in range(3)], 1))
        np.testing.assert_allclose(g.scales[:, :2], kiri_scales[:, :2], rtol=1e-5)
        np.testing.assert_allclose(g.rotations, np.stack([kiri[f"rot_{i}"] for i in range(4)], 1), atol=1e-5)
        kiri_colors = np.stack([kiri[f"f_dc_{i}"] for i in range(3)], 1) * SH_C0 + 0.5
        np.testing.assert_allclose(g.colors, kiri_colors, atol=1e-5)
        self.assertTrue(np.all(np.isinf(kiri["opacity"])))    # the converter's bug...
        self.assertTrue(np.all(np.isfinite(g.opacities)))     # ...which M0 does not copy

    def test_layouts(self):
        v, t = square()
        for n in (1, 3, 4, 6):
            g = baseline.convert(v, t, np.full((2, 3, 3), 0.5), n)
            self.assertEqual(len(g), 2 * n)
        with self.assertRaises(ValueError):
            baseline.convert(v, t, np.full((2, 3, 3), 0.5), 2)

    def test_colors_averaged_per_vertex(self):
        v, t = square()
        colors = np.array([[[1.0, 0, 0]] * 3, [[0, 0, 1.0]] * 3])  # red and blue triangle
        g = baseline.convert(v, t, colors, 1)
        self.assertTrue(np.all(g.colors[:, 0] > 0) and np.all(g.colors[:, 2] > 0))  # mixed at shared vertices

    def test_surface_area(self):
        v, t = square(2.0)
        self.assertAlmostEqual(analytic.surface_area(v, t), 4.0)


class AxesTest(unittest.TestCase):
    def covariance(self, g):
        q = g.rotations / np.linalg.norm(g.rotations, axis=1, keepdims=True)
        r = quaternion_to_matrix(q.astype(np.float64))
        s2 = g.scales.astype(np.float64) ** 2
        return np.einsum("nij,nj,nkj->nik", r, s2, r)

    def test_matrices(self):
        np.testing.assert_array_equal(axes.matrix("RFU"), np.eye(3))
        np.testing.assert_array_equal(axes.matrix("RDF") @ [0, 0, 1], [0, -1, 0])  # Blender up -> y down
        np.testing.assert_array_equal(axes.matrix("RUF") @ [0, 1, 0], [0, 0, 1])   # front viewer's forward
        self.assertLess(np.linalg.det(axes.matrix("RUF")), 0)   # Unity is left-handed
        self.assertGreater(np.linalg.det(axes.matrix("RUB")), 0)
        with self.assertRaises(ValueError):
            axes.matrix("RRU")

    def test_covariance_maps_exactly(self):
        rng = np.random.default_rng(3)
        q = rng.normal(size=(200, 4))
        q /= np.linalg.norm(q, axis=1, keepdims=True)
        g = analytic.GaussianSet(positions=rng.normal(size=(200, 3)), scales=rng.random((200, 3)) + 0.1,
                                 rotations=q, opacities=np.full(200, 0.5), colors=rng.random((200, 3)))
        for code in axes.PRESETS:
            m = axes.matrix(code)
            out = axes.convert(g, code)
            np.testing.assert_allclose(out.positions, g.positions @ m.T, atol=1e-5)
            np.testing.assert_allclose(self.covariance(out), m @ self.covariance(g) @ m.T, atol=1e-5)
            np.testing.assert_allclose(np.linalg.norm(out.rotations, axis=1), 1.0, atol=1e-5)


class PlyTest(unittest.TestCase):
    def test_round_trip_and_layout(self):
        v, t = square()
        g = analytic.convert(v, t, [(0.2, 0.4, 0.6)] * 2, 1234, seed=1)
        with tempfile.TemporaryDirectory() as tmp:
            path = os.path.join(tmp, "out.ply")
            ply.write(path, g)
            raw = open(path, "rb").read()
            header_end = raw.index(b"end_header\n") + len(b"end_header\n")
            self.assertEqual(len(raw) - header_end, 1234 * 17 * 4)
            self.assertIn(b"format binary_little_endian 1.0", raw[:header_end])
            back = ply.read(path)
        for name in ("positions", "scales", "rotations", "opacities", "colors"):
            np.testing.assert_allclose(getattr(back, name), getattr(g, name), rtol=1e-5, atol=1e-6)
        self.assertTrue(np.all(np.isfinite(back.opacities)))


class SpzTest(unittest.TestCase):
    def test_round_trip_within_quantization(self):
        v, t = square()
        g = analytic.convert(v, t, [(0.2, 0.4, 0.6), (0.9, 0.1, 0.5)], 500, seed=1)
        rng = np.random.default_rng(0)
        g.rotations = rng.normal(size=(len(g), 4))  # any orientation, signs included
        g.rotations /= np.linalg.norm(g.rotations, axis=1, keepdims=True)
        g.positions = g.positions + rng.uniform(-3, 3, size=(len(g), 3))
        with tempfile.TemporaryDirectory() as tmp:
            path = os.path.join(tmp, "out.spz")
            spz.write(path, g)
            with gzip.open(path, "rb") as f:
                raw = f.read()
            back = spz.read(path)
        magic, version, count = struct.unpack_from("<III", raw)
        self.assertEqual((magic, version, count), (0x5053474E, 3, 500))
        self.assertEqual(len(raw), 16 + 500 * (9 + 1 + 3 + 3 + 4))
        np.testing.assert_allclose(back.positions, g.positions, atol=0.5 / 4096 + 1e-9)
        np.testing.assert_allclose(back.opacities, g.opacities, atol=0.5 / 255 + 1e-9)
        np.testing.assert_allclose(back.colors, g.colors, atol=0.004)
        np.testing.assert_allclose(np.log(back.scales), np.log(g.scales), atol=0.5 / 16 + 1e-9)
        # q and -q are the same rotation
        np.testing.assert_allclose(np.abs(np.sum(back.rotations * g.rotations, axis=1)), 1.0, atol=2e-3)

    def test_positions_out_of_range(self):
        v, t = square()
        g = analytic.convert(v, t, [(0.5, 0.5, 0.5)] * 2, 10, seed=1)
        g.positions = g.positions + 5000.0
        with tempfile.TemporaryDirectory() as tmp, self.assertRaises(ValueError):
            spz.write(os.path.join(tmp, "far.spz"), g)


if __name__ == "__main__":
    unittest.main()
