"""A 360-degree panorama of a splat scene from one point: the light for a CG object placed
there (the World in Blender, so the lit color mode bakes the scene's own light).

    <python with torch + gsplat> mesh2gs/panorama.py <scene.ply> <out.png> --eye x,y,z [--width 2048]

The scene PLY and the eye are in Blender world space; the PNG is equirectangular in the
layout of Blender's Environment Texture (u from atan2(y, x), v from elevation, top row up)
and holds the splats' display colors (sRGB), so the World runs at strength 1. Six 90-degree
views (a cube) are rendered with gsplat and resampled. Runs outside Blender, like train.py;
the benchmark imports panorama() directly.
"""

import argparse
import math
import os
import sys
import types

import numpy as np

if __package__ in (None, ""):  # run as a script: see train.py
    _package = types.ModuleType("mesh2gs")
    _package.__path__ = [os.path.dirname(os.path.abspath(__file__))]
    sys.modules.setdefault("mesh2gs", _package)
from mesh2gs.io import ply  # noqa: E402


def cube_cameras(eye, face):
    """Six 90-degree views from `eye` along +-x, +-y, +-z: (forward, K, world-to-camera),
    OpenCV axes (x right, y down, z forward)."""
    f = face / 2
    k = np.array([[f, 0, face / 2], [0, f, face / 2], [0, 0, 1]])
    cams = []
    for axis in range(3):
        for sign in (1.0, -1.0):
            z = np.zeros(3)
            z[axis] = sign
            helper = np.array([0.0, 0.0, 1.0]) if axis == 1 else np.array([0.0, 1.0, 0.0])
            x = np.cross(helper, z)
            x /= np.linalg.norm(x)
            y = np.cross(z, x)
            c2w = np.eye(4)
            c2w[:3, 0], c2w[:3, 1], c2w[:3, 2], c2w[:3, 3] = x, y, z, eye
            cams.append((z, k, np.linalg.inv(c2w)))
    return cams


def render_faces(g, cams, face):
    """sRGB images (face, face, 3) of the views, over black."""
    import torch
    from gsplat import rasterization
    t = lambda a: torch.as_tensor(np.ascontiguousarray(a), dtype=torch.float32, device="cuda")
    quats = t(g.rotations)
    quats = quats / quats.norm(dim=1, keepdim=True)
    images = []
    for _, k, vm in cams:  # one at a time: scenes are large
        with torch.no_grad():
            colors, _, _ = rasterization(
                means=t(g.positions), quats=quats, scales=t(g.scales), opacities=t(g.opacities),
                colors=t(np.clip(g.colors, 0, 1)), viewmats=t(vm)[None], Ks=t(k)[None], width=face, height=face)
        images.append(colors[0].clamp(0, 1).cpu().numpy())
    return images


def panorama(g, eye, to_scene=np.eye(3), width=2048, face=1024):
    """Equirectangular sRGB panorama (H, W, 3) of `g` seen from `eye`, for directions of a
    Blender frame; `to_scene` turns those into the scene's directions (identity when the
    scene is in Blender world space)."""
    height = width // 2
    u = (np.arange(width) + 0.5) / width
    v = 1 - (np.arange(height) + 0.5) / height
    phi = (0.5 - u)[None, :] * 2 * math.pi
    theta = (v - 0.5)[:, None] * math.pi
    d = np.stack(np.broadcast_arrays(np.cos(theta) * np.cos(phi), np.cos(theta) * np.sin(phi), np.sin(theta)), -1)
    d = d @ np.asarray(to_scene, dtype=np.float64).T
    cams = cube_cameras(np.asarray(eye, dtype=np.float64), face)
    faces = render_faces(g, cams, face)
    best = np.argmax(np.stack([d @ forward for forward, _, _ in cams], -1), -1)
    out = np.zeros((height, width, 3), dtype=np.float32)
    for i, (_, k, vm) in enumerate(cams):
        sel = best == i
        p = d[sel] @ vm[:3, :3].T
        px = np.clip((p[:, :2] / p[:, 2:3] * k[0, 0] + face / 2).astype(int), 0, face - 1)
        out[sel] = faces[i][px[:, 1], px[:, 0]]
    return out


def save_png(image, path):
    from PIL import Image
    os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
    Image.fromarray((np.clip(image, 0, 1) * 255 + 0.5).astype(np.uint8)).save(path)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("scene")
    parser.add_argument("out")
    parser.add_argument("--eye", required=True, help="x,y,z in Blender world space")
    parser.add_argument("--width", type=int, default=2048)
    args = parser.parse_args()
    import torch
    if not torch.cuda.is_available():
        sys.exit("needs an NVIDIA graphics card with CUDA: torch.cuda.is_available() = False")
    image = panorama(ply.read(args.scene), [float(c) for c in args.eye.split(",")], width=args.width)
    save_png(image, args.out)
    print(f"panorama {image.shape[1]}x{image.shape[0]} -> {args.out}", flush=True)


if __name__ == "__main__":
    main()
