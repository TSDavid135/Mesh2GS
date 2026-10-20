"""M2 / M3: optimize Gaussians against rendered views of the model (gsplat).

    <python with torch + gsplat> mesh2gs/train.py --cameras <dir>/cameras_train.json --out out.ply
        (--init random --bbox x0,y0,z0,x1,y1,z1 | --init <m1.ply>) --budget N [--iters K]

M2  starts from N random points in the model's bounding box, as the original 3DGS does
    for synthetic scenes.
M3  starts from the M1 result and refines it for a few iterations.
The count stays N: gsplat's MCMC strategy only relocates Gaussians that faded out
(cap_max = N), so the result compares with M1 at the same budget. Colors are SH
degree 0, the same as the add-on writes. Training images are the reference views
composited on black; the evaluation views are a different set of cameras.

Runs outside Blender, in a Python that has PyTorch (CUDA) and gsplat: the add-on starts
it as a subprocess (views.py renders the cameras) and so does the benchmark. Lines
"progress <done> <total>" on stdout report the iterations.
"""

import argparse
import json
import os
import sys
import time
import types

import numpy as np

if __package__ in (None, ""):
    # Run as a script: make core/ and io/ importable as mesh2gs.* without running the
    # package's __init__ (it registers the Blender add-on).
    _package = types.ModuleType("mesh2gs")
    _package.__path__ = [os.path.dirname(os.path.abspath(__file__))]
    sys.modules.setdefault("mesh2gs", _package)
from mesh2gs.core.gaussians import GaussianSet  # noqa: E402
from mesh2gs.io import ply  # noqa: E402

SH_C0 = 0.28209479177387814
DEVICE = "cuda"
PROGRESS_EVERY = 20


def load_views(path):
    import torch
    from PIL import Image
    meta = json.load(open(path))
    root = os.path.dirname(path)
    images, viewmats, ks = [], [], []
    for v in meta["views"]:
        rgba = np.asarray(Image.open(os.path.join(root, v["image"])).convert("RGBA"), dtype=np.float32) / 255
        images.append(rgba[..., :3] * rgba[..., 3:])
        viewmats.append(v["viewmat"])
        ks.append(v["K"])
    t = lambda a: torch.tensor(np.array(a), dtype=torch.float32, device=DEVICE)
    return t(images), t(viewmats), t(ks), meta["views"][0]["width"], meta["views"][0]["height"]


def ssim(a, b, size=11, sigma=1.5):
    """Differentiable SSIM of (H, W, 3) images."""
    import torch
    import torch.nn.functional as F
    a, b = a.permute(2, 0, 1)[None], b.permute(2, 0, 1)[None]
    x = torch.arange(size, device=a.device, dtype=a.dtype) - size // 2
    w = torch.exp(-x ** 2 / (2 * sigma ** 2))
    w = ((w[:, None] * w[None]) / w.sum() ** 2).expand(3, 1, size, size)
    blur = lambda img: F.conv2d(img, w, groups=3)
    mu_a, mu_b = blur(a), blur(b)
    var_a, var_b = blur(a * a) - mu_a ** 2, blur(b * b) - mu_b ** 2
    cov = blur(a * b) - mu_a * mu_b
    c1, c2 = 0.01 ** 2, 0.03 ** 2
    return (((2 * mu_a * mu_b + c1) * (2 * cov + c2)) / ((mu_a ** 2 + mu_b ** 2 + c1) * (var_a + var_b + c2))).mean()


def random_init(bbox, n, rng):
    lo, hi = np.array(bbox[:3]), np.array(bbox[3:])
    spacing = float(np.prod(np.maximum(hi - lo, 1e-6)) / n) ** (1 / 3)
    return GaussianSet(
        positions=lo + rng.random((n, 3)) * (hi - lo),
        scales=np.full((n, 3), spacing * 0.5),
        rotations=np.tile([1.0, 0, 0, 0], (n, 1)),
        opacities=np.full(n, 0.5),  # gsplat's MCMC setting; 0.1 lets small sets fade out entirely
        colors=np.full((n, 3), 0.5),
    )


def to_params(g):
    import torch
    t = lambda a: torch.nn.Parameter(torch.tensor(np.asarray(a, dtype=np.float32), device=DEVICE))
    alpha = np.clip(g.opacities, 1e-4, 1 - 1e-4)
    return torch.nn.ParameterDict({
        "means": t(g.positions),
        "scales": t(np.log(np.maximum(g.scales, 1e-7))),
        "quats": t(g.rotations / np.linalg.norm(g.rotations, axis=1, keepdims=True)),
        "opacities": t(np.log(alpha / (1 - alpha))),
        "sh0": t(((np.clip(g.colors, 0, 1) - 0.5) / SH_C0)[:, None, :]),
    })


def to_gaussians(params):
    d = {k: v.detach().cpu().numpy() for k, v in params.items()}
    q = d["quats"] / np.linalg.norm(d["quats"], axis=1, keepdims=True)
    return GaussianSet(positions=d["means"], scales=np.exp(d["scales"]), rotations=q,
                       opacities=1 / (1 + np.exp(-d["opacities"])),
                       colors=np.clip(d["sh0"][:, 0, :] * SH_C0 + 0.5, 0, 1))


def safe_mcmc(**kwargs):
    """gsplat's MCMCStrategy, except that relocation is skipped when no Gaussian is left
    alive (it samples targets among live ones and fails on CUDA with none)."""
    import torch
    from gsplat.strategy import MCMCStrategy

    class SafeMCMC(MCMCStrategy):
        def _relocate_gs(self, params, optimizers, binoms):
            alive = torch.sigmoid(params["opacities"].flatten()) > self.min_opacity
            if not bool(alive.any()):
                return 0
            return super()._relocate_gs(params, optimizers, binoms)

    return SafeMCMC(**kwargs)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--cameras", required=True)
    parser.add_argument("--out", required=True)
    parser.add_argument("--init", required=True, help="'random' (M2) or a PLY to refine (M3)")
    parser.add_argument("--bbox", help="x0,y0,z0,x1,y1,z1 for random init")
    parser.add_argument("--budget", type=int, required=True)
    parser.add_argument("--iters", type=int, default=7000)
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()

    import torch
    if not torch.cuda.is_available():
        sys.exit("needs an NVIDIA graphics card with CUDA: torch.cuda.is_available() = False")
    from gsplat import rasterization

    torch.manual_seed(args.seed)
    rng = np.random.default_rng(args.seed)
    images, viewmats, ks, width, height = load_views(args.cameras)
    if args.init == "random":
        init = random_init([float(c) for c in args.bbox.split(",")], args.budget, rng)
    else:
        init = ply.read(args.init)
    params = to_params(init)
    extent = float(torch.linalg.norm(params["means"].max(0).values - params["means"].min(0).values)) / 2

    means_lr = 1.6e-4 * extent
    lrs = {"means": means_lr, "scales": 5e-3, "quats": 1e-3, "opacities": 5e-2, "sh0": 2.5e-3}
    optimizers = {k: torch.optim.Adam([{"params": params[k], "lr": lr, "name": k}], eps=1e-15)
                  for k, lr in lrs.items()}
    # Positions decay to 1% of their rate over the run, as in the gsplat reference trainer.
    means_sched = torch.optim.lr_scheduler.ExponentialLR(optimizers["means"], gamma=0.01 ** (1 / max(args.iters, 1)))
    strategy = safe_mcmc(cap_max=len(init), refine_start_iter=min(500, args.iters // 4),
                         refine_stop_iter=int(args.iters * 0.8), refine_every=100)
    strategy.check_sanity(params, optimizers)
    state = strategy.initialize_state()

    started = time.perf_counter()
    for step in range(args.iters):
        i = int(rng.integers(len(images)))
        colors, _, info = rasterization(
            means=params["means"], quats=params["quats"], scales=torch.exp(params["scales"]),
            opacities=torch.sigmoid(params["opacities"]), colors=params["sh0"], sh_degree=0,
            viewmats=viewmats[i:i + 1], Ks=ks[i:i + 1], width=width, height=height,
        )
        pred, gt = colors[0], images[i]
        loss = 0.8 * (pred - gt).abs().mean() + 0.2 * (1 - ssim(pred, gt))
        # MCMC regularizers: discourage large or needlessly opaque Gaussians.
        loss = loss + 0.01 * torch.sigmoid(params["opacities"]).mean() + 0.01 * torch.exp(params["scales"]).mean()
        loss.backward()
        for opt in optimizers.values():
            opt.step()
            opt.zero_grad(set_to_none=True)
        means_sched.step()
        strategy.step_post_backward(params, optimizers, state, step, info, lr=means_sched.get_last_lr()[0])
        if (step + 1) % PROGRESS_EVERY == 0 or step == args.iters - 1:
            print(f"progress {step + 1} {args.iters} loss {loss.item():.4f}", flush=True)
    seconds = time.perf_counter() - started

    ply.write(args.out, to_gaussians(params))
    json.dump({"seconds": seconds, "iters": args.iters, "count": len(params["means"])},
              open(os.path.splitext(args.out)[0] + ".train.json", "w"))
    print(f"trained {args.iters} iterations in {seconds:.1f} s -> {args.out}", flush=True)


if __name__ == "__main__":
    main()
