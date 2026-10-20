"""M2 / M3 from the add-on: render training views, run train.py in an external Python, read
the result back.

Blender's own Python has no PyTorch, so training runs in a separate interpreter that has
torch (CUDA) and gsplat, set in the add-on preferences. A job advances in small steps
(one view, or one look at the trainer's output) so a modal operator can keep the UI
responsive and show progress; run() does the same in one go for background mode.
"""

import os
import shutil
import subprocess
import sys
import tempfile
import threading
import time

import bpy

from .core.gaussians import GaussianSet
from .i18n import tr
from .io import ply
from .views import ViewRenderer

TRAIN_SCRIPT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "train.py")
# A checkout of the repository with its development environment (the one the install button
# makes, in .cache/mesh2gs_env; the benchmark adds lpips and matplotlib to it).
DEV_PYTHON = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), ".cache", "mesh2gs_env",
                          "python", "python.exe" if sys.platform == "win32" else os.path.join("bin", "python3"))
NO_WINDOW = getattr(subprocess, "CREATE_NO_WINDOW", 0)
CHECK_CODE = ("import torch, gsplat; ok = torch.cuda.is_available(); "
              "print(torch.__version__, gsplat.__version__, "
              "torch.cuda.get_device_name(0) if ok else 'CUDA unavailable')")

# Setting up the training Python (the versions the benchmark ran with). gsplat's prebuilt
# wheels need no CUDA Toolkit; elsewhere `pip install gsplat` builds it with one.
SETUP_PYTHON = "3.10"
GSPLAT_WHEEL = ("https://github.com/nerfstudio-project/gsplat/releases/download/v1.5.3/"
                "gsplat-1.5.3%2Bpt24cu124-cp310-cp310-{}.whl")
SETUP_STEPS = (
    "1. Install Python 3.10 and the NVIDIA driver.",
    "2. Press Copy commands and run them in a terminal in any folder:",
    "    this makes mesh2gs-venv with PyTorch (CUDA 12.4) and gsplat (≈5 GB, a few minutes).",
    "3. Set the field above to mesh2gs-venv/Scripts/python.exe (Linux: mesh2gs-venv/bin/python).",
    "4. Press Check: the graphics card's model should appear.",
)


def setup_commands(platform=sys.platform):
    """Shell commands that make a training Python in ./mesh2gs-venv."""
    torch = ("-m pip install torch==2.4.1+cu124 torchvision==0.19.1+cu124 "
             "--index-url https://download.pytorch.org/whl/cu124")
    if platform == "win32":
        py = r"mesh2gs-venv\Scripts\python"
        return [f"py -{SETUP_PYTHON} -m venv mesh2gs-venv",
                f"{py} {torch}",
                f'{py} -m pip install {GSPLAT_WHEEL.format("win_amd64")} "numpy<2" pillow packaging']
    py = "mesh2gs-venv/bin/python"
    return [f"python{SETUP_PYTHON} -m venv mesh2gs-venv",
            f"{py} {torch}",
            f'{py} -m pip install {GSPLAT_WHEEL.format("linux_x86_64")} "numpy<2" pillow packaging']


ACTIVE = None   # the running TrainJob, if any; the panel shows its progress
CHECKED = {}    # python path -> (ok, message) of the last check, shown in the preferences


def python_path(context=None):
    """The Python set in the preferences, else the repository's development environment if any."""
    context = context or bpy.context
    addon = context.preferences.addons.get(__package__)
    path = bpy.path.abspath(addon.preferences.python) if addon and addon.preferences.python else ""
    if not path and os.path.isfile(DEV_PYTHON):
        path = DEV_PYTHON
    return path


def check_python(path, timeout=120):
    """(ok, message): can `path` import torch and gsplat and see a CUDA device?"""
    if not path or not os.path.isfile(path):
        return False, tr("Python file not found")
    try:
        done = subprocess.run([path, "-c", CHECK_CODE], capture_output=True, text=True,
                              timeout=timeout, creationflags=NO_WINDOW)
    except (OSError, subprocess.TimeoutExpired) as err:
        return False, str(err)
    if done.returncode != 0:
        lines = (done.stderr or done.stdout).strip().splitlines()
        return False, lines[-1] if lines else tr("exit code {}").format(done.returncode)
    out = done.stdout.strip()
    return "CUDA unavailable" not in out, out


PANORAMA_SCRIPT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "panorama.py")


def scene_panorama(context, scene, eye, width=2048, timeout=600):
    """Render the splat scene `scene` (GaussianSet, Blender world space) as a 360-degree
    panorama from `eye` in the external Python; returns the PNG path (remove_panorama
    deletes it)."""
    python = python_path(context)
    if not python or not os.path.isfile(python):
        raise RuntimeError(tr("no Python with PyTorch and gsplat is set (Mesh2GS add-on preferences)"))
    folder = tempfile.mkdtemp(prefix="mesh2gs_pano_")
    scene_path, out = os.path.join(folder, "scene.ply"), os.path.join(folder, "panorama.png")
    try:
        ply.write(scene_path, scene)
        done = subprocess.run([python, PANORAMA_SCRIPT, scene_path, out, "--width", str(width),
                               "--eye=" + ",".join(f"{c:.6f}" for c in eye)],  # '=': values may start with '-'
                              capture_output=True, text=True, encoding="utf-8", errors="replace",
                              timeout=timeout, creationflags=NO_WINDOW)
        if done.returncode != 0 or not os.path.isfile(out):
            lines = (done.stderr or done.stdout).strip().splitlines()
            raise RuntimeError(lines[-1] if lines else tr("panorama: exit code {}").format(done.returncode))
    except (RuntimeError, OSError, subprocess.TimeoutExpired) as err:
        shutil.rmtree(folder, ignore_errors=True)
        raise RuntimeError(str(err)) from err
    os.remove(scene_path)
    return out


def remove_panorama(path):
    shutil.rmtree(os.path.dirname(path), ignore_errors=True)


class TrainJob:
    """One M2 / M3 run. Stages: VIEWS → TRAIN → DONE, or FAILED / CANCELLED."""

    def __init__(self, context, method, sources, budget, iterations, *, init=None, bbox=None,
                 views=64, size=512, lit=False, samples=64, emission=True, python=None):
        self.method, self.budget, self.iterations = method, budget, iterations
        self.source_names = [o.name for o in sources]
        self.python = python or python_path(context)
        self.dir = tempfile.mkdtemp(prefix="mesh2gs_train_")
        self.out = os.path.join(self.dir, "result.ply")
        self.started = time.perf_counter()
        self.stage, self.error, self.views_done, self.steps_done = "VIEWS", "", 0, 0
        self.process, self.log = None, []
        if init is not None:
            self.init = os.path.join(self.dir, "init.ply")
            ply.write(self.init, init)  # Blender world space, as the views
            self.args = ["--init", self.init]
        else:
            self.args = ["--init", "random", "--bbox=" + ",".join(f"{c:.6f}" for c in bbox)]
        try:
            self.renderer = ViewRenderer(sources, context.evaluated_depsgraph_get(), self.dir,
                                         count=views, size=size, lit=lit, samples=samples,
                                         emission=emission)
        except Exception:
            shutil.rmtree(self.dir, ignore_errors=True)
            raise

    # -- progress ---------------------------------------------------------------------
    @property
    def running(self):
        return self.stage in ("VIEWS", "TRAIN")

    @property
    def fraction(self):
        if self.stage == "VIEWS":
            return self.views_done / self.renderer.count
        return self.steps_done / self.iterations if self.stage == "TRAIN" else 1.0

    @property
    def text(self):
        if self.stage == "VIEWS":
            return tr("{}: views {}/{}").format(self.method, self.views_done, self.renderer.count)
        if self.stage == "TRAIN":
            if self.steps_done == 0:
                return tr("{}: starting PyTorch…").format(self.method)
            return tr("{}: training {}/{}").format(self.method, self.steps_done, self.iterations)
        return f"{self.method}: {self.stage.lower()}"

    @property
    def seconds(self):
        return time.perf_counter() - self.started

    # -- work -------------------------------------------------------------------------
    def step(self):
        """Advance a little: render one view, or check on the trainer."""
        try:
            if self.stage == "VIEWS":
                self.renderer.render(self.views_done)
                self.views_done += 1
                if self.views_done == self.renderer.count:
                    self._start_trainer()
            elif self.stage == "TRAIN":
                self._poll_trainer()
        except Exception as err:  # noqa: BLE001 - any failure ends the job with a message
            self._fail(str(err))

    def run(self):
        """The whole job, blocking (background mode, scripts)."""
        while self.running:
            self.step()
            if self.stage == "TRAIN":
                self.process.wait()
                self._reader.join()
                self._poll_trainer()
        return self.stage == "DONE"

    def _start_trainer(self):
        cameras = self.renderer.write_index()
        self.renderer.close()
        if not self.python or not os.path.isfile(self.python):
            raise RuntimeError(tr("no Python with PyTorch and gsplat is set (Mesh2GS add-on preferences)"))
        cmd = [self.python, "-u", TRAIN_SCRIPT, "--cameras", cameras, "--out", self.out,
               "--budget", str(self.budget), "--iters", str(self.iterations), *self.args]
        self.process = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True,
                                        encoding="utf-8", errors="replace", creationflags=NO_WINDOW)
        self._reader = threading.Thread(target=self._read_output, daemon=True)
        self._reader.start()
        self.stage = "TRAIN"

    def _read_output(self):
        for line in self.process.stdout:
            line = line.rstrip()
            parts = line.split()
            if len(parts) >= 3 and parts[0] == "progress":
                self.steps_done = int(parts[1])
            else:
                self.log = (self.log + [line])[-20:]

    def _poll_trainer(self):
        code = self.process.poll()
        if code is None:
            return
        if code == 0 and os.path.isfile(self.out):
            self.stage = "DONE"
        else:
            lines = [l for l in self.log if l.strip()]
            self._fail(lines[-1] if lines else tr("the trainer exited with code {}").format(code))

    def _fail(self, message):
        self.error, self.stage = message, "FAILED"
        self._stop()

    def cancel(self):
        if self.running:
            self.stage = "CANCELLED"
        self._stop()

    def _stop(self):
        if self.renderer.scene is not None:
            self.renderer.close()
        if self.process is not None and self.process.poll() is None:
            self.process.terminate()

    def result(self) -> GaussianSet:
        return ply.read(self.out)

    def cleanup(self):
        self._stop()
        if self.process is not None:
            try:
                self.process.wait(timeout=10)
            except subprocess.TimeoutExpired:
                pass
        shutil.rmtree(self.dir, ignore_errors=True)
