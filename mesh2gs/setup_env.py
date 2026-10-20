"""One-button setup of the training Python for M2 / M3 and the splat-scene light.

gsplat ships prebuilt CUDA wheels for Python 3.10 only, and Blender bundles a newer Python,
so the add-on brings its own: a standalone CPython 3.10 build (python-build-standalone,
checked against its published SHA-256), then PyTorch with CUDA 12.4 and gsplat into it.
About 5 GB, no system Python and no administrator rights needed. pip runs without a cache
and with its temporary files inside the target folder, so nothing lands on the system drive.

The work runs in a background thread (download, unpack) and in pip subprocesses; the
preferences poll `progress` and `text` and can cancel it.
"""

import hashlib
import os
import shutil
import subprocess
import sys
import tarfile
import threading
import time
import urllib.request

from .i18n import tr

PBS = "https://github.com/astral-sh/python-build-standalone/releases/download/20260929/"
PYTHONS = {  # platform -> (archive, sha256, interpreter inside the unpacked "python" folder)
    "win32": ("cpython-3.10.21%2B20260929-x86_64-pc-windows-msvc-install_only.tar.gz",
              "806de086b36a83ad2863a9d6c2945de105f707fded114e248e8974a2dc64b7eb", "python.exe"),
    "linux": ("cpython-3.10.21%2B20260929-x86_64-unknown-linux-gnu-install_only.tar.gz",
              "b6c7d6cff8e70f143b5bbd47c2bb7ecc8353ee70ef30fe4ff79bfbe7179624f7", os.path.join("bin", "python3")),
}
TORCH = ["torch==2.4.1+cu124", "--index-url", "https://download.pytorch.org/whl/cu124"]
GSPLAT = "https://github.com/nerfstudio-project/gsplat/releases/download/v1.5.3/gsplat-1.5.3%2Bpt24cu124-cp310-cp310-{}.whl"
WHEEL_PLATFORM = {"win32": "win_amd64", "linux": "linux_x86_64"}
EXTRAS = ["numpy<2", "pillow", "packaging"]  # gsplat's MCMC strategy imports packaging without requiring it
NEEDED_GB = 9.0      # PyTorch unpacks (≈4.5 GB) next to its downloaded wheel (≈2.4 GB)
EXPECTED_GB = 5.0    # what the installed environment takes (measured: 5.0 GB on Windows)
NO_WINDOW = getattr(subprocess, "CREATE_NO_WINDOW", 0)

INSTALLER = None     # the running or last EnvInstaller


def supported(platform=sys.platform):
    return platform in PYTHONS


def interpreter(folder, platform=sys.platform):
    return os.path.join(folder, "python", PYTHONS[platform][2])


def free_gb(folder):
    """Free space where `folder` is (or would be created)."""
    probe = os.path.abspath(folder)
    while not os.path.exists(probe):
        parent = os.path.dirname(probe)
        if parent == probe:
            break
        probe = parent
    return shutil.disk_usage(probe).free / 1e9


def folder_size_gb(folder):
    total = 0
    for root, _, files in os.walk(folder):
        for name in files:
            try:
                total += os.path.getsize(os.path.join(root, name))
            except OSError:
                pass
    return total / 1e9


class EnvInstaller:
    """Stages: DOWNLOAD → UNPACK → TORCH → GSPLAT → CHECK → DONE, or FAILED / CANCELLED."""

    STAGES = ("DOWNLOAD", "UNPACK", "TORCH", "GSPLAT", "CHECK")

    def __init__(self, folder, platform=sys.platform):
        self.folder, self.platform = os.path.abspath(folder), platform
        self.stage, self.error, self.result = "DOWNLOAD", "", ""
        self.fraction = 0.0           # within the current stage, when known
        self.started = time.perf_counter()
        self.log = []
        self._cancel = threading.Event()
        self._process = None
        self._thread = threading.Thread(target=self._run, daemon=True)

    # -- state the UI reads --------------------------------------------------------------
    @property
    def running(self):
        return self.stage in self.STAGES

    @property
    def python(self):
        return interpreter(self.folder, self.platform)

    @property
    def progress(self):
        """Overall 0..1: the downloads dominate, so they get most of the bar."""
        weights = {"DOWNLOAD": (0.0, 0.03), "UNPACK": (0.03, 0.05), "TORCH": (0.05, 0.85),
                   "GSPLAT": (0.85, 0.95), "CHECK": (0.95, 1.0)}
        if self.stage not in weights:
            return 1.0
        lo, hi = weights[self.stage]
        return lo + (hi - lo) * min(self.fraction, 1.0)

    @property
    def text(self):
        step = self.STAGES.index(self.stage) + 1 if self.running else len(self.STAGES)
        names = {"DOWNLOAD": tr("downloading Python 3.10"), "UNPACK": tr("unpacking Python"),
                 "TORCH": tr("installing PyTorch (≈2.5 GB)"), "GSPLAT": tr("installing gsplat"),
                 "CHECK": tr("checking the graphics card")}
        if self.running:
            return tr("{}/{}: {}").format(step, len(self.STAGES), names[self.stage])
        return {"DONE": tr("Done"), "CANCELLED": tr("Cancelled")}.get(self.stage, tr("Failed"))

    @property
    def seconds(self):
        return time.perf_counter() - self.started

    # -- work ----------------------------------------------------------------------------
    def start(self):
        if not supported(self.platform):
            raise RuntimeError(tr("only Windows and Linux on x86-64 with an NVIDIA graphics card"))
        free = free_gb(self.folder)
        if free < NEEDED_GB:
            raise RuntimeError(tr("{:.1f} GB free on that drive, {:.0f} GB needed").format(free, NEEDED_GB))
        os.makedirs(self.folder, exist_ok=True)
        self._thread.start()

    def cancel(self):
        self._cancel.set()
        process = self._process
        if process is not None and process.poll() is None:
            process.terminate()

    def wait(self, timeout=None):
        self._thread.join(timeout)

    def _run(self):
        try:
            if not os.path.isfile(self.python):
                self._download_python()
                self._unpack_python()
            self._set("TORCH")
            self._pip(*TORCH, watch=EXPECTED_GB)
            self._set("GSPLAT")
            self._pip(GSPLAT.format(WHEEL_PLATFORM[self.platform]), *EXTRAS)
            self._set("CHECK")
            from .training import check_python  # its own messages are translated lookups only
            ok, message = check_python(self.python, timeout=300)
            if not ok:
                raise RuntimeError(message)
            self.result = message
            self.stage = "DONE"
        except Exception as err:  # noqa: BLE001 - any failure ends the run with a message
            self.stage = "CANCELLED" if self._cancel.is_set() else "FAILED"
            self.error = str(err)
        finally:
            shutil.rmtree(os.path.join(self.folder, "tmp"), ignore_errors=True)

    # The methods below run in the worker thread: their messages stay in English (no bpy
    # calls off the main thread) and the preferences translate them when drawing.
    def _set(self, stage):
        if self._cancel.is_set():
            raise RuntimeError("Cancelled")
        self.stage, self.fraction = stage, 0.0

    def _download_python(self):
        archive, sha256, _ = PYTHONS[self.platform]
        path = os.path.join(self.folder, "python.tar.gz")
        digest = hashlib.sha256()
        with urllib.request.urlopen(PBS + archive, timeout=60) as response, open(path, "wb") as out:
            total = int(response.headers.get("Content-Length") or 0)
            done = 0
            while chunk := response.read(1 << 20):
                if self._cancel.is_set():
                    raise RuntimeError("Cancelled")
                out.write(chunk)
                digest.update(chunk)
                done += len(chunk)
                self.fraction = done / total if total else 0.0
        if digest.hexdigest() != sha256:
            os.remove(path)
            raise RuntimeError("the downloaded Python does not match its checksum")

    def _unpack_python(self):
        self._set("UNPACK")
        path = os.path.join(self.folder, "python.tar.gz")
        shutil.rmtree(os.path.join(self.folder, "python"), ignore_errors=True)
        with tarfile.open(path) as archive:
            try:
                archive.extractall(self.folder, filter="data")
            except TypeError:  # Python without extraction filters
                archive.extractall(self.folder)
        os.remove(path)

    def _pip(self, *packages, watch=None):
        """pip install into the new Python, temporary files and no cache inside the folder.
        With `watch` (GB), progress follows how much the folder has grown."""
        tmp = os.path.join(self.folder, "tmp")
        os.makedirs(tmp, exist_ok=True)
        env = dict(os.environ, TMP=tmp, TEMP=tmp, TMPDIR=tmp, PIP_NO_CACHE_DIR="1",
                   PIP_DISABLE_PIP_VERSION_CHECK="1", PYTHONNOUSERSITE="1")
        cmd = [self.python, "-m", "pip", "install", "--no-warn-script-location", "--progress-bar", "off", *packages]
        start = folder_size_gb(self.folder) if watch else 0.0
        self._process = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True,
                                         encoding="utf-8", errors="replace", env=env, creationflags=NO_WINDOW)
        reader = threading.Thread(target=self._read, args=(self._process,), daemon=True)
        reader.start()
        while self._process.poll() is None:
            if watch:
                self.fraction = min(0.99, (folder_size_gb(self.folder) - start) / watch)
            time.sleep(1.0)
        reader.join()
        if self._cancel.is_set():
            raise RuntimeError("Cancelled")
        if self._process.returncode != 0:
            lines = [l for l in self.log if l.strip()]
            raise RuntimeError(lines[-1] if lines else f"pip exited with code {self._process.returncode}")

    def _read(self, process):
        for line in process.stdout:
            self.log = (self.log + [line.rstrip()])[-30:]
