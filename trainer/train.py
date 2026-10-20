"""The trainer moved into the add-on (mesh2gs/train.py) so the M2 / M3 buttons can run it;
this path keeps old commands working."""

import os
import runpy

runpy.run_path(os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "mesh2gs", "train.py"),
               run_name="__main__")
