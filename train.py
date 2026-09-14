"""Compatibility launcher for the canonical training script.

Use ``train_improve.py`` as the official training entry point. This file is
kept so existing commands do not silently run a second configuration.
"""

import runpy
from pathlib import Path


if __name__ == "__main__":
    runpy.run_path(str(Path(__file__).with_name("train_improve.py")), run_name="__main__")
