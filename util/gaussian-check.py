#!/usr/bin/env python3
"""Declared-dependency CLI for Alaric's all-filter Gaussian check."""

from __future__ import annotations

import sys
from pathlib import Path


def _dependency_directory() -> Path:
    """Prefer Seamless's staged cwd, then support a direct source-tree run."""
    script_dir = Path(__file__).resolve().parent
    candidates = (
        Path.cwd(),
        script_dir,
        script_dir.parent / "alaric",
    )
    for candidate in candidates:
        if (candidate / "gaussian_check.py").is_file():
            return candidate
    raise FileNotFoundError("gaussian_check.py is not present in the declared dependencies")


dependency_dir = _dependency_directory()
if str(dependency_dir) not in sys.path:
    sys.path.insert(0, str(dependency_dir))

from gaussian_check import main  # noqa: E402


if __name__ == "__main__":
    raise SystemExit(main())
