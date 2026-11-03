"""Test-only A5 launcher that rejects undeclared scientific imports."""
from __future__ import annotations

import importlib.machinery
from pathlib import Path
import runpy
import sys


forbidden = Path(sys.argv[1].removeprefix("forbid=")).resolve()
script = Path(sys.argv[2]).resolve()
sys.argv = [str(script), *sys.argv[3:]]
sys.path.insert(0, str(script.parent))
sys.path.insert(0, str(Path.cwd()))


class DeclaredImportsOnly:
    def find_spec(self, fullname, path=None, target=None):
        if fullname.split(".")[0] in {
            "jax", "jaxlib", "scipy", "nb_kernel", "forcefields", "alaric", "util",
            "grid_io", "grid_generator", "reproduce_grid_score",
        }:
            raise ImportError(f"A5 imported forbidden backend: {fullname}")
        spec = importlib.machinery.PathFinder.find_spec(fullname, path, target)
        if spec is not None and spec.origin not in (None, "built-in", "frozen"):
            origin = Path(spec.origin).resolve()
            if forbidden == origin or forbidden in origin.parents:
                raise ImportError(f"A5 imported undeclared checkout code: {fullname}: {origin}")
        return None


sys.meta_path.insert(0, DeclaredImportsOnly())
runpy.run_path(str(script), run_name="__main__")
