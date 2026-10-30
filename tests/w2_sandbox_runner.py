"""Test-only CLI launcher: reject undeclared imports from the source checkout."""
from __future__ import annotations

import importlib.machinery
from pathlib import Path
import runpy
import sys


forbidden = Path(sys.argv[1].removeprefix("forbid=")).absolute()
script = Path(sys.argv[2]).absolute()
sys.argv = [str(script), *sys.argv[3:]]
sys.path.insert(0, str(script.parent))
sys.path.insert(0, str(Path.cwd()))


class NoHiddenCheckoutImports:
    def find_spec(self, fullname, path=None, target=None):
        spec = importlib.machinery.PathFinder.find_spec(fullname, path, target)
        if spec is not None and spec.origin not in (None, "built-in", "frozen"):
            origin = Path(spec.origin).absolute()
            if origin == forbidden or forbidden in origin.parents:
                raise ImportError(f"undeclared checkout import: {fullname} from {origin}")
        return None


sys.meta_path.insert(0, NoHiddenCheckoutImports())
runpy.run_path(str(script), run_name="__main__")
