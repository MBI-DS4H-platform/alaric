from __future__ import annotations

import argparse
from pathlib import Path

try:
    from .checksum import write_array_sidecar, write_pose_sidecars
    from ..npy_io import find_npy
except ImportError:  # direct script execution with ALARIC_DIR on PYTHONPATH
    from middle.checksum import write_array_sidecar, write_pose_sidecars
    from npy_io import find_npy


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("kind", choices=("array", "pose"))
    parser.add_argument(
        "path",
        help="Result path. An array is named logically (score.npy / mask.npy); the "
        "compressed form is resolved here, and the sidecar keeps the logical name.",
    )
    args = parser.parse_args(argv)
    path = Path(args.path)
    if args.kind == "array":
        # The action decides whether to compress its array; the generated script names
        # it either way, so resolve the name rather than teaching the deployer about it.
        print(write_array_sidecar(find_npy(path) or path))
    else:
        print(write_pose_sidecars(path))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
