from __future__ import annotations

import argparse
from pathlib import Path

try:
    from .checksum import write_array_sidecar, write_pose_sidecars
except ImportError:  # direct script execution with ALARIC_DIR on PYTHONPATH
    from middle.checksum import write_array_sidecar, write_pose_sidecars


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("kind", choices=("array", "pose"))
    parser.add_argument("path")
    args = parser.parse_args(argv)
    path = Path(args.path)
    if args.kind == "array":
        print(write_array_sidecar(path))
    else:
        print(write_pose_sidecars(path))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
