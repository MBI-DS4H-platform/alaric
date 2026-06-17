#!/usr/bin/env python3
"""Count poses in an organized pose directory (or .arc files) using only headers.

Each .arc / .arc.zst file stores its pose count (nP) in the 21-byte header, so the
total is obtained without reading the pose payload.
"""

from __future__ import annotations

import argparse
from pathlib import Path
import sys

_CODE_DIR = Path(__file__).resolve().parent
if str(_CODE_DIR) not in sys.path:
    sys.path.insert(0, str(_CODE_DIR))

from poses import discover_organized, read_arc_header


def _arc_paths(target: Path) -> list[Path]:
    if target.is_dir():
        paths = discover_organized(target)
        if not paths:
            raise FileNotFoundError(
                f"No organized poses-*.arc files found in {target}"
            )
        return paths
    if not target.exists():
        raise FileNotFoundError(f"No such file or directory: {target}")
    return [target]


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "target",
        type=Path,
        help="Pose directory (organized poses-*.arc) or a single .arc/.arc.zst file",
    )
    parser.add_argument(
        "--per-file",
        action="store_true",
        help="Print the pose count for each .arc file as well as the total",
    )
    args = parser.parse_args()

    paths = _arc_paths(args.target)

    total = 0
    for path in paths:
        _, _, nposes, _ = read_arc_header(path)
        if args.per_file:
            print(f"{nposes}\t{path}")
        total += int(nposes)

    print(total)
    return 0


if __name__ == "__main__":
    sys.exit(main())
