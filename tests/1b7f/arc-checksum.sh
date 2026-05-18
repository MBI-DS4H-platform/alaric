#!/usr/bin/env bash
# Print the canonical sha256 of a pose directory's organized output.
#
# Canonical hash = sha256 over the raw bytes of the organized plain `.arc`
# files (poses-1.arc, poses-2.arc, ...) concatenated in `discover_organized`
# numeric order. `.arc.zst` is never hashed: zstd is not byte-reproducible
# across versions/levels, which is exactly why `organize.py` emits plain
# `.arc` for organized output.
set -euo pipefail
here=$(cd "$(dirname "$0")" && pwd)
PYTHONPATH="$here/code" python3 - "$1" <<'PY'
import hashlib
import sys

from poses import discover_organized

paths = discover_organized(sys.argv[1])
if not paths:
    sys.exit(f"no organized poses-*.arc files in {sys.argv[1]}")
h = hashlib.sha256()
for path in paths:
    with open(path, "rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            h.update(block)
print(h.hexdigest())
PY
