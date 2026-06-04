#!/usr/bin/env bash
# Print the canonical sha256 of a pose directory's organized output.
#
# Canonical hash = sha256 over the uncompressed bytes of the organized `.arc`
# files (poses-1.arc, poses-2.arc[.zst], ...) concatenated in
# `discover_organized` numeric order. `.arc.zst` is stream-decompressed first,
# so zstd framing/version choices do not affect the hash.
set -euo pipefail
here=$(cd "$(dirname "$0")" && pwd)
PYTHONPATH="$here/.alaric" python3 - "$1" <<'PY'
import hashlib
import sys

from poses import discover_organized

paths = discover_organized(sys.argv[1])
if not paths:
    sys.exit(f"no organized poses-*.arc files in {sys.argv[1]}")
h = hashlib.sha256()
for path in paths:
    if path.name.endswith(".arc.zst"):
        import zstandard as zstd

        with open(path, "rb") as compressed:
            with zstd.ZstdDecompressor().stream_reader(compressed) as handle:
                for block in iter(lambda: handle.read(1 << 20), b""):
                    h.update(block)
    else:
        with open(path, "rb") as handle:
            for block in iter(lambda: handle.read(1 << 20), b""):
                h.update(block)
print(h.hexdigest())
PY
