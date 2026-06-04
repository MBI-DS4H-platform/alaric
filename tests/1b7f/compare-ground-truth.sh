#!/usr/bin/env bash
# Correctness gate: the organized .arc output of frag4-fwd/ and frag4-bwd/
# must hash to the committed alaric checksums.
#
# Run stack-frag4-fwd.sh and stack-frag4-bwd.sh first; this only verifies.
#
# The checksums in ground-truth/*.arc.CHECKSUM pin the canonical organized
# alaric output. A mismatch means alaric output changed; it does NOT by
# itself tell you which poses differ. To investigate, decode both outputs
# to sorted rows -- see ground-truth/README.md ("Debugging a mismatch").
set -euo pipefail
here=$(cd "$(dirname "$0")" && pwd)
cd "$here"

status=0
for d in fwd bwd; do
  expected=$(awk '{print $1}' "ground-truth/frag4-$d.arc.CHECKSUM")
  actual=$(bash arc-checksum.sh "frag4-$d")
  if [ "$actual" = "$expected" ]; then
    echo "frag4-$d: OK ($actual)"
  else
    echo "frag4-$d: MISMATCH"
    echo "  expected (blessed): $expected"
    echo "  actual:             $actual"
    echo "  -> see ground-truth/README.md 'Debugging a mismatch'"
    status=1
  fi
done
exit $status
