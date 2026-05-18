#!/usr/bin/env bash
# Determinism: organized .arc must be byte-identical regardless of
# --nprocs / --cache-size. Both configs must hash to the SAME committed,
# blessed frag4-fwd checksum -- which also re-checks correctness for free.
#
# This intentionally tests against the checksum rather than diffing the two
# runs against each other: a run-to-run diff only proves "stable", while
# matching the blessed checksum proves "stable AND still correct". If this
# ever fails, build run-to-run byte instrumentation at that point (e.g.
# `diff -r frag4-det-a frag4-det-b`) to localize the nondeterminism.
set -euo pipefail
here=$(cd "$(dirname "$0")" && pwd)
cd "$here"

expected=$(awk '{print $1}' ground-truth/frag4-fwd.arc.CHECKSUM)

run() {  # <outdir> <cache-size> <nprocs>
  rm -rf "$1"
  python3 code/stack.py --sequence GU --protein pdbs/1b7f_dom2.pdb \
    --pdb-exclude 1b7f \
    --resid 214 --first \
    --angle 30 --dihedral 45 -45 \
    --test-conformers 100 --test-rotamers 1000 \
    --cache-size "$2" --nprocs "$3" \
    --output "$1/"
  python3 code/organize.py "$1/" --nprocs "$3"
}

status=0
run frag4-det-a 250000 1
run frag4-det-b 100000 4
for o in frag4-det-a frag4-det-b; do
  actual=$(bash arc-checksum.sh "$o")
  if [ "$actual" = "$expected" ]; then
    echo "$o: OK ($actual)"
  else
    echo "$o: MISMATCH expected=$expected actual=$actual"
    status=1
  fi
done
exit $status
