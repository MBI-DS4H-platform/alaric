# frag4-bwd
set -euo pipefail
cd "$(dirname "$0")"
rm -rf frag4-bwd
python3 code/stack.py --sequence GU --protein pdbs/1b7f_dom2.pdb \
  --pdb-exclude 1b7f \
  --resid 256 --second \
  --angle 30 --dihedral 45 -45 \
  --conformer-range 1 100 --rotamer-range 1 1000 \
  --nprocs 10 \
  --output frag4-bwd/
python3 code/organize.py frag4-bwd/ --nprocs 8 --compress
