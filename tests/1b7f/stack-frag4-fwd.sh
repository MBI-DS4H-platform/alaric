# frag4-fwd
set -euo pipefail
cd "$(dirname "$0")"
rm -rf frag4-fwd
python3 .alaric/anchor.py --sequence GU --protein pdbs/1b7f_dom2.pdb \
  --pdb-exclude 1b7f \
  --resid 214 --first \
  --angle 30 --dihedral 45 -45 \
  --conformer-range 1 100 --rotamer-range 1 1000 \
  --nprocs 10 \
  --output frag4-fwd/
python3 .alaric/organize.py frag4-fwd/ --nprocs 8 --compress
