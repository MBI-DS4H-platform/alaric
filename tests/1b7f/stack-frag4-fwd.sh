# frag4-fwd
set -euo pipefail
cd "$(dirname "$0")"
rm -rf frag4-fwd
python3 code/stack.py --sequence GU --protein pdbs/1b7f_dom2.pdb \
  --pdb-exclude 1b7f \
  --resid 214 --first \
  --angle 30 --dihedral 45 -45 \
  --test-conformers 100 --test-rotamers 1000 \
  --output frag4-fwd/
python3 code/organize.py frag4-fwd/
