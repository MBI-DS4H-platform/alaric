# This version is for the MBI cluster (using ramscratch to store the results)

# It uses Seamless to launch the job, but don't count on it to provide caching for you.

set -u -e

deploymentdir=$1
output=1b7f-frag4-bwd
seamless-run -vvv -y --conda alaric --dry --write-remote-job "$deploymentdir" \
--metavar outdir=/ramscratch/$output -I code/stack.py.DEPS.txt \
--prologue "rm -rf \$outdir" \
"""python -u code/stack.py --sequence GU --protein pdbs/1b7f_dom2.pdb \
  --pdb-exclude 1b7f \
  --resid 256 --second \
  --angle 25 --dihedral 45 -45 \
  --output \$outdir/files \
  && seamless-checksum-index /ramscratch/$output/files && \
  mkdir /ramscratch/$output/bufferdir
  seamless-upload -y --hardlink --destination /ramscratch/$output/bufferdir /ramscratch/$output/files
  cat /ramscratch/$output/files.INDEX
"""
