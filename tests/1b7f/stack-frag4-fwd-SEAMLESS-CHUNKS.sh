set -u -e

deploymentdir=$1
output=1b7f-frag4-fwd
libsize=7740

first=-1
chunk=1
for last in $(python -c '
import sys; mx = int(sys.argv[1]); inc = int(sys.argv[2])
for n in list(range(0,mx,inc)) + [mx]: 
    print(n)
' 7740 1000); do
  echo $last
  if [ $first -gt 0 ]; then
    echo $first $last
seamless-run -vvv -y --conda alaric --dry --write-remote-job "$deploymentdir/chunk$chunk" \
--metavar outdir=/ramscratch/$output -I code/stack.py.DEPS.txt \
"""python -u code/stack.py --sequence GU --protein pdbs/1b7f_dom2.pdb \
  --pdb-exclude 1b7f \
  --resid 214 --first \
  --angle 25 --dihedral 45 -45 \
  --conformer-range $first $last \
  --output \$outdir/files \
"""
    ((chunk++))
  fi
  first=$last
  ((first+=1))
  echo $first
done
exit


#Todo: organize
#  && seamless-checksum-index /ramscratch/$output/files && \
#  mkdir /ramscratch/$output/bufferdir
#  seamless-upload -y --hardlink --destination /ramscratch/$output/bufferdir /ramscratch/$output/files
#  cat /ramscratch/$output/files.INDEX
