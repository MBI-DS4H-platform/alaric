deploymentdir=$1
output=1b7f-frag4-fwd

seamless-run -vvv -y --conda alaric --dry --write-remote-job "$deploymentdir" \
--metavar outdir=/ramscratch/$output/files -i code/poses.py 'python3 code/organize.py --compress  --local-tempdir --local-stagedir $outdir' 