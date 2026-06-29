NCHUNKS={{ nchunks }}
IDX={{ chunk_index }}
chunk_range() {
  local total=$1 idx=$2 n=$3
  local first=$(( (total * (idx - 1)) / n + 1 ))
  local last=$(( (total * idx) / n ))
  echo "$first $last"
}
TOTAL={{ nconformers }}
SINGLE_CONFORMER={{ single_conformer }}
if [ -n "$SINGLE_CONFORMER" ]; then
  FIRST="$SINGLE_CONFORMER"
  LAST="$SINGLE_CONFORMER"
else
  read FIRST LAST < <(chunk_range "$TOTAL" "$IDX" "$NCHUNKS")
fi
# Non-load-bearing tuning knobs (uncomment to override; they never change the result):
anchor_opts=(
#  --nprocs 8
#  --cache-size 100000000
#  --poselock 4
)
{{ python }} {{ alaric_dir }}/anchor.py \
  --protein {{ protein_path }} \
  --resid {{ resid }} \
  --sequence {{ sequence }} \
  --dihedral {{ dihedral_args }} \
  --angle {{ angle }} \
  --margin {{ margin }} \
  {{ nucleotide_flag }} \
  --output {{ output_path }} \
  {{ exclude_args }} \
  --bucket-size 16 \
  --conformer-range "$FIRST" "$LAST" \
  --unorganized-subdirs \
  ${anchor_opts[@]+"${anchor_opts[@]}"}
### ORGANIZE ###
{{ organize_command }}
