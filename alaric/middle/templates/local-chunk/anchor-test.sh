NCHUNKS={{ nchunks }}
chunk_range() {
  local total=$1 idx=$2 n=$3
  local first=$(( (total * (idx - 1)) / n + 1 ))
  local last=$(( (total * idx) / n ))
  echo "$first $last"
}
TOTAL={{ nconformers }}
for IDX in $(seq 1 "$NCHUNKS"); do
  read FIRST LAST < <(chunk_range "$TOTAL" "$IDX" "$NCHUNKS")
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
    ${ALARIC_ANCHOR_EXTRA_ARGS:-}
done
{{ organize_command }}
