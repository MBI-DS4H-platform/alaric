NCHUNKS={{ nchunks }}
chunk_range() {
  local total=$1 idx=$2 n=$3
  local first=$(( (total * (idx - 1)) / n + 1 ))
  local last=$(( (total * idx) / n ))
  echo "$first $last"
}
TOTAL=$({{ python }} - <<"PY"
from poses import PoseReader
print(PoseReader.get_nposes({{ input_result_path }}))
PY
)
for IDX in $(seq 1 "$NCHUNKS"); do
  read FIRST LAST < <(chunk_range "$TOTAL" "$IDX" "$NCHUNKS")
  {{ python }} {{ alaric_dir }}/grow.py \
    --source-poses {{ input_result_path }} \
    --source-sequence {{ source_sequence }} \
    --target-sequence {{ target_sequence }} \
    --direction {{ direction }} \
    --crmsd {{ crmsd }} \
    --ov-rmsd {{ ovrmsd }} \
    --output {{ output_path }} \
    {{ exclude_args }} \
    --bucket-size 16 \
    --pose-range "$FIRST" "$LAST" \
    --unorganized-subdirs \
    ${ALARIC_GROW_EXTRA_ARGS:-}
done
{{ organize_command }}
