NCHUNKS={{ nchunks }}
IDX={{ chunk_index }}
chunk_range() {
  local total=$1 idx=$2 n=$3
  local first=$(( (total * (idx - 1)) / n + 1 ))
  local last=$(( (total * idx) / n ))
  echo "$first $last"
}
TOTAL=$({{ python }} - <<"PY"
from poses import PoseReader
import os
print(PoseReader.get_nposes(os.path.expandvars({{ input_result_python }})))
PY
)
read FIRST LAST < <(chunk_range "$TOTAL" "$IDX" "$NCHUNKS")
# Non-load-bearing tuning knobs (uncomment to override; they never change the result):
grow_opts=(
#  --nprocs 8
#  --cache-size 100000000
)
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
  ${grow_opts[@]+"${grow_opts[@]}"}
### ORGANIZE ###
{{ organize_command }}
