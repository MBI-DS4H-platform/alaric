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
#   export SCORE_BATCH_SIZE=100000
CHUNK_DIR={{ score_chunks_path }}/chunk-${IDX}
rm -rf "$CHUNK_DIR"
mkdir -p "$CHUNK_DIR"
bash {{ alaric_dir }}/score.sh \
  {{ score_exclude_args }} \
  {{ input_result_path }} \
  "$FIRST" \
  "$LAST" \
  {{ sequence }} \
  {{ protein_path }} \
  {{ nb_kernel }} \
  "$CHUNK_DIR/score.npy"
### ORGANIZE ###
{{ score_concat_command }}
