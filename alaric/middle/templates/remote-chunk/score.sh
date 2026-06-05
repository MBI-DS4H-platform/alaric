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
mkdir -p "$PWD/chunks/chunk-${IDX}"
bash {{ alaric_dir }}/score.sh \
  {{ score_exclude_args }} \
  {{ input_result_path }} \
  "$FIRST" \
  "$LAST" \
  {{ sequence }} \
  {{ protein_path }} \
  {{ nb_kernel }} \
  "$PWD/chunks/chunk-${IDX}/score.npy"
### ORGANIZE ###
{{ python }} - <<"PY"
from pathlib import Path
import os
import numpy as np
chunks = sorted(Path("chunks").glob("chunk-*"), key=lambda p: int(p.name.split("-")[1]))
arrays = [np.load(p / "score.npy") for p in chunks]
out = Path(os.path.expandvars({{ output_path_python }}))
out.mkdir(parents=True, exist_ok=True)
np.save(out / "score.npy", np.concatenate(arrays) if arrays else np.empty((0,), dtype=float))
PY
