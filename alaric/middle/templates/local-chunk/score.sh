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
mkdir -p "$PWD/chunks"
for IDX in $(seq 1 "$NCHUNKS"); do
  read FIRST LAST < <(chunk_range "$TOTAL" "$IDX" "$NCHUNKS")
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
done
{{ python }} - <<"PY"
from pathlib import Path
import numpy as np
chunks = sorted(Path("chunks").glob("chunk-*"), key=lambda p: int(p.name.split("-")[1]))
arrays = [np.load(p / "score.npy") for p in chunks]
out = Path({{ output_path }})
out.mkdir(parents=True, exist_ok=True)
np.save(out / "score.npy", np.concatenate(arrays) if arrays else np.empty((0,), dtype=float))
PY
