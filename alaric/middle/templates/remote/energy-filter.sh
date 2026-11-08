# One action owns the whole undecided set.  The Python runner chunks scorer calls
# internally, while its cascade threshold is evaluated over the whole action.
# Node-local scratch for the prepared minfor arrays and grids.
export TMPDIR="${ALARIC_REMOTE_SCRATCH_DIR:-${TMPDIR:-/tmp}}"
{{ python }} {{ alaric_dir }}/energy_filter_action.py \
  --pose-dir {{ input_result_path }} \
  --protein {{ protein_path }} \
  --sequence {{ sequence }} \
  --fragment {{ fragment }} \
  --protein-range {{ protein_range }} \
  --threshold {{ threshold }} \
  --epsilon {{ epsilon }} \
  --cascade {{ cascade_json }} \
  --min-cascade {{ min_cascade }} \
  {{ exclude_args }} \
  --nprocs {{ nprocs }} \
  --chunk-poses {{ chunk_poses }} \
  --output {{ output_path }}
