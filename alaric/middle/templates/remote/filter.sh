# Node-local scratch for the filter pass: the kept-pose ids are spilled per input
# file and concatenated at the end, and a compressed score/mask input is staged here.
export TMPDIR="${ALARIC_REMOTE_SCRATCH_DIR:-${TMPDIR:-/tmp}}"
if [[ {{ filter_mode }} == mask ]]; then
  {{ python }} {{ alaric_dir }}/select-poses.py \
    {{ input_result_path }} \
    {{ mask_input_path }} \
    {{ output_path }} \
    --force \
    --compress
else
  {{ python }} {{ alaric_dir }}/filter-poses.py \
    {{ input_result_path }} \
    {{ score_input_path }} \
    {{ threshold }} \
    {{ output_path }} \
    --force \
    --compress
fi
