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
