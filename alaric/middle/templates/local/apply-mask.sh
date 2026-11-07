{{ python }} {{ alaric_dir }}/apply_mask.py \
  {{ input_result_path }} \
  {{ mask_pair_result_path }} \
  {{ output_path }} \
  --mask {{ mask_side }} \
  --force \
  --compress
