bash {{ alaric_dir }}/score.sh \
  {{ score_exclude_args }} \
  {{ input_result_path }} \
  {{ sequence }} \
  {{ protein_path }} \
  {{ nb_kernel }} \
  {{ score_output_path }}
