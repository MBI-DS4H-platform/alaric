{{ python }} {{ alaric_dir }}/rmsd.py \
  {{ input_result_path }} \
  --reference {{ reference_path }} \
  --fragment {{ fragment }} \
  --outputfile {{ score_output_path }} \
  {{ exclude_args }}
