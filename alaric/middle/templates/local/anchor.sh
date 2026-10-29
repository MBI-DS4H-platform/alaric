{{ python }} {{ alaric_dir }}/anchor.py \
  --protein {{ protein_path }} \
  --resid {{ resid }} \
  --sequence {{ sequence }} \
  {{ nucleotide_flag }} \
  --output {{ output_path }} \
  {{ exclude_args }} \
  {{ precalculated_args }} \
  --bucket-size 16 \
  ${ALARIC_ANCHOR_EXTRA_ARGS:-}
{{ organize_command }}
