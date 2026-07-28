{{ python }} {{ alaric_dir }}/anchor_refe.py \
  --reference {{ reference_path }} \
  --fragment {{ fragment }} \
  --sequence {{ sequence }} \
  {{ nucleotide_flag }} \
  --ov-rmsd {{ ovrmsd }} \
  --output {{ output_path }} \
  {{ exclude_args }} \
  --bucket-size 16 \
  --unorganized-subdirs \
  ${ALARIC_ANCHOR_REFE_EXTRA_ARGS:-}
{{ organize_command }}
