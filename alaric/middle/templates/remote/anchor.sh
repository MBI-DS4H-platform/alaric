{{ python }} {{ alaric_dir }}/anchor.py \
  --protein {{ protein_path }} \
  --resid {{ resid }} \
  --sequence {{ sequence }} \
  --dihedral {{ dihedral_args }} \
  --angle {{ angle }} \
  --margin {{ margin }} \
  {{ nucleotide_flag }} \
  --output {{ output_path }} \
  {{ exclude_args }} \
  --bucket-size 16 \
  --unorganized-subdirs \
  ${ALARIC_ANCHOR_EXTRA_ARGS:-}
{{ organize_command }}
