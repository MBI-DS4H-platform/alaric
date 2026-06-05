{{ python }} {{ alaric_dir }}/grow.py \
  --source-poses {{ input_result_path }} \
  --source-sequence {{ source_sequence }} \
  --target-sequence {{ target_sequence }} \
  --direction {{ direction }} \
  --crmsd {{ crmsd }} \
  --ov-rmsd {{ ovrmsd }} \
  --output {{ output_path }} \
  {{ exclude_args }} \
  --bucket-size 16 \
  --unorganized-subdirs \
  ${ALARIC_GROW_EXTRA_ARGS:-}
{{ organize_command }}
