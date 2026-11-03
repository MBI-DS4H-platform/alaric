# Cascade decisions are synchronized over all currently undecided poses inside
# energy_filter_action.py, including when its scorer evaluates bounded chunks.
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
