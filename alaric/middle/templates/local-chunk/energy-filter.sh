# Each chunk speculatively runs every cascade step and the gridless pass over its own
# global pose range and records the decided pose ids. organize.sh replays the cascade
# on whole-action counts (min_cascade is never applied per chunk), so the result and
# its statistics do not depend on nchunks.
{{ python }} {{ alaric_dir }}/energy_filter_chunk.py chunk \
  --nchunks {{ nchunks }} \
  --chunk-index {{ chunk_index }} \
  --chunks-dir {{ chunks_path }} \
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
  --chunk-poses {{ chunk_poses }}
### ORGANIZE ###
# Replays the whole-action cascade from the chunk records; only poses added by an early
# whole-action exit (fewer than min_cascade) are scored here.
{{ python }} {{ alaric_dir }}/energy_filter_chunk.py assemble \
  --nchunks {{ nchunks }} \
  --chunks-dir {{ chunks_path }} \
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
