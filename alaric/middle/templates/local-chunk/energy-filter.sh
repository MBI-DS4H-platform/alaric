# Each chunk applies min_cascade to its own grid-eligible poses and records
# its accepted IDs and route counts. Assembly combines results without rescoring.
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
# Combine accepted IDs and actual route counts from the completed chunks.
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
  {{ nprocs_args }} \
  --chunk-poses {{ chunk_poses }} \
  --output {{ output_path }}
