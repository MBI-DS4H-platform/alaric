# Middle-action fixtures

`score`, `filter`, and `score_add` are supported actions. `score` computes per-pose scores, `filter` selects poses from scores or an explicit mask/index array, and `score_add` combines score arrays. `energy-filter` provides a separate cascading energy-threshold action; it does not retire those actions.

The action configurations and generated graphs for `1b7f-anchor-refe`, `1b7f-gaussian-anchor-all-unbound`, and `1b7f-gaussian-anchor-nori-unbound` are restored to their original directories under `tests/`. The score-threshold and mask/index CLI tests are restored to `tests/1b7f/test_filter_zstd.py` and `tests/1b7f/test_pose_filter.py`, using the existing `.alaric` source link.

The remaining project snapshots in this directory were moved here by the W8 change. Their location does not indicate that their actions are retired. Additional energy-filter fixtures live in `tests/middle/test_pool_graph.py` and `tests/fixtures/w8/`.
