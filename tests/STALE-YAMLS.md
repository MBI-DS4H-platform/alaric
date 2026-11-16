# Middle-action fixture locations

`score`, `filter`, and `score_add` are supported actions. The W8 change moved their examples into `tests/legacy-middle-actions/`; that move did not establish a retirement policy for these actions.

The action configurations and generated graphs for `tests/1b7f-anchor-refe/`, `tests/1b7f-gaussian-anchor-all-unbound/`, and `tests/1b7f-gaussian-anchor-nori-unbound/` have been restored to those project directories. Other project snapshots remain under `tests/legacy-middle-actions/`.

Score-threshold and explicit mask/index CLI tests live in `tests/1b7f/test_filter_zstd.py` and `tests/1b7f/test_pose_filter.py`. Additional energy-filter examples are in `tests/middle/test_pool_graph.py` and `tests/fixtures/w8/`.
