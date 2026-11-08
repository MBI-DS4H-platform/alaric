# Retired middle-action examples

These archived project fragments document legacy `score`, `filter`, and `score_add` configurations. Standalone `score` has since been restored as a grid-score action; `filter` and `score_add` remain retired. `energy-filter` takes one protein range per action, with multiple ranges chained as successive AND filters. The `score_add` fixtures encoded a sum and had no separate per-range thresholds, so they are kept only as historical source. Tests for the retired score-threshold `filter-poses.py` utility are archived here too; `select-poses.py` remains for explicit mask/index selection.

Current executable replacements use energy-filter fixtures in `tests/middle/test_pool_graph.py` and the W8 contract fixtures in `tests/fixtures/w8/`. The 1b7f anchor-refe directory retains its non-retired reference threshold YAMLs consumed by `tests/test_w2_contract.py`; it is no longer a runnable full project.
