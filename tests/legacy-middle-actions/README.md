# Retired middle-action examples

These archived project fragments document the retired `score`, `filter`, and `score_add` middle actions. They are intentionally outside runnable project directories. W8 replaced that pipeline with `energy-filter`, which takes one protein range per action; multiple ranges are chained as successive AND filters. The retired `score_add` fixtures encoded a sum and had no separate per-range thresholds, so they are kept only as historical source. Tests for the retired score-threshold `filter-poses.py` utility are archived here too; `select-poses.py` remains for explicit mask/index selection.

Current executable replacements use energy-filter fixtures in `tests/middle/test_pool_graph.py` and the W8 contract fixtures in `tests/fixtures/w8/`. The 1b7f anchor-refe directory retains its non-retired reference threshold YAMLs consumed by `tests/test_w2_contract.py`; it is no longer a runnable full project.
