# Export generations E1 results

## Scope

Implemented Alaric E1 items C1–C3 from `export-gens-plan.md`:

- Added strict `apply-mask` schema, dependency and output kinds, graph checks for the
  selected source side, matching sequence/fragment metadata, backend dispatch, and local,
  remote, local-chunk, and remote-chunk templates. The runtime reads `mask1.npy` or
  `mask2.npy` from the paired-mask result directory, validates its one-dimensional boolean
  shape and length, checks a source-order fingerprint recorded in `mask-pair.json`, filters
  in source order, and writes pose provenance.
- Made `identity.input2` optional. The explicit `--single-input INPUT OUTPUT` path performs
  bucket-wise deduplication with one full archive read per source file and writes `map-1`.
  The existing two-input path and positional CLI are unchanged for distinct dependency result
  sigils. When both dependencies resolve to the same result sigil, the operation uses the
  single-pass deduplicator and emits both legacy maps. The hashed identity parameter record
  adds `same_input_dedup_version: 1` for this same-result case, preventing reuse of a prior
  duplicate-producing self-intersection cache while leaving single-input records marker-free.
- Centralized schema-1 constraints validation in `middle/resolve.py`. It validates fragment
  keys and RNA sequences, adjacent unique pair addresses, pair/fragment sequence agreement,
  positive finite thresholds with at most two decimal places, and non-conflicting lowercase
  or documented uppercase threshold keys. Requested fragment sequences must be present.
- Updated the `tests/1b7f-project/DATA/constraints.json` fixture to the strict schema-1 shape
  and documented that shape in `documents/middle-level-plan.md`. This fixture keeps its
  fragment sequences and pair values rounded to two decimal places; legacy score/fit fields
  are not part of the current constraints contract.
- Single-input identity rejects a missing or unorganized pool with no organized archive
  entries, while accepting a valid organized zero-pose archive.
- Updated the schema-1 fixture and action-key documentation in `tests/middle/test_middle_core.py`,
  `documents/middle-level.txt`, and `plans/alaric-project-layout.md`.

## Verification

Environment: `alaric` conda environment.

Command:

```sh
conda run --live-stream -n alaric pytest -q tests/middle/test_export_e1_contract.py tests/test_export_e1_actions.py
```

Final result after the same-result cache-version marker and focused regression test:
**95 passed in 0.59s** (exit code 0).

The initial run of the same command had 89 passed and 4 failures. Three identified missing
implementation cases (single-input pool classification, `python -m` imports, and repeated
records split across same-pool files) were fixed. The fourth was an invalid empty-pool test
fixture that attempted to write a zero-pose archive through a writer that rejects empty
archives; A1 replaced it with a valid zero-pose archive header. A later review also required
single-input identity to reject a directory with no organized archive entries; A1 added a
regression test for that case while retaining the valid zero-pose archive case. A1 also added
a focused test confirming the cache-version marker appears only when both dependency sigils
are equal, while the distinct-input saved record and one-input record remain unchanged. The
final scoped rerun passed after those corrections.

The contract tests cover graph wiring, selector validation and sigils, side/source matching,
local and remote generated scripts, strict constraints consumers, the same-result cache marker,
and the saved distinct-input two-input sigil. Runtime tests cover both mask sides, malformed
masks, source-order changes, CLI use, single-input deduplication and archive-read counts, and
unchanged distinct-input identity artifacts (pose archives, maps, and manifest).

## Regression verification

In addition to the scoped E1 set, A4 ran these selected regressions in `alaric`:

```sh
conda run --live-stream -n alaric pytest -q tests/middle/test_middle_core.py tests/middle/test_anchor_refe_action.py tests/middle/test_pool_graph.py::test_dedup_records_both_maps_but_one_lineage tests/middle/test_propagate_provenance.py::test_merge_map_is_selected_by_next_directory_sigil
```

Result: **42 passed in 2.11s**. Pytest emitted the existing best-effort
`[seamless.close] called via atexit...` warning.

```sh
conda run --live-stream -n alaric pytest -q tests/1b7f/test_organize_zstd.py::test_identity_filter_can_write_compressed_pose_output tests/1b7f/test_organize_zstd.py::test_identity_filter_compresses_its_map_arrays
```

Result: **2 passed in 0.09s**.

A4 first attempted the same regression selection without `PYTHONPATH`:

```sh
conda run --live-stream -n alaric pytest -q tests/middle/test_middle_core.py tests/middle/test_anchor_refe_action.py tests/middle/test_pool_graph.py::test_dedup_records_both_maps_but_one_lineage tests/middle/test_propagate_provenance.py::test_merge_map_is_selected_by_next_directory_sigil tests/legacy-middle-actions/test_pose_filter.py::test_filter_keeps_input_order_and_records_provenance tests/legacy-middle-actions/test_pose_filter.py::test_filter_is_independent_of_worker_count tests/legacy-middle-actions/test_pose_filter.py::test_select_with_a_boolean_mask_needs_no_order_array
```

Collection failed with `ModuleNotFoundError: No module named 'npy_io'` because those tests
insert the absent `tests/legacy-middle-actions/.alaric` checkout path. Retrying with the
Alaric source directory on `PYTHONPATH` collected the suite, but the four selected legacy
pose-filter cases failed because their subprocesses invoke missing
`tests/legacy-middle-actions/.alaric/filter-poses.py` or
`tests/legacy-middle-actions/.alaric/select-poses.py` scripts:

```sh
conda run --live-stream -n alaric env PYTHONPATH=/home/agent/alaric/alaric pytest -q tests/middle/test_middle_core.py tests/middle/test_anchor_refe_action.py tests/middle/test_pool_graph.py::test_dedup_records_both_maps_but_one_lineage tests/middle/test_propagate_provenance.py::test_merge_map_is_selected_by_next_directory_sigil tests/legacy-middle-actions/test_pose_filter.py::test_filter_keeps_input_order_and_records_provenance tests/legacy-middle-actions/test_pose_filter.py::test_filter_is_independent_of_worker_count tests/legacy-middle-actions/test_pose_filter.py::test_select_with_a_boolean_mask_needs_no_order_array
```

Result: **42 passed, 4 failed in 2.47s**. The failed cases were
`test_filter_keeps_input_order_and_records_provenance[1]`,
`test_filter_keeps_input_order_and_records_provenance[3]`,
`test_filter_is_independent_of_worker_count`, and
`test_select_with_a_boolean_mask_needs_no_order_array`. `ls` confirmed the `.alaric` path
and scripts were absent.

After the cache-version marker was added, A4 reran the exact scoped E1 command:

```sh
conda run --live-stream -n alaric pytest -q tests/middle/test_export_e1_contract.py tests/test_export_e1_actions.py
```

Result: **95 passed in 0.56s**. A4 also ran the named sigil regression:

```sh
conda run --live-stream -n alaric pytest -q tests/middle/test_middle_core.py::test_sigil_is_deterministic_and_writes_parameters
```

Result: **1 passed in 0.09s**. Both emitted only the existing best-effort Seamless atexit warning.

## Acceptance checks not run

- No representative 1B7F project or large-pool performance run was made.
- Generated local/remote scripts were inspected by the contract tests, but a full deployed
  action run against local or remote result caches was not performed.
