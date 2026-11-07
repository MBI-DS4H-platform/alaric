# Export E2 result record

E2 adds the pure snapshot normalizer, admission report, X5 warning calculation, and canonical digest in Preparic's `export_snapshot.py`.

## Targeted test

Command:

```sh
conda run --live-stream -n preparic pytest -q tests/backend/test_export_snapshot.py
```

Result after contract review and focused regressions: **51 passed** in 0.07 s. An intermediate rerun after adding malformed filter-name mutations exposed shared-list aliasing in the test fixture; the fixture now copies `GAUSSIAN_NAMES` for each producer pin, and the final scoped run passes.

The test set covers named producer-shaped pin fixtures, schema-1 admission, ordered X5 warnings without re-rounding (including implied and below-RRM-95 results), addressed hard blocks, stale structure/range/fit/stacking echoes, exact prepared-RNA and clip agreement, matrix-derived interaction runs/anchors, canonical RNA base identity, exact T22 row coverage, strict helper validation, boolean/overflow rejection, digest canonicalization/order invariance and byte/version sensitivity, and exclusion of validator sentinels and view state.

## Regression selection

Command:

```sh
conda run --live-stream -n preparic pytest -q tests/backend/test_export_snapshot.py tests/backend/test_w45_stage5_acceptance.py::test_handoff_contains_only_kept_rows_and_actual_candidate_anchors tests/backend/test_w45_stage5_acceptance.py::test_m4_roundtrip_separated_joined_and_cross_range_runs tests/backend/test_w7_stage6_backend.py::test_b08_t19_rejects_malformed_active_class_rows tests/backend/test_w7_stage6_backend.py::test_w8a_kept_candidate_alone_controls_jobs_classes_and_gaussian_validation tests/backend/test_w3_stage4.py::test_t10_refuses_missing_duplicate_or_extra_pair_rows tests/backend/test_w7_stage6_backend.py::test_b10_implied_rejection_is_non_green_and_has_no_invented_fill tests/backend/test_w7_stage6_backend.py::test_b05_worst_unclipped_score_sets_class_and_clipped_candidate_is_ignored
```

After the final malformed-Gaussian parser guards and fixture isolation, A4 reran the same
selection. Final result: **64 passed** in 0.40 s.

## Acceptance checks not run in E2

- Real Seamless Context or spawn-worker execution of either export transformer; transformer wiring is E5.
- HTTP download behavior, pending/failed cell handling, changed and equal-value PUT invalidation, or the configured download proxy; the E0 Context probe covers those assumptions, and E5 exercises the wired export.
- Browser checks, archive rendering, Alaric action-schema parsing, generator recipe tests, or end-to-end extracted-project execution; these belong to E3–E7.
- Validation against a live mounted structure and the complete 1B7F pipeline. E2's tests use deterministic schema-shaped fixtures.
