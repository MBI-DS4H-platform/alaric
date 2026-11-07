# Export gens E7 results

**Scope:** E7 end-to-end acceptance, selected catalogue extraction and deployment, bounded 1B7F action execution, replacement of the synthetic W8 export-bundle checks, and final contract/usage documentation.

## New acceptance tests

```text
conda run --live-stream -n preparic pytest -q tests/backend/test_export_e7_acceptance.py
```

**Result:** 67 passed. The test uses the real E2 pin-shaped fixture to verify exact prepared bytes, X1/X2 identities, X3 sentinels and per-gen deduplication, X6 non-overridable blocks, and digest/current-tree behavior.

From `/home/agent/preparic`:

```text
PYTHONPATH=/home/agent/alaric:/home/agent/preparic conda run --live-stream -n alaric pytest -q tests/backend/test_export_e7_selected_projects.py
```

**Result:** 4 passed. The tests extract the committed §10 catalogue, select compatible Gaussian and reference anchor gens, resolve real Alaric graphs, verify fixed/unfixed scoring, merges and common-conformer masks, and deploy selected actions locally.

## Regression tests

From `/home/agent/alaric`:

```text
conda run --live-stream -n alaric pytest -q tests/test_export_e1_actions.py tests/middle/test_export_e1_contract.py tests/middle/test_anchor_refe_action.py::test_disabled_anchor_refe_fails_at_runtime_threshold_validation tests/middle/test_anchor_refe_action.py::test_disabled_positive_control_resolves_and_deploys_without_blocking_siblings
```

**Result:** 97 passed.

```text
conda run --live-stream -n alaric pytest -q tests/test_w8_actions_contract.py tests/test_w8_energy_filter_integration.py
```

**Result:** 36 passed, 3 opt-in tests skipped.

```text
conda run --live-stream -n alaric pytest -q tests/test_w8_contract_followup.py tests/test_w6_anchor_emission_parity.py::test_real_1b7f_a9_emission_matches_anchor_action_for_all_filters
```

**Result:** 8 passed, with one existing `PytestUnknownMarkWarning`.

```text
conda run --live-stream -n preparic pytest -q tests/backend/test_export_download.py tests/backend/test_export_backend_bindings.py
```

**Result:** 11 passed. The real attached-wrapper HTTP test compared every downloaded ZIP member byte-for-byte with the rendered tree and verified that blocked admission returns HTTP 409.

```text
conda run --live-stream -n preparic npx vitest run tests/frontend/export.test.ts tests/frontend/export-download.test.ts
```

**Result:** 30 passed.

## Bounded 1B7F runtime checks

The opt-in real energy-filter integration passed across the local and local-chunk templates:

```text
ALARIC_W8_INTEGRATION=1 conda run --live-stream -n alaric pytest -q tests/test_w8_energy_filter_integration.py::test_actual_energy_filter_bytes_ids_and_stats_across_templates
```

**Result:** 1 passed in 67.99 seconds; two existing attract-jax parameter-division warnings were emitted.

The reference-anchor best-fit bracket check passed:

```text
conda run --live-stream -n alaric pytest -q tests/1b7f/test_anchor_refe.py::test_anchor_refe_brackets_the_best_fitting_reference_pose
```

**Result:** 1 passed in 3.14 seconds.

Using only temporary `/tmp/e7-1b7f-*` outputs, bounded Gaussian anchors emitted 5,419 GU poses and 4,907 UU poses from 20-rotamer searches. A forward grow using 500 source poses, target conformer 3995, 20 rotamers, seed 25, cRMSD 2.0 and ovRMSD 1.73 produced 14,182 unrestricted rows and 28 restricted rows. Scoring the target anchor pool retained 4,843 poses. Two-input identity retained 8 unique poses; single-input identity reduced the restricted grow to the same 8. Comparing `(conformer, rotamer, grid translation)` keys found equal nonempty sets: no restricted-only or intersection-only poses. The restricted output's 28 rows contained 20 duplicates.

The exact bounded runtime commands were:

```bash
conda run --live-stream -n alaric python alaric/anchor.py \
  --protein tests/1b7f/pdbs/1b7f_dom2.pdb --resid 256 \
  --sequence GU --second --conformer 1976 --rotamer-range 1 20 \
  --pdb-exclude 1b7f \
  --precalculated-anchor precomputed-filters/gaussian/rrm95 \
  --precalculated-filter-name rrm95 \
  --anchor-reference-ring ring-refe/refe-C.npy \
  --output /tmp/e7-1b7f-anchor-source --nprocs 1 --cache-size 1000

conda run --live-stream -n alaric python alaric/anchor.py \
  --protein tests/1b7f/pdbs/1b7f_dom2.pdb --resid 256 \
  --sequence UU --first --conformer 3995 --rotamer-range 1 20 \
  --pdb-exclude 1b7f \
  --precalculated-anchor precomputed-filters/gaussian/rrm95 \
  --precalculated-filter-name rrm95 \
  --anchor-reference-ring ring-refe/refe-C.npy \
  --output /tmp/e7-1b7f-anchor-target --nprocs 1 --cache-size 1000

conda run --live-stream -n alaric python alaric/grow.py \
  --source-poses /tmp/e7-1b7f-anchor-source \
  --source-sequence GU --target-sequence UU --direction forward \
  --pose-range 1 500 --conformer 3995 --test-rotamers 20 --test-seed 25 \
  --crmsd 2.0 --ov-rmsd 1.73 --pdb-exclude 1b7f --nprocs 1 \
  --cache-size 1000 --output /tmp/e7-1b7f-grow-seed25-unrestricted

conda run --live-stream -n alaric python alaric/grow.py \
  --source-poses /tmp/e7-1b7f-anchor-source \
  --source-sequence GU --target-sequence UU --direction forward \
  --pose-range 1 500 --conformer 3995 --test-rotamers 20 --test-seed 25 \
  --crmsd 2.0 --ov-rmsd 1.73 --pdb-exclude 1b7f \
  --restrict-poses /tmp/e7-1b7f-anchor-target --nprocs 1 \
  --cache-size 1000 --output /tmp/e7-1b7f-grow-seed25-restricted

conda run --live-stream -n alaric python alaric/energy_filter_action.py \
  --pose-dir /tmp/e7-1b7f-anchor-target \
  --protein tests/1b7f/pdbs/1b7f_dom2.pdb --sequence UU --fragment 5 \
  --protein-range range-2 --threshold 1000000000 --epsilon 15 \
  --cascade '[]' --min-cascade 0 --pdb-exclude 1b7f --nprocs 1 \
  --chunk-poses 500 --output /tmp/e7-1b7f-anchor-target-scored

conda run --live-stream -n alaric python alaric/identity_filter.py \
  /tmp/e7-1b7f-grow-seed25-unrestricted \
  /tmp/e7-1b7f-anchor-target-scored \
  /tmp/e7-1b7f-grow-seed25-scored-intersection --compress

conda run --live-stream -n alaric python alaric/identity_filter.py \
  --single-input /tmp/e7-1b7f-grow-seed25-restricted \
  /tmp/e7-1b7f-grow-seed25-restricted-unique --compress
```

All runtime outputs were temporary.

No Preparic services were started for E7. Final process checks found no Vite/backend process or listeners on ports 5173/5813. E6's read-only Chrome DevTools browser smoke is recorded in [the E6 results](export-gens-e6-results.md). A3's E7 contract review found no remaining breach after the doc corrections. The old `export_bundle.py` module, its synthetic fixture and export-only tests were removed after their useful assertions moved to real pin/tree tests; independent W8 action tests remain.
