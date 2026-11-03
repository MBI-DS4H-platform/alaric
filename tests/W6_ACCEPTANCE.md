# W6 acceptance: Gaussian anchor validation

Run the W6 checks in the `alaric` environment. A7/A8 are an accepted prerequisite: inspect the committed `alaric/gaussian_stacking_filter.json`, the verifier and the installed `precomputed-filters/gaussian/<name>/` arrays; do not run either generator or rewrite the JSON/checksums during W6.

## Regression commands

Focused W6 and anchor-refactor checks:

```sh
conda run -n alaric python -m pytest -q \
  tests/test_w6_gaussian_check.py \
  tests/test_w6_gaussian_sandbox.py \
  tests/test_w6_anchor_precalc.py \
  tests/test_w6_anchor_constants.py \
  tests/test_w6_anchor_emission_parity.py \
  tests/test_anchor_precalculated.py \
  tests/test_gaussian_verify.py \
  tests/test_gaussian_caches.py \
  tests/middle/test_w6_anchor_filter_identity.py
```

The mirror check runs in Preparic and reads Alaric's Python constants without importing the Alaric package:

```sh
(cd ../preparic && npm test -- tests/frontend/w6-anchor-constants.test.ts)
```

The A7/A8 checks above verify all twelve installed cache entries. The W6 tests must additionally prove that A9 consumes the exact resolved paths returned by `gaussian_cache.verify_filter_cache`, returns installation failures with the generator command, and never silently generates missing files.

Recompressed-cache semantic identity and the actual 1b7f A9/anchor-action parity:

```sh
conda run -n alaric python -m pytest -q \
  tests/test_w6_gaussian_check.py::test_a9_recompressed_cache_preserves_semantic_identity_and_decision \
  tests/test_w6_anchor_emission_parity.py::test_real_1b7f_a9_emission_matches_anchor_action_for_all_filters
```

Fresh-process spawn check for A9 (the main A9 fixture runs the full list in a clean spawned worker):

```sh
conda run -n alaric python -m pytest -q \
  tests/test_w6_gaussian_sandbox.py::test_a9_cli_runs_1b7f_job_with_declared_dependencies_in_seamless_sandbox
```

Cold timing acceptance (prints the job count, `.npy` cache form, fresh-spawn status, environment and elapsed seconds for capture):

```sh
ALARIC_RUN_W6_COLD_TIMING=1 conda run -n alaric python -m pytest -q -s \
  tests/test_w6_gaussian_check.py::test_a9_twenty_jobs_cold_spawn_under_five_seconds
```

Run the isolated 1b7f anchor-action parity acceptance after the focused tests:

```sh
conda run -n alaric python -m pytest -q \
  tests/1b7f/test_anchor_refe.py \
  tests/1b7f/test_rmsd_refe_best_fit.py
```

## Required oracles

- **Cache resolution and identity:** exercise verified `.npy`, compressed `.npy.zst`, and recompressed `.npy.zst` files. A recompressed file with matching decompressed `npy_sha256` is accepted and returns `recompressed`; the logical identity and A9 result remain unchanged when only compressed `sha256` changes. If both forms exist, a corrupt preferred `.npy` fails even beside a valid `.zst`. Missing or corrupt arrays raise `GaussianCacheInstallationError` with the exact generator command. Mutation tests use temporary directories and never touch installed caches.
- **Semantic identities:** changing compressed bytes alone leaves `spec_digest`, `npy_sha256` identities and A9 decisions unchanged. Changing an authoritative array identity or semantic spec digest changes the anchor/A9 identity. Include declared code and runtime inputs in the transformation identity; keep compression-only verification metadata out of it.
- **Twelve-filter behavior:** read the legal names, exact five-decimal cutoffs, applied strict `d < 5 Å` / `φ < 60°` mask and prefilters from the committed JSON. Verify all twelve names and delivered score/decision goldens; do not add a second cutoff table. A high score outside the applied mask fails `pose_pass`.
- **Containment and implied rows:** derive containment from JSON and evaluate from broad to strict. For every candidate, emitted/`pose_pass` are monotone under containment. Once a containing filter does not emit, each unvisited contained entry is explicitly `{emitted: false, implied_by: <containing filter>}` and carries no invented `count`, `capacity`, `fill` or `score`. If the containing filter emits, continue into its children.
- **Anchor parity and capacity:** on representative 1b7f jobs, A9 `emitted` exactly matches actual anchor-action emission for all twelve filters. Every emitted anchor pose satisfies `count / capacity >= 1/3`; emission uses `>=`, including equality. The shared `candidate_cells` and `cell_fill` helpers are tested against the action path.
- **A11 constants:** `L_OV == sqrt(3)` Å (not the internal 1.3 Å loop bound); both `grow` and `anchor-refe` reject any requested `ovrmsd > L_OV` and accept the exact boundary. The boundary recompute tolerance remains `1e-5 Å`. `MIN_CELL_FRACTION == 1/3` is shared with Preparic. Standalone `anchor.py` may override its default through `ALARIC_ANCHOR_MIN_CELL_FRACTION`; A9 stays pinned to 1/3, and Preparic/export configuration must leave the override unset.
- **Per-job failures:** an incomplete ring and a best-fit reconstruction error greater than `0.002 Å` create an `error` for that job while other valid jobs still return filter rows. They do not abort the complete job list or turn an implied rejection into a measured result.
- **Cold spawned timing:** launch A9 from a fresh Python interpreter using a `spawn` worker and no persistent Seamless result cache, hashserver or database. Use installed, already decompressed `.npy` arrays for this timed pass so decompression is excluded. Include verification, array loading, capacity construction and all filter evaluations in the measurement. With about 20 representative 1b7f jobs, the cold check must finish within five seconds. A separate compressed-cache check records decompression time but does not include it in that budget. The spawn test must not depend on a warm module, worker or process cache.

## Evidence to retain

Keep the pytest output for the commands above and record the cold timing with job count, cache form (`.npy` for the timed run), Python/environment, and whether it was a new spawned process. Include one recompressed-cache run showing unchanged semantic identity and decisions, one preferred-corrupt-`.npy` installation failure, and the twelve-filter 1b7f emission-parity result. Do not commit generated cache files.

## Fresh A1 selection (design only; no execution evidence)

The focused set below adds the latest shared-verifier executor and selective
library-loading regressions, standalone CLI filter-name guards, and A9's own
pose predicate against the committed descriptor goldens. It never invokes a
Gaussian generator. `test_gaussian_verify.py` is deliberately omitted here:
its `built` fixture calls `gaussian_precalc.build_filter`; temporary tiny arrays
in `test_w6_runtime_regressions.py` cover verification without that prerequisite
rebuild.

```sh
ALARIC_GAUSSIAN_CACHES_REQUIRED=1 conda run -n alaric python -m pytest -q \
  tests/test_w6_runtime_regressions.py \
  tests/test_w6_gaussian_check.py \
  tests/test_w6_gaussian_sandbox.py \
  tests/test_w6_anchor_precalc.py \
  tests/test_w6_anchor_constants.py \
  tests/test_w6_anchor_cli.py \
  tests/test_w6_anchor_emission_parity.py \
  tests/test_anchor_precalculated.py \
  tests/test_gaussian_caches.py \
  tests/middle/test_w6_anchor_filter_identity.py
```

Run the separate Preparic constants mirror, explicit cold-timing invocation and
1b7f commands already listed above. Missing installed caches must fail this
acceptance run rather than silently skip. No test in this design pass has been
run; the commands become evidence only when their output is captured.

Remaining coverage limits to report with results: the semantic-identity tests
mutate spec/array identities, but do not exercise a complete A9 transformation
identity under every runtime/code/reference-ring mutation. Recompression tests
exercise a real broad-filter rejection; actual action parity evaluates one
representative 1b7f job across all twelve names, while the spawned A9 fixture
covers twenty jobs. A separate compressed-only elapsed/decompression measurement
is not currently emitted by the suite. The sandbox CLI fixture submits one job;
the separate clean spawned fixture submits the full twenty-two-row list.
