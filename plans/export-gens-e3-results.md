# Export gens E3 result

**Scope:** Preparic's pure generator planner, `export_gens.plan_gens`, with the committed §10 1B7F snapshot and catalogue golden.

**Command:**

```text
conda run --live-stream -n preparic pytest -q tests/backend/test_export_gens.py
```

**Result before A3 expansion:** 14 passed in 0.15s.

The suite checks the 46-gen §10 catalogue, canonical paths and pools, fixed/unfixed mode hoisting, anchor traversal alternatives, clipping and matrix boundaries, R11/R12 uniqueness and restrictions, fixed scoring order, merge inputs and endpoints, common-conformer masks, reference sentinels and descendants, multiple-stacking folds, action-schema fields, external producer contracts, and recipe DAG validation. The catalogue golden is committed under `preparic/tests/backend/data/export-gens/` and is derived independently from the plan.

**Expanded E3 rerun after A3 review:** The first run exposed an implementation naming error in the new three-stacking binary-fold case: the temporary fold was named `fragN-anchor-range2-raw-ident-1`, while the contract and test require `fragN-anchor-range2-ident-1`. The planner now names numbered temporaries from the anchor stem while preserving the unsuffixed raw pool as the final fold output.

```text
conda run --live-stream -n preparic pytest -q tests/backend/test_export_gens.py
```

**Result:** 15 passed in 0.14s. The expanded test also verifies three-range pair-qualified pools and mask-side mappings, three raw anchor stackings with deterministic binary folds, reference-candidate deduplication, and the distinct Gaussian/reference R12 restriction paths.

**K=2 inner-fragment regression rerun after A3 follow-up:** A3 identified that two inner `from-N` candidates with multiple stackings could reuse the same temporary fold name. Temporary folds now include the candidate step (for example, `frag4-anchor-from-4-ident-1` and `frag4-anchor-from-5-ident-1`); ordinary raw candidate folds retain `fragN-anchor-ident-i`.

```text
conda run --live-stream -n preparic pytest -q tests/backend/test_export_gens.py
```

**Result:** 16 passed in 0.16s, including the new K=2 collision regression.

**Boundary-merge R12 regression rerun after A3 follow-up:** When a multi-stacking Gaussian anchor pool and its deduplicated reference candidate choose different R12 paths, the planner now emits `anchor/` and `anchor-refe/` merge alternatives immediately before `gen-fragN`. The generic merge path remains for identical action graphs, including grown-target merges.

```text
conda run --live-stream -n preparic pytest -q tests/backend/test_export_gens.py
```

**Result:** 17 passed in 0.19s, including the new boundary-merge R12 path regression.

**Producer-choice regression rerun after A3 follow-up:** For a source-qualified boundary merge, only the anchor restriction pool's permitted producer paths are filtered to the matching `anchor/` or `anchor-refe/` tree. The grown-side pool retains all compatible producers.

```text
conda run --live-stream -n preparic pytest -q tests/backend/test_export_gens.py
```

**Result:** 17 passed in 0.18s, including the producer-choice assertions.

**Not run in E3:** Context/spawn-worker execution, Alaric YAML parsing or graph resolution, selected-run deployment/action execution, archive rendering/download, live browser checks, full Preparic/Alaric regression suites, and E4–E7 acceptance. Those remain for their planned phases.

## A3 contract review

The final review checked `export_gens.plan_gens` and the expanded fixture suite against R0–R12 and §§5.2–5.4, including external-frame contracts, candidate-qualified `from-N` folds, three-range common-conformer outputs, and conditional boundary-merge R12 variants with matching producer choices. No remaining E3 contract issue was identified. The conditional merge path extension is documented in `plans/export-gens-plan.md` R7 and §5.3/§5.4.

**A4 regression names:** `test_apply_mask_schema_graph_metadata_and_backend`, `test_apply_mask_sigil_contains_selector_and_both_dependencies`, `test_single_identity_has_one_dependency_and_inherits_metadata`, `test_two_input_identity_saved_sigil_fixture_is_unchanged`.

## A4 verification

The E3 suite and the four selected Alaric regressions passed after A3's final review. E3 ran in `preparic`; existing Alaric contract regressions ran in `alaric`.

```text
conda run --live-stream -n preparic pytest -q tests/backend/test_export_gens.py
```

**Result:** 17 passed in 0.16s.

```text
conda run --live-stream -n alaric pytest -q tests/middle/test_export_e1_contract.py::test_apply_mask_schema_graph_metadata_and_backend tests/middle/test_export_e1_contract.py::test_apply_mask_sigil_contains_selector_and_both_dependencies tests/middle/test_export_e1_contract.py::test_single_identity_has_one_dependency_and_inherits_metadata tests/middle/test_export_e1_contract.py::test_two_input_identity_saved_sigil_fixture_is_unchanged
```

**Result:** 5 passed in 0.10s (the schema/backend test is parameterized).
