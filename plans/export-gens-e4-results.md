# Export gens E4 result

**Scope:** Preparic's pure `export_tree.render_tree(snapshot, recipes, digest)` renderer. It builds the relative folder-cell mapping, including exact input PDB buffers, schema-1 configuration and compact manifest, action YAML, and path collision checks.

**Command:**

```text
conda run --live-stream -n preparic pytest -q tests/backend/test_export_tree.py
```

**Result:** 26 passed, 2 skipped in 0.47s.

The skipped tests, `test_emitted_constraints_are_accepted_by_alaric_resolver` and `test_emitted_action_yamls_are_accepted_by_alaric_schema`, are Alaric consumer checks. A4 ran both in the `alaric` environment after the A3 review handoff. The Preparic suite covers exact member inventory and PDB byte pass-through, constraints and energy schemas, fixed score-filter data, manifest projection and producer choices, deterministic rerendering, disabled-reference comments/paths, digest mismatch, unsafe/duplicate/prefix-colliding paths, and root-specific generator path grammar including merge source qualifiers.

The initial renderer run reported a grammar error before detecting a generated-path prefix collision. The validator now checks duplicates and path-prefix conflicts across the catalogue before checking path semantics. A new negative test also uses a real source-qualified merge recipe from the divergent fixture. The final gated run above is green.

## A3 contract review

The final review checked the E4 renderer and tests against §§5.1–5.2/5.4 and the tree-path requirements: exact PDB bytes, constraints and energy JSON shapes, manifest inventory, action YAML comments, digest/reference checks, deterministic output, member-key safety, and root-specific path grammar. The tests also send the actual rendered `constraints.json` through `middle.resolve._constraints` and all rendered action YAML through `middle.schema.normalize_action` in the Alaric environment. No remaining E4 contract issue was identified.

**A4 existing regression names:** `test_schema1_constraints_lowercase_and_uppercase_aliases`, `test_invalid_constraints_refused_by_both_consumers`, `test_missing_requested_constraints_entry_does_not_use_fallback`.

## A4 verification

The final Preparic run and both Alaric consumer checks passed after A3's clearance.

```text
conda run --live-stream -n preparic pytest -q tests/backend/test_export_tree.py
```

**Result:** 26 passed, 2 skipped in 0.43s. The two Alaric consumer tests were run separately:

```text
conda run --live-stream -n alaric env PYTHONPATH=/home/agent/alaric:/home/agent/preparic pytest -q tests/backend/test_export_tree.py::test_emitted_constraints_are_accepted_by_alaric_resolver tests/backend/test_export_tree.py::test_emitted_action_yamls_are_accepted_by_alaric_schema
```

**Result:** 2 passed in 0.20s.

```text
conda run --live-stream -n alaric pytest -q tests/middle/test_export_e1_contract.py::test_schema1_constraints_lowercase_and_uppercase_aliases tests/middle/test_export_e1_contract.py::test_invalid_constraints_refused_by_both_consumers tests/middle/test_export_e1_contract.py::test_missing_requested_constraints_entry_does_not_use_fallback
```

**Result:** 46 passed in 0.11s.

**Not run in E4:** full project graph resolution/deployment, Context or spawn-worker execution, archive download/HTTP, live browser checks, and E5–E7 acceptance.
