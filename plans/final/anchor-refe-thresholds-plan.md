# W8b — computed `anchor-refe` thresholds, no editable bases

Draft, 2026-11-04. **Relation:** this is an amendment work package that follows [implementation-plan.md](implementation-plan.md), in the same way as its W8a section (§12). It implements the author's amendment of 2026-11-04 on `anchor-refe` thresholds, which is recorded in [oracle-contract.md](oracle-contract.md): the sources list, P4.2, P4.3, the 4→5 gate, P6.6, X3, X5, X6, D4 and D11. The amendment has also been applied to [state-transitions.md](state-transitions.md) and [contract-boundaries.md](contract-boundaries.md). The amendment also clarifies P4.2. W8b follows W8a and supersedes the earlier packages' requirements for editable bases, T23/S20 and reference-anchor validation; their other requirements and evidence remain in force. Its evidence feeds W9's documentation work (implementation-plan §13). The document edit changes no code: W8b is not yet implemented.

Status, 2026-11-07: W8b was implemented (Alaric `bb592b4`, preparic `c26a14a`) and documented in W9. Its export-seam proposal below is historical and superseded by E0–E7 in [export-gens-plan.md](../export-gens-plan.md); E7 retired the synthetic bundle API and transferred its useful assertions to real pin fixtures and rendered trees.

## 1. The amendment in brief

Stage 4 has three statistics:

| Statistic | Source | Role |
|---|---|---|
| Shared-nucleotide ovRMSD and cRMSD, per pair | A3/A4 `pairs` | the editable threshold table (S11); exported as `grow` thresholds |
| Dinucleotide best-fit `rmsd`, per fragment | A1 | read-only, shown; it informs the clip |
| Base ovRMSD of nucleotide 1 or 2, per anchor candidate | A3/A4 `bases` | computed, never edited; the `anchor-refe` thresholds |

`anchor-refe` is a positive control. Its computed thresholds are A4's rounded bases values, strictly greater than their raw values (G7). Nothing edited enters them. A threshold within L_ov is emitted unchanged; one above L_ov is replaced by the deliberately failing sentinel described below. The sentinel is not a rounded threshold and is not covered by G7's passing-pose guarantee.

The W8b plan originally described a short-circuit from S9 `bases` through browser-edited S11 bases and T23/S20 to a synthetic export seam. That wiring is historical: E0–E7 removed editable bases and T23/S20. In the delivered flow, fresh S9 bases feed T29's same-snapshot admission and export snapshot; S11 retains editable pairs, while S9 still supplies their Copy values and computed-value display/difference marks.

A candidate whose rounded computed value exceeds L_ov no longer blocks the export (X6). Instead its action is written with `ovrmsd: -1` and an explanatory comment (X3). Bases values never trigger "Proceed anyway" (X5). `grow` pair values above L_ov still block (D11).

## 2. Preparic backend

1. **S11 shape.**
   - `threshold_edits` becomes `{schema:1, basis, active, pairs}`.
   - Update the exact-key sets and the startup loader in `backend.py` (`_MARKER_KEYS`, the `threshold_edits` branch of the marker validator) and `stage4.py`'s `_validate_threshold_edits` (T10).
   - A stored file that still has `bases` is malformed. It loads as undecided with its diagnostics kept and is never migrated (G9, I7).
   - **S11 version (author clarification, 2026-11-04).** Keep `schema: 1`. This is unreleased code, and the author has deleted the existing S11 JSON files. This is an explicit exception to G8's version-bump rule for this change only; record the exception in this plan and nowhere else. I7 still governs refusal and diagnostics if a legacy-shaped file is supplied, but is not the reason for retaining schema 1.
2. **T11 `clip_fragments`.** Retain T11, as required by the amended boundary contract. It writes `thresholds.json` as `{schema:1, pairs:[...]}`, containing only the edited pairs whose two fragments are kept. The kept PDBs and `clip.json` are unchanged. T11 does not acquire an S9 dependency and remains an input to the delivered T29/T30 export.
3. **T23 and S20.**
   - Remove `_stage6_validate_refe_anchors`, its wiring (`context.validate_refe_anchors`), the `refe_validation` cell, its mount and its share.
   - Remove `validate_refe_anchors` from `stage6.py`, and the `refe-validation.json` mount path.
   - Remove the stage-6 banner scope for T23.
4. **S9.** Unchanged. It stays mounted and shared, because the gate needs its freshness and the export needs its `bases`.

## 3. Preparic frontend

The functional change is to remove the bases from the stage 4 table. The layout follow-up comes later, so keep everything else as it is.

1. **`src/lib/stage4.ts`.**
   - Drop `ThresholdBase` and the `bases` member of `ThresholdMarker`.
   - `makeThresholdSkeleton` creates pairs only.
   - `validateThresholdEdits` checks the exact keys `active,basis,pairs,schema` and the pair rows against S22.
   - `copyComputedThresholds`, the reconcile/"differs from computed" logic and `thresholdsCompleteForClip` handle pairs only.
   - `canContinueStage4` takes a new `thresholdsFresh` input (S9 fresh) and requires it.
2. **`src/lib/project-state.ts`.**
   - The S11 well-formedness check drops `bases`.
   - **Hidden dependency:** `matrixRowsMatchClip` currently derives the fragment keys from `threshold_edits.bases`. Preserve the edit-only stage derivation required by OC P3 and state-transitions §3.1: S22 is computed, so it cannot replace that dependency in the derived-stage predicate. S22 remains the canonical inventory for Continue and result-dependent read validation.
   - The persisted pair identities can supply the full ordered fragment keys when there are at least two fragments: a pair `(down, up, XYZ)` identifies `frag-down-XY` and `frag-up-YZ`. Require a consistent, contiguous chain starting at fragment 1, then compare its clipped extent with S14's rows. Empty pair values do not affect these identities.
   - For a single-fragment RNA, S11 has `pairs: []`. The edit-only check requires zero clip at both ends and exactly one S14 row with a valid `frag-1-SEQ` key; its sequence is checked against fresh S22 by the matrix read validation. Do not make stage restoration wait for S22, S9, S10 or S13. No new persisted inventory field is needed.
3. **`src/lib/seamless.svelte.ts`.** Remove `refe_validation` from the cell lists, the freshness/echo logic (the `refe_validation` echo case) and the computed-state types.
4. **`src/components/Stage4BestFits.svelte`.**
   - Remove the fragment-row `edits.bases` loop with its inputs, revert buttons and diff counting.
   - The table keeps the pair rows as they are.
   - Show fragment information (residues, sequence, conformer, Fit RMSD) read-only, in whatever minimal form keeps it visible. The layout is a follow-up.
   - Pass S9 freshness to `canContinueStage4`.
5. **`src/App.svelte`.** Pass S9 freshness wherever `canContinueStage4` is called. Skeleton creation on Continue 3→4 follows the new shape.
6. **`src/components/Stage6Gaussian.svelte`.** Remove the reference-anchor validation section and its derived state. The Gaussian validation and class table are unaffected.

## 4. Historical Alaric export-seam proposal — superseded

> This section records the W8b proposal for a synthetic `alaric/export_bundle.py` interface. It was never the production Preparic API. E2/E4 now validate and render the explicit cell-pin snapshot in `export_snapshot.py` and `export_tree.py`; E7 removed the synthetic module after transferring its useful assertions. For the current contract see [export-gens-plan.md](../export-gens-plan.md) and [oracle-contract.md](oracle-contract.md) §6.

1. **Candidate input.**
   - Each `reference_candidates` entry retains its candidate identity and carries `ovrmsd` (A4's rounded threshold) and `ovrmsd_raw` (A4's raw value). For `first`, take `nuc1_ovrmsd` and `nuc1_ovrmsd_raw`; for `second`, take the corresponding `nuc2_*` fields from the matching S9 bases row. Copy these values without re-rounding. Require both values; missing or malformed computed evidence is still refused under X6, not converted to a sentinel.
   - The emitted `ovrmsd` is that rounded value unless the sentinel rule below applies.
   - In the original W8b proposal, the snapshot producer was not yet written. E0–E7 now implement this contract: export admission requires fresh S9 from the same consistent snapshot (X6, X7); freshness at an earlier Continue 4→5 is not a permanent export receipt.
   - Require explicit S9 freshness evidence in `validations.results.thresholds`, alongside the existing fit, energetics and Gaussian results. Missing, stale, running or failed S9 evidence blocks the seam under X6. The separate `validations.decisions.thresholds` entry continues to describe S11's validity, not S9 freshness.
   - Preserve candidate deduplication and completeness checks. Duplicate candidate entries must agree on both original computed values before sentinel substitution; two different over-limit values must not become indistinguishable merely because both emit `-1`.
2. **Sentinel.**
   - Replace the X6 hard block for `anchor-refe` (the `ovrmsd exceeds L_OV` loop) with this rule: if the rounded value is greater than `L_OV`, emit the action with numeric `ovrmsd: -1`. Compare the rounded value with the shared `L_OV = sqrt(3)` constant, not with the raw value or the display approximation 1.73. Thus a rounded 1.73 is emitted unchanged and 1.74 gets the sentinel, including when its raw value is below L_OV.
   - Attach a comment in the bundle, associated with the deduplicated action by its `(fragment, nucleotide)` identity but outside the action mapping, because the action schema allows no extra keys. The precise sidecar representation is an implementation choice; it must retain the association through copying/deduplication. The future YAML writer emits it as `#` lines in that action's file. W8b delivers and tests the bundle comment; it does not implement the deferred writer. The comment:
     - states the computed raw and rounded value and L_ov;
     - explains that the action cannot work with the required rounded positive-control threshold, because that threshold exceeds the ovRMSD that Alaric can search without misses. If the raw value also exceeds L_ov, it may additionally say that the best-fit pose itself lies beyond the coverage limit; do not claim that when only the rounded threshold exceeds it;
     - says that the action is kept on purpose, as a disabled positive control.
   - Keep the `grow` pairs L_ov hard block.
3. **"Proceed anyway".** The seam takes the browser's `proceed_anyway_required` as before. The rule that bases never contribute belongs to the producer of that flag.
4. **Alaric actions and middle layer.** The runtime is unchanged: `anchor-refe` already rejects `-1` through `validate_ovrmsd` ("ovRMSD threshold must be finite and positive"). However, `alaric/middle/schema.py` currently rejects every nonpositive `anchor-refe` ovRMSD before deployment. Amend that check to admit exactly `-1` as the disabled-control sentinel, retaining rejection of zero and other negative values. The middle layer must load, resolve and deploy the sentinel action; its own execution then fails at the existing runtime threshold check. This implements A12/X3's requirement that one sentinel action must not prevent the project's other actions from deploying and running.

## 5. Shared contract artifacts

In `contracts/preparation-v1/build_artifacts.py`:
- remove `bases` from the S11 schema and fixtures;
- remove the S20 schema;
- keep S9's `bases`.

Regenerate `schemas.json` and `fixtures.json`. Remove T23/S20 from the generated transformer/share inventory and remove S20 fixtures. Keep a fixture of a legacy S11 with `bases`, which must be refused as malformed. Update editable-threshold fixtures and clip/completeness cases to use pairs; do not remove computed S9 bases from the shared `thresholds(computed)` helper's computed branch.

## 6. Unchanged

- A1–A4 and their 1b7f goldens: the bases are still computed for every fragment, which is harmless.
- The rounding rule.
- Clip rules (W8a).
- Energetics, matrix behavior and Gaussian validation. The matrix's edit-only extent check changes as specified in §3, while its behavior and persisted shape stay unchanged.
- `grow` thresholds from edited kept pairs.
- The atom set of `anchor-refe` and of the bases table: base heavy atoms, as today.

## 7. Evidence

The W8b test-name list below is historical. E7 transferred the export assertions
to real Preparic input-pin fixtures and rendered-tree tests; the retired bundle
tests and their synthetic fixture are no longer part of the active suite.

Update the existing tests that encode the editable bases, rather than adding parallel ones:
- preparic `tests/frontend/{w3-stage4,w1-project-state,w45-stage5,w7-class-state,w7-stage6-editor,w1-energy-threshold-read}.test.ts`;
- preparic `tests/backend/{test_w3_stage4,test_w1_project,test_w7_stage6_backend,test_w7_runtime_bindings,test_result_backups}.py`;
- preparic `tests/W7_ACCEPTANCE.md` and `tests/browser/w1/README.md`;
- Alaric's useful schema, deployment and runtime action checks in `tests/test_w8_actions_contract.py`, `tests/test_w8_energy_filter_integration.py` and the retained non-export tests in `tests/test_w8_contract_followup.py`.

The transferred export admission/digest assertions are in Preparic's `tests/backend/test_export_e7_acceptance.py`; selected catalogue graphs and Alaric resolution/deployment are in `tests/backend/test_export_e7_selected_projects.py`.

The tests must then establish:

- **Stage 4.**
  - The skeleton has pair rows only.
  - With every kept pair filled and the existing clip, S13 and T10 gate conditions satisfied, the gate opens only once S9 is fresh. With no kept pairs, completeness is vacuously satisfied but all the other gate conditions still apply.
  - A stored S11 with `bases` loads as undecided, with the reason shown.
  - The matrix row check still works after the change, including on a single-fragment RNA, a larger RNA clipped down to one fragment, and a clip edit racing Continue. Stage 5/6 restores from persisted decisions before computed results arrive after a restart; loading or refreshing S22 must not change the derived stage.
- **Backend.**
  - T10 and T11 accept and produce the pairs-only shape.
  - No `refe_validation` cell, mount, share or transformer remains.
  - Stage 6 renders without a reference panel.
- **Export seam.**
  - Every emitted `anchor-refe` `ovrmsd` equals A4's rounded value for its candidate when that value is at most L_ov, and equals `-1` otherwise. Both nucleotide mappings, duplicate agreement and rejection of missing/malformed computed values or missing/nonfresh S9 evidence are covered.
  - A candidate whose rounded value is above L_ov gets `ovrmsd: -1` plus the comment. By itself it is not a hard block and does not set "Proceed anyway"; independent X5/X6 conditions retain their effect. On 1b7f, a stacking on the 12th residue gives candidate (fragment 11, `second`, 2.25 Å); a synthetic snapshot is also acceptable. Cover 1.73, 1.74 and a raw value below L_ov that rounds above it; clipping the sentinel candidate removes both its action and its comment.
  - A kept pair ovRMSD above L_ov still hard-blocks.
- **Alaric.**
  - The middle layer accepts and deploys the `-1` sentinel, while still rejecting zero and other negative thresholds at schema time.
  - `anchor-refe` with `ovrmsd: -1` fails with the threshold error.
  - In a project containing one such action, every other action deploys and runs.
- **Unchanged behaviour.** The 1b7f A3/A4 goldens and the W8a clip evidence still pass.

**Exit:** the amended oracles (P4.2, P4.3, the 4→5 gate, P6.6, X3, X5, X6, D4, D11 and the A12 `anchor-refe` oracle) pass. No preparic code path consumes bases as an accepted editable decision or writes edited bases; legacy-key detection/refusal remains. The stage 4 threshold table holds pairs only, with the read-only fragment information still visible. Record the results for W9.
