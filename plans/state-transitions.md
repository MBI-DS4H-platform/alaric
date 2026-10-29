# Preparic stage state: browser, cells and transformers

Plan only. This document describes preparic's state over stages 1–6 in three layers:
- the browser: which stage is shown and what the user sees;
- the shared cells: which are defined and which are empty;
- the transformers: which run, which wait and which are blocked.

It then defines every stage transition, and what happens when an input changes from elsewhere while a transition is under way. It builds on `contract-boundaries.md` (§8 in particular), uses that document's S-, B- and T-numbers, and proposes changes to it (§9). The export stage is only sketched.

Baseline:
- preparic HEAD `5c85f67`;
- Seamless behaviour from `seamless/docs/agent/contracts/{shares,node-state-lifecycle,pins,cells,attachments,cancellation,execution-backends}.md` and `seamless-client.js`.

Author decisions taken in this session:
- `cif` is mounted;
- Back and Continue are decided per stage;
- the fit waits for stage 3;
- Continue closes when there is nothing to anchor on;
- stage 4's main purpose is the cRMSD/ovRMSD threshold table, which the user may copy from the reference values and/or fill in themselves;
- the clip is changed in stage 4, never beyond a stacking anchor. On 5→4→5 the interaction matrix is reused: the clip it was built for is read from its grid extent, and the first and last runs are filled in or reduced.

Short version:
- **One linear stage, shared by every tab and derived from edit cells.** Stage k is reached when the marker cells of stages 2..k are all active and their recorded bases match the current upstream decisions. There is no stage cell.
- **Every transition is one marker write.** Continue activates the next stage's marker, with its default or its kept value. Back deactivates the current stage's marker.
- **Computed cells never move the stage.** Their freshness, read from an echo of the decisions they were computed from, opens or closes Continue and drives the "computing…" and error display, nothing else.
- **Every decision records what it was decided against.** A write from a stale tab, an edited mount file or a race then lowers the stage, instead of leaving a stage that silently contradicts its data.

## 1. Seamless behaviour that shapes this design

These are the points where an intuitive reading goes wrong.

1. **An edit cell is never absent.** A writable share installs null on an empty cell, so the browser always reads 200 or 204. Null is the only "undecided" a browser can see or write. An empty PUT body writes null, so `set("")` on a `text` cell writes null (shares.md *The initial decision*, *Writing: PUT*).
2. **A sentinel does not stop a transformer.**
   - Null or `[]` on a `plain`, `mixed` or `bytes` pin is an ordinary value, and the transformer runs.
   - Only null on a strict pin (`text`, `int`, `str`, …) blocks it: `blocked-by-error`, with no exception.
   - A cell with no producer and no value, which only the server can make, leaves its consumers `blocked-by-unwired`.
   - There is no gate or hold primitive (pins.md *Null*; node-state-lifecycle.md).
3. **A computed share does not show staleness.** While its node is `waiting`, `blocked` or `failed`, the share keeps serving the last value under the same marker. A share that has had no value since the process started answers 404 (shares.md *Reading*).
4. **The state graph lags.** It is rebuilt at most every 0.5 s, can miss a recompute that returns to the same graph, and can still say `complete` about 0.5 s after a PUT. It explains state; it is not a freshness test (shares.md *The state graph*).
5. **No ordering across shares, and no multi-key write.** Two PUTs are two controller turns. The replacement run launches without debounce, so a transformer that reads both cells may run first on (new A, old B) (workflow-context.md; node-state-lifecycle.md *Speculation*).
6. **Superseded runs are not stopped at once.**
   - A spawned run is killed only after a hold of 30 s, or up to about 5 min.
   - An in-process run continues to completion and holds the process lock, delaying every other in-process transformer (cancellation.md; execution-backends.md *The process lock*).
7. **A PUT of the value already served is a no-op:** 200, same marker, no `update`. By inference it also triggers no recompute, so repeating a write costs nothing (shares.md *Writing: PUT*).
8. **A conditional PUT protects one cell only.**
   - A stale marker gets a 409.
   - A write to a cell nobody else has touched succeeds, even if the writer's view of other cells is out of date.
   - An edit that the app debounces is sent with the marker current at send time. It can therefore overwrite a remote edit without a 409 (shares.md; seamless-client.js).
9. **Computed updates are paced:** at least 2/3 s apart per share, with intermediate values coalesced. PUT acceptances are announced at once (shares.md *Cell to share*).
10. **No "snapshot complete" event.** After the `shares` message, every undotted key is read in parallel, in arbitrary order. The dotted key `cif.cif` is never read (shares.md *seamless-client.js*).
11. **Across a backend restart:**
    - an open tab keeps the values it has, adopts the new markers and re-reads nothing;
    - queued writes that carry old markers normally get 409 and are dropped;
    - without a persistent result cache, everything is recomputed (shares.md *Lifecycle*).

## 2. Three layers of state

### 2.1 Browser state (per tab; never shared, never persisted)
- **Connection:** `connecting` (no complete snapshot yet), `live` or `disconnected`. Outside `live` the UI is read-only.
- **Snapshot** of the shared values, plus this tab's pending writes. A pending value is shown optimistically until the server answers 200 or 409.
- **The stage, derived from the snapshot (§3).** It is stored nowhere: not in the URL, not in localStorage and not in a cell.
- **View-local state:** Mol* camera, focused row, open dialogs. It never causes a write except through a user action.

Every tab shows the same stage. There is no per-tab navigation, so Back and Continue are shared actions. Earlier decisions are shown read-only in the progress sidebar, so looking back never needs Back.

### 2.2 Cell state
**Edit cells** (writable shares with no incoming edge): S1 `pdbcode`, S2 `biounit`, S3 `cif`, S4 `ranges`, S8 `stacking_selection`, S11 `threshold_edits`, S12 `best_fit_clip`, S14 `interaction_matrix` and S18 `gaussian_classes`.

**Markers.** S4, S8, S11, S14 and S18 are the stage markers. Each holds:
- null, if its stage was never entered;
- or `{basis, active, …decision}`.

`active: false` means the stage was left with Back. The decision is kept, so it can be restored (§4.3).

**Computed cells** (read-only shares). For a tab, each is in one of three states:
- **absent:** 404, not computed since the backend started;
- **fresh:** its echo (§6) equals the current values of the edit cells it derives from, as this tab sees them, pending writes included;
- **stale:** the echo differs. The state graph then says why: `computing` or `waiting` is shown as "computing…"; `failed` or `blocked-by-error` is shown as an error.

A *sentinel* is the empty result a transformer returns when a decision is null or inactive. Consumers treat an inactive marker exactly like null.

### 2.3 Transformer state
Seamless has seven node states: `unwired`, `miswired`, `blocked`, `waiting`, `computing`, `complete` and `failed`. This document uses five working labels:

| Label | Seamless state | Meaning |
|---|---|---|
| real | `computing` → `complete`/`failed` | runs on project data |
| sentinel | `complete` | ran on a null or inactive decision and produced the empty result in milliseconds |
| gated | `blocked` | not run on purpose, because its stage is not reached (§5.3); also covers transformers downstream of a gated one |
| waiting | `waiting` | an upstream is still computing |
| blocked | `blocked-by-error` | an upstream failed |

Transformer state never feeds the derived stage, and no Continue gate reads it. It only chooses between "computing…" and an error, and it drives banners.

## 3. The stage is derived from markers

### 3.1 Markers
| Stage | Marker | Basis recorded in the value | Default when first entered |
|---|---|---|---|
| 1 | none; always reached | | |
| 2 | S4 `ranges` | `pdbcode`, `biounit` | rows from the viewer's chains (browser-local) |
| 3 | S8 `stacking_selection` | range rows (chain, start, end, normalised as by T0) | `{key: bool}` by the default rule (distance ≤ 5, tilt ≤ 30) |
| 4 | S11 `threshold_edits` | `pdbcode`, range-1 row | a skeleton: one row per pair and per base from S22 `fragments`, all values empty |
| 5 | S14 `interaction_matrix` | the selected stacking keys; the clip, as recorded by the grid's extent (the cells it provides) | Recompute |
| 6 | S18 `gaussian_classes` | the selected stacking keys | the computed default classes |

**valid(M_k)** ⇔ M_k is non-null, `active`, passes the frontend's validation, and its basis equals the current upstream values.

**Stage** = the largest k such that M2..Mk are all valid. If M2 is not valid, the stage is 1.

Properties:
- The stage depends on edit cells only, so it is known as soon as the snapshot is complete. It does not flicker while computations run, after a restart, or while a computed share lags.
- It is the same in every tab, up to the propagation delay of one `update`.
- A marker beyond the valid prefix is ignored. That covers inactive markers, stale ones (basis mismatch) and orphaned ones (active beyond the prefix). Observers do not repair them.

### 3.2 Why not a stage cell
1. **It would be a second source of truth.** A stage cell can disagree with the decisions (stage 5 with a null matrix), and every reader has to handle that anyway.
2. **Every transition would write two cells,** the decision and the stage.
   - Other tabs would see the intermediate state.
   - A Back racing a Continue could leave the stage and the decisions inconsistent with no conflict detected, because the PUTs hit different cells (§1.8).
3. **It would not remove the need for bases.** An out-of-band or stale-tab write to a decision still has to be detected (§3.3).

### 3.3 Why bases
A conditional PUT does not stop a write that is legal for its own cell but wrong for the stage.

Example: a tab that slept during stage 2 wakes up, and its queued `ranges` PUT lands while everyone else is on stage 5. Nobody else wrote S4 since, so the marker matches (§1.8).
- Without bases, the project stays at stage 5. The selection, matrix and classes then silently refer to a different stacking table.
- With bases, M3 no longer matches S4. Every tab drops to stage 2 without anyone writing.

### 3.4 Decisions that are not markers, and where decisions can be edited
**The rule.** A decision may stay editable in later stages only if no later stage's decision has to be rewritten when it changes. Otherwise one user action becomes a multi-cell write (§1.5).
- **S11 `threshold_edits`** (the stage-4 marker) passes this rule. Nothing decided later depends on it; only computed checks do (T23, and the export). By OQ:8 it stays visible and editable in stages 4–6, and the stage-6 refe panel edits its `bases`.
- **S12 `best_fit_clip`** is a stage-4 parameter, not a marker.
  - It always holds a value: `{clip1, clip2, basis}`, with loader default `{0,0}`.
  - Its basis is `pdbcode` plus the range-1 row, the same as S11's.
  - **It is editable in stage 4 only (author decision)**, so it passes the rule. The matrix that depends on it is inactive whenever the clip can change. The matrix is adapted once, on Continue 4→5 (§4.2), not rewritten on every clip edit.
  - **It never cuts a selected anchor (author decision).** The stage-4 clip control stops at the outermost anchor fragments. To clip further, the user first disables that anchor in stage 3.
  - Whether it fits (it does not cut an anchor, and clip1 + clip2 < fragments in S22) is a gate and banner condition, not a stage condition.
- **S1 `pdbcode`, S2 `biounit` and S3 `cif`** are the stage-1 decisions and form the basis of M2.

### 3.5 Invariant and orphans
- In normal flow, every marker beyond the current stage is null or inactive.
- **Orphans** are markers that are active beyond stage + 1. They appear only through races (§7.1) or exogenous writes. They do not affect derivation.
- **Continue into stage k** first deactivates any active M_{>k}, and resets a stale non-marker decision of stage k (S12 when entering 4). It waits for those writes to be accepted, and only then writes M_k.
  - Without that order, an orphan whose basis happens to match could become valid again and skip stage k's gate.
  - In normal flow nothing needs deactivating, so Continue is a single write.

## 4. Transitions

### 4.1 Common rules
- **One marker write per transition.** The clicking tab applies it optimistically, so its stage changes at once. Other tabs follow when the `update` arrives. On a 409 the tab adopts the server value and tells the user.
- **Continue is disabled unless its gate holds** in this tab's snapshot. There is no "armed" Continue that fires later.
- **A write belongs to the stage it was issued in.** A deferred write is dropped if, when it is about to be sent:
  - the derived stage has changed;
  - the tab is not `live`; or
  - the target cell has received a remote `update` since the edit began.

  Deferred writes include debounced edits, a structure-download completion and the idle CIF flush. The last condition is the lost-update rule of §1.8: the server wins, as it would on a 409.
- **Back never destroys a decision.** It only deactivates the marker (§4.3), and so needs no confirmation.

### 4.2 Continue
**On entering stage k, Continue writes M_k:**
- **kept:** if M_k exists and is inactive with a matching basis, it is reactivated and its value is kept;
- **merged:** if M_k is stale, it is replaced by the default. For M3 the old ticks are merged in by key;
- **default:** if M_k is null, the default is written.

Per-ligand `error` entries in S13 do not block Continue: Recompute stops at those rows (EN:53–56).

**1→2**
- **Gate:** S1 is non-empty; this tab shows the structure for S1; S21 = S1.
- **Input absent or computing:** "uploading structure…".
- **Input failed:** S21 ≠ S1 once settled: banner (§7.1, last row).
- **Writes:** S4 := `{basis:{pdbcode,biounit}, active, rows}`.
- **Cascade:** T0 → S5 → T1 → S6 → T2, T3, T4 → S7. Once B4 exists: T6 → T24 → S22. Group B stays gated (§5.3).

**2→3**
- **Gate:**
  - the rows pass frontend validation;
  - no S4 write is pending;
  - S7 is fresh and has at least one row (author decision).
- **Input absent or computing:** "detecting stackings…".
- **Input failed:** banner for T0 (parse), T1 (extraction) or T4 (detection).
- **Writes:** S8 := `{basis: rows, active, selected: {key: bool}}`.
- **Cascade:**
  - The gate opens, and group B starts: T5 → T7 → T8 → S9; T9 → S10; T12 → T13×k → T14 → S13; T18 → S17.
  - T15, T19, T20 and T23 rerun as sentinel or waiting.

**3→4**
- **Gate:**
  - S8 is valid and its keys equal those of a fresh S7;
  - at least one stacking is selected (author decision);
  - S22 is fresh. The fit is **not** needed.
- **Input absent or computing:** "preparing all-atom structure…".
- **Input failed:** banner for T3 (aareduce, for example a modified nucleotide), T6 or T24.
- **Writes:**
  - If S12's basis is stale: S12 := `{0,0, basis}`, awaited.
  - Then S11, kept or as a skeleton.
- **Clip cutting an anchor.** If a selection change in stage 3 put an anchor inside the kept clip, stage 4 opens with the clip flagged as cutting that anchor, and 4→5 stays closed until the clip is reduced. No automatic clamp is applied (§11.4b).
- **Cascade:** T10 becomes real, and T11 and T23 become real once S10 and S9 are fresh. T20 → T21 runs once S10 is fresh. T15 and T16 run as sentinels.

**4→5**
- **Gate:**
  - every unclipped pair and base value in S11 is filled, and T10 is not failed;
  - S12 fits S22 and cuts no selected anchor;
  - S13 is fresh.

  The values may come from Copy, from typing, or both. S9 is not needed.
- **Input absent or computing:** S13 not fresh: "fitting…" or "scoring…". In practice the user waits for the fit only here, and can fill in thresholds meanwhile.
- **Input failed:** a T5 failure, a T13 receptor failure or a T14 failure gives a banner.
- **Writes:** S14 := `{basis: selected keys, active, grid}`, in one of two ways.
  - **Kept and adapted.** If an inactive S14 with matching selected keys exists, it is reactivated and adapted to the current clip:
    - the clip it was built for is read from its grid extent;
    - only the first and last runs are affected: they are filled in where the new clip uncovered rows, and reduced where it covers more;
    - every other run is unchanged.
  - **Recompute** otherwise.
- **Cascade:** T15 → B14; T16 → S15.

**5→6**
- **Gate:**
  - S14 passes read validation (EN:81–84) against fresh S13, S10 and S12;
  - S16 is present and its golden self-test passes;
  - S17 is fresh.
- **Input absent or computing:** S17 not fresh: "computing best-fit stacking geometry…".
- **Input failed:** a failed self-test or a T18 failure gives a banner. A failed self-test makes stage 6 unusable (S6:50).
- **Writes:** S18 := `{basis: selected keys, active, rows}`, kept or by the computed default.
- **Cascade:** T19; T22 → S19.

**6→export:** deferred (§10.5).

### 4.3 Back
Back from stage k writes one cell: M_k with `active: false`, everything else unchanged. Earlier markers stay active, and later ones are already inactive.

| Back | Deactivates | Also kept as it is | Restored on Continue when |
|---|---|---|---|
| 2→1 | S4 | | `pdbcode` and `biounit` are unchanged |
| 3→2 | S8 (the gate closes: group B goes gated; a running fit is superseded, §1.6) | | the range rows are unchanged; otherwise the ticks are merged by key |
| 4→3 | S11 | S12 (always active) | `pdbcode` and range 1 are unchanged, which is the usual case: one goes back to stage 3 to change the selection, and the thresholds do not depend on it |
| 5→4 | S14 | | the selected keys are unchanged (they cannot change from stage 4). The kept matrix is adapted to the clip on the way back (§4.2). Reusing it this way is an author decision. |
| 6→5 | S18 | | the selected keys are unchanged |

**Consequences of a kept decision, checked one by one:**
1. **Transformers.** An inactive marker reads as null, so its consumers run as sentinels while it is inactive and come back from cache on return. Example: T10, T11 and T23 for S11. The clip stays active and stays consumed throughout, but every consumer of the clip also needs an active marker or a gated input.
2. **Wrong data.** The basis catches a kept value that was made for another structure or another range 1, even when the fragment keys coincide (same RNA numbering under a different PDB code). Such a value is replaced by the default, never restored.
3. **Computed drift while away.** A kept value can come back to changed computed values. Examples: a new fit after a restart with new code; energetics after a protein-range change; classes after a fit change. The plans already handle each case as "kept, differing rows marked" (S4:71, EN:89, S6:94), and the read checks catch shape changes. A restored value is shown with those marks, and the gates re-check it.
4. **Gate.** A restored S11 must still pass the 4→5 gate. If the clip has changed meanwhile, newly unclipped rows may be empty and must be filled.
5. **Revert.** If the user restores an earlier `pdbcode` or range rows before going forward, the kept decisions match again and are restored together with every cached result.
6. **Export** reads decisions only at the export stage, after every gate has held.

The alternative (A), clearing the marker to null, is simpler. It would throw the threshold table away on every Back 4→3, and the matrix or the classes on Back from 5 or 6 (§10.3).

### 4.4 Edits within a stage
| Cell | Editable in | Write policy | Recomputes |
|---|---|---|---|
| S3 + S1 | 1 | as a pair, in the same tick, only after a successful download (§8.6.1); arbiter rule (§7.1) | S21; everything else is a sentinel because M2 is null or inactive |
| S2 | 1 | immediately | nothing on the backend |
| S4 | 2 | on row commit (blur or Enter), only rows that pass frontend validation; never per keystroke | T0–T4, T6, T24 (group B gated) |
| S8 | 3 | per toggle | T15, T19, T20, T23 (sentinel or waiting) |
| S11 | 4–6 (OQ:8); `bases` also in the stage-6 refe panel | per committed table cell; Copy (enabled when S9 is fresh, with confirmation before overwriting typed values) | T10, T11, T23 |
| S12 | 4 only (author); the control stops at the outermost anchors | debounced, about 300 ms | T11, T20 → T21 (seconds), T23. T15 and T16 run as sentinels, because S14 is inactive in stage 4. |
| S14 | 5 | per toggle | T15, T16 |
| S18 | 6 | debounced, 500 ms (S6) | T19, T22 |

Notes:
- Debouncing S12 matters: superseded in-process T21 runs would otherwise queue behind the process lock (§1.6).
- **Clip and matrix.** No user action writes both cells.
  - A clip edit happens only in stage 4, while S14 is inactive.
  - The adaptation to the new clip is part of the single Continue 4→5 write (§4.2).
  - This replaces EN:33 (`ClipControls` shared by stages 4–6) and EN:88 (`applyClip` written on every clip edit, in any tab with a matrix loaded).

## 5. Per-stage state
Steady state means after all computation has finished.

### 5.1 Edit cells
"A" is an active, valid marker. "kept" is an inactive value from an earlier visit, or null.

| Cell | 1 | 2 | 3 | 4 | 5 | 6 |
|---|---|---|---|---|---|---|
| S1 `pdbcode`, S3 `cif` | empty or set | set | set | set | set | set |
| S2 `biounit` | int | int | int | int | int | int |
| S4 `ranges` (M2) | kept | A | A | A | A | A |
| S8 `stacking_selection` (M3) | kept | kept | A | A | A | A |
| S11 `threshold_edits` (M4) | kept | kept | kept | A, being filled | A | A |
| S12 `best_fit_clip` | value (maybe stale) | value | value | value, valid | value | value |
| S14 `interaction_matrix` (M5) | kept | kept | kept | kept | A | A |
| S18 `gaussian_classes` (M6) | kept | kept | kept | kept | kept | A |

### 5.2 Computed shared cells
"F" means fresh. A gated cell serves its last value (stale by echo) or 404.

| Cell | 1 | 2 | 3 | 4 | 5 | 6 |
|---|---|---|---|---|---|---|
| S21 `structure_id` (new) | F | F | F | F | F | F |
| S5 `range_specs` | sentinel | F | F | F | F | F |
| S7 `stackings` | sentinel | F | F | F | F | F |
| S22 `fragments` (new) | sentinel | F | F | F | F | F |
| S9 `best_fit_thresholds`, S10 `best_fits` | gated | gated | computing → F | F | F | F |
| S13 `energetics` | gated | gated | computing → F | F | F | F |
| S15 `energy_thresholds` | gated | gated | sentinel | sentinel | F | F |
| S16 `gaussian_model` | F | F | F | F | F | F |
| S17 `bestfit_stackings` | gated | gated | computing → F | F | F | F |
| S19 `gaussian_validation` | gated | gated | sentinel | sentinel¹ | sentinel¹ | F |
| S20 `refe_validation` | gated | gated | sentinel | F | F | F |

¹ T21 already runs from stage 4. Only T22's merge with S18 waits for stage 6.

S9, S10 and S13 are often still computing at stage 4. Stage 4 does not wait for them (§4.2).

### 5.3 Transformers
**Group B** is T5, T7, T8, T9, T12, T13×k, T14 and T18: everything downstream of the fit. By author decision it waits for stage 3: T5 depends on S8 being active.

| Transformer(s) | 1 | 2 | 3 | 4 | 5 | 6 |
|---|---|---|---|---|---|---|
| T0–T4, T6, T24 | sentinel | real | complete | complete | complete | complete |
| group B | gated | gated | real (minutes: library loading in T5 and T7) | complete, or still running | complete | complete |
| T10 | sentinel | sentinel | sentinel | real | real | real |
| T11, T23 | gated | gated | sentinel | real once S9 and S10 are fresh | real | real |
| T15, T16 | gated | gated | sentinel (T16 waits for S13 first) | sentinel | real | real |
| T17 | complete | complete | complete | complete | complete | complete |
| T19 | sentinel | sentinel | sentinel | sentinel | sentinel | real |
| T20 → T21 | gated | gated | sentinel (empty jobs) | real (seconds) | complete | complete |
| T22 | gated | gated | sentinel | sentinel | sentinel | real |

T13 instances are added and removed with S5 already at stage 2 (contract-boundaries T13). They stay gated until stage 3.

### 5.4 What the user sees
| Stage | Main content | Pending wording | Failures shown |
|---|---|---|---|
| 1 | PDB code, biounit, viewer | "uploading structure…" | download error (browser-local); S21 ≠ S1 |
| 2 | ranges table, viewer | none (S5 is not shown) | T0, T1 |
| 3 | stacking table and selection | "detecting stackings…", "preparing all-atom structure…" | + T3, T4, T6, T24 |
| 4 | threshold table (editable, reference values alongside, "differs from computed" marks), Copy, clip, best-fit overlays | "computing reference thresholds…" (Copy disabled), "fitting…" | + T5–T11 |
| 5 | energetics, matrix, energy thresholds, threshold table, clip (read-only) | "scoring…" | + T12–T16 |
| 6 | classes, validation, refe panel, threshold table, clip (read-only) | "validating…" | + T17–T23 |

**Banner scope.** A stage shows the failures of its own transformers and of earlier stages' transformers, never of later ones. A later-stage transformer may legitimately fail on kept or orphaned decisions, and that must stay hidden. A failure is also hidden while this tab has a pending write on one of the transformer's inputs (contract-boundaries §8.5 rule 2). The first state graph arrives within about 0.5 s of the `shares` message; no banner is shown before it.

## 6. Freshness: echoes
An echo must cover **every edit-cell value on the path**, restricted to the parts that matter. A key set derived from one upstream is not enough.

Example: S13's fragment keys stay the same when a protein range is edited. If the echo were the fragment keys only, the user could go back to stage 2, edit a protein range, return, and Recompute from energies that are still being rescored.

| Cell | Echo (values) | Fresh when it equals |
|---|---|---|
| S7 | structure id; range rows (T0-normalised) | S1; this tab's S4 rows |
| S22 | `exclude_pdb_code`; range-1 row | lower-cased S1; S4 range 1 |
| S9, S10 | `exclude_pdb_code`; range-1 row and resids | lower-cased S1; S4 range 1 |
| S13 | S10's echo plus every protein range row | S1; S4 |
| S15 | S13's echo, S14's grid, S12 | the same, plus S14 and S12 |
| S16 | `filter_set_version` | the version the page loaded with |
| S17 | S10's echo plus the protein range rows | S1; S4 |
| S19 | S18 rows, S12, selected keys, S10's echo | the same |
| S20 | S11 `bases`, S12, selected keys, S10's echo | the same |

- **Comparisons use this tab's local values, pending writes included.** A tab that has just written therefore sees its dependents go stale at once, before any `update` arrives.
- **A revert brings no `update`.** When the inputs return to values whose result was already delivered, no `update` follows (§1.7). The old value matches again and is fresh.
- **Echoes cannot see code or library changes.** Those change computed values for unchanged decisions. Within one backend process only edit cells change, so code and library changes come with a restart (§8).

## 7. Exogenous changes mid-flight
For every input, the question is whether it can change from elsewhere while something is under way, and what happens then.

| Source | What it can change | Outcome |
|---|---|---|
| Another live tab | any edit cell writable at the current stage; Back and Continue | Same-cell conflicts give a 409, and the loser adopts the server value. Different-cell races: §7.1. |
| Stale tab (lagging, asleep, reconnecting) | whatever it issued before going stale; the client retries it | Passes the marker check if no one else wrote that cell (§1.8). Bases lower the stage (§3.3, §7.2a). |
| Deferred writes in a live tab (download → S1/S3; debounced S4, S12, S18; idle CIF flush) | their own cells | Dropped by the §4.1 rule if the stage changed or the cell changed remotely meanwhile. |
| URL `?pdb=`, `?continue=1` | S1/S3; the stage | Applied only when the derived stage is 1, and `?pdb=` only when S1 is empty. Otherwise a link opened in a new tab would reset the shared project. |
| Mount files edited by hand (`authority="file"`) | any mounted edit cell, at any time | Same as a stale-tab write. Malformed content is invalid, so the stage falls back. The loader repairs content only at startup. |
| Read-only Alaric mounts (B15, B16) | S16 | The `filter_set_version` echo changes, and stage 6 shows that the model changed. |
| Backend restart | computed cells, which are absent until recomputed; markers reset | §8 |
| Code or library update (implies a restart) | computed values for unchanged decisions | §8; invisible to echoes |
| Seamless itself | transient runs on mixed inputs; held superseded runs; paced updates; a cache hit completing one turn late | No effect on the stage. Echoes mark the transient values stale. |

### 7.1 Races between actions
| Tab A | Tab B, at the same moment | Result |
|---|---|---|
| Continue k→k+1 | Continue k→k+1 | Same cell: one 409, and both tabs reach k+1. |
| Continue k→k+1 | Back k→k−1 | Different cells: M_k is inactive and M_{k+1} is an active orphan, so every tab is at k−1. The next Continue into k deactivates the orphan first (§3.5). |
| Continue 2→3 | edits S4 | M3's basis is the old rows, so M3 is invalid and the stage stays 2. A is told. The gate stays closed until S7 is fresh again. |
| Back k→k−1 | edits M_k | Same cell: one 409. Either the edit or the Back is lost, and its tab is told. |
| edits S11 at stage 5 | Back 5→4 | Different cells: stage 4, with A's threshold edit. |
| Continue 4→5 (adapts the kept S14 to clip c) | edits S12 to c′ at stage 4 | Different cells. S14 lands built for c, but S12 = c′. M5's basis includes the clip read from its grid extent, so M5 is invalid, and every tab stays at, or drops back to, stage 4. The next Continue 4→5 adapts the matrix to c′. Nothing needs repairing. |
| new PDB code (S3 + S1) | new PDB code (S3 + S1) | Four PUTs; S1 decides (**arbiter rule**). The tab whose S1 was refused gives up and loads the winner's structure. The tab whose S1 was accepted but whose S3 got 409 resends S3. Until then S21 ≠ S1, and Continue 1→2 stays closed. |

### 7.2 Worked cascades
**(a) A tab sleeps during stage 2 with a committed `ranges` PUT queued.** Meanwhile the others reach stage 4 and fill in the thresholds. The tab wakes, the client retries, and the PUT lands because nobody wrote S4 since.
1. S4 changes. T0 → S5 → T1 → S6 → T2/T3/T4 → S7 recompute. If range 1 changed:
   - T3 → B4 → T6 → T24 → S22;
   - T5 (the gate is open, since S8 is still active) → T7 → S9, and S10, recompute.
2. In every tab, M3's basis no longer matches S4, so the stage is 2. M4 (S11) stays active, but it is beyond the valid prefix, so it is an orphan. If range 1 changed, S11 and S12 are also stale.
3. Transformers run on the mix (new ranges with the old selection, thresholds and clip). T10, T11, T15, T20 and T23 may fail. They are out of scope at stage 2, so no banners appear.
4. Resolution, either way:
   - The users review the ranges and Continue 2→3. That deactivates the orphan S11 first, then writes M3 merged by key. Continue 3→4 then restores S11 if range 1 did not change, and writes a skeleton if it did.
   - Or someone restores the old ranges. Then everything becomes valid again, the cached results return, and stage 4 reappears.

**(b) Back 4→3 by A while B clicks Continue 4→5.**
1. S11 becomes inactive, and S14 is written with basis = the selection.
2. The stage is 3 everywhere, and S14 is an orphan.
3. A Continue 3→4 with an unchanged selection would reactivate S11 and make S14 valid again. The stage would then jump to 5 and skip the 4→5 gate (for example a threshold left empty). That is why that Continue first deactivates S14 and waits (§3.5).

**(c) A new PDB code in stage 1 while the download is still running, and another tab clicks Continue 1→2 with the old code.**
1. The download completes, but the derived stage is now 2, so the deferred S3 + S1 write is dropped (§4.1), and the user is told.
2. Without that rule, the pair would land, M2's basis would fail, and the project would drop back to stage 1 for everyone.

## 8. Load, reconnect and restart
**Load** (a new tab or a reload):
1. `connecting`: an overlay; nothing is editable.
2. Wait until S1, S2, S4, S8, S11, S12, S14 and S18 have been read. Edit cells answer 200 or 204, never 404, so each one arrives. Computed cells may still be unread or 404.
3. Derive the stage and render it. Computed cells that are unread or 404 show the pending wording.
4. Show banners only after the first state graph.
5. The viewer loads the structure for S1. There is no upload, because S3 is mounted (author decision).
6. Apply URL parameters as in §7.

**Websocket blip** (same server process):
- The tab is `disconnected`: a read-only overlay over the last state.
- Deferred writes are held, and §4.1 applies to them on reconnect.
- On reconnect the client re-reads every share, and loading continues from step 3.

**Backend restart:**
- Edit cells, S3 included, reload from `deployment-mounts/`. The derived stage is unchanged, and nothing needs a browser.
- Computed cells are absent until recomputed.
  - From stage 3 on, the gate is open, so group B restarts at once.
  - Without a persistent result cache that means minutes of library loading. Meanwhile stage 4 stays editable, but Continue 4→5 and later are closed.
  - With a persistent cache, results return one turn later as cache hits.
- **An open tab must re-sync.** It keeps its old computed values, whose echoes still match, so they would pass as fresh even after a code update (§1.11). A tab that detects a restart therefore reloads, or discards its computed values and re-reads everything. Its queued writes are refused (409), and their users are told.

## 9. Proposed changes

### 9.1 To `contract-boundaries.md`
1. **Markers (§8.1, §8.2).**
   - S4, S8, S11, S14 and S18 become `null | {basis, active, …}`. Their loaders write null.
   - `[]` stops being an empty value; a genuinely empty result is then never mistaken for "undecided".
   - S8 becomes keyed by stacking key instead of a `list[bool]` indexed by row.
   - Consumers treat an inactive marker as null.
2. **S12 `best_fit_clip`** keeps its loader default `{0,0}` and gains a basis (`pdbcode`, range-1 row). Continue 3→4 resets it when it is stale.
3. **S3 `cif` is mounted with `authority="file"` (author decision).** "`cif` is the only unmounted edit cell", the re-upload, and the "pending after a restart until the browser re-uploads" case all go.
4. **New computed shares.**
   - S21 `structure_id`: the entry id read from the CIF. It is the consistency check between S3 and S1.
   - S22 `fragments`: the fragment keys (and so the pair and base rows) of range 1, from T6's output via a new cheap preparic transformer, T24 `list_fragments`. It is not gated, and it is ready in stage 2.
5. **Stage 4 (§8.1 "On entering a stage…", OQ:8).**
   - Entering stage 4 creates the threshold table as a skeleton.
   - Values come from Copy and/or typing.
   - Before stage 5 the table must be **complete** (every unclipped value filled), not necessarily copied.
   - T10 validates S11 against S22 instead of S9, so it no longer waits for the fit.
   - S9 still feeds Copy, the "differs from computed" marks, T23, and the export's raw-value comparison.
6. **§8.5, resets.**
   - The stage-3 change no longer resets S14 and S18: they are inactive at stage 3 (§3.5).
   - The `best_fits` signature reset of S11 and S12 is replaced by their bases.
   - Back deactivates instead of resetting.
   - The clip-driven matrix rewrite goes: the clip is edited in stage 4 only, and the kept matrix is adapted on Continue 4→5 (author decision; replaces EN:33 and EN:88).
   - The clip control stops at the outermost selected anchors.
   - The only remaining multi-cell write is S3 + S1, under the arbiter rule.
7. **§8.4, echoes** are extended as in §6.
   - S13's echo must cover the protein ranges.
   - S9, S15, S17 and S22 gain echoes.
   - Freshness compares against local values, pending writes included.
8. **New: group B waits for S8 to be active** (author decision).
9. **§8.6.1** gains the arbiter rule (§7.1) and the "drop if the stage changed" rule for the deferred S3 + S1 write (§7.2c).

### 9.2 Behaviour at preparic HEAD that this replaces
- **Restore.** A one-shot restore effect waits for the structure and the stacking table, and after a restart usually lands on stage 2. → The stage is derived from edit cells.
- **Back.** `goBack` writes `ranges`, `range_specs` and `stacking_selection`, and `backToRanges` writes `stacking_selection`. → One deactivating write per Back.
- **`range_specs`.** The stage-2 effect writes it, even on the first entry, with values typed differently from the loader's. → Computed by T0.
- **Default selection.** `complete()` computes it from the in-memory table even when that table is empty or stale. → Gated on a fresh S7.
- **Forced stage 1.** A remote `pdbcode` or `biounit` change forces stage 1 tab by tab. → Derived through M2's basis.
- **`pdbcode` writes.** It is written per keystroke. → Written only with S3, after the download.
- **CIF upload.** The CIF is uploaded on every structure load, and never after a reconnect. → Mounted.
- **Stale stackings.** The `staleStackingRevision` marker barrier misses `always_visible` toggles. → Echo.
- **`?pdb=`** overrides a shared project. → Applied at stage 1 only.

## 10. Open points
1. **Gate for group B: decided (author), wait for stage 3.** The fit starts on entering stage 3, not on every committed range-1 change in stage 2.
   - It avoids speculative 8 GB library loads; superseded spawned fits live 30 s to about 5 min (§1.6).
   - The fit overlaps stage 3 and the threshold editing in stage 4.
2. **Zero stackings or zero selected anchors: decided (author), both close Continue.**
3. **Back semantics: (B), the `active` flag.** The author's matrix reuse across 5→4→5 requires (B) for M5. This document applies it to every marker; confirmation is pending (§11.3).
   - (A), null, would throw away the threshold table on every Back 4→3.
4. **Clip and matrix: decided (author).**
   - The clip is edited in stage 4 only, never beyond a selected anchor.
   - The kept matrix is adapted on Continue 4→5 from the clip recorded in its grid extent: the first and last runs are filled in or reduced.
5. **Export (deferred).**
   - The trigger is an edit cell, `export_request`, carrying the basis: every decision value plus the "proceed anyway" checkboxes.
   - The export runs only if that basis still matches when it executes, and its result echoes it.
   - An edit made while the export runs then leaves a stale export, never a mixed one.
6. **Seamless points to confirm.**
   - (a) An equal-checksum PUT does not revoke downstream (§1.7).
   - (b) Whether preparic's result cache survives a restart (§8).
   - (c) Whether the app can withdraw a write that is queued or retrying in `seamless-client.js`. If it can, §7.2a cannot happen.
   - (d) How the app learns of a restart, given that the client silently adopts lower markers.

## 11. Questions for the author
1. ~~Gate for group B~~: answered, wait for stage 3.
2. ~~Empty selections~~: answered, both 2→3 and 3→4 are closed.
3. **Back semantics (§10.3).** The matrix reuse implies deactivate-and-keep for M5. Should it apply uniformly to every marker, including S4, S8, S11 and S18?
4. ~~Clip and matrix~~: answered. The clip is edited in stage 4 only, never beyond an anchor, and the matrix is adapted on 5→4→5.
   - (a) **Filling in uncovered rows.** When the new clip uncovers rows, do the first and last runs simply extend to the new boundary? Or are the new rows decided by Recompute's energy rule (|E| ≥ 1, stopping at error rows), applied to those rows only?
   - (b) **An anchor selected inside a kept clip.** Suppose a user selects, in stage 3, a stacking whose fragment lies inside the kept clip. §4.2 then opens stage 4 with the clip flagged and 4→5 closed until the clip is reduced. Would you prefer Continue 3→4 to clamp the clip to the outermost anchor automatically (one more awaited write before S11)?
5. **Stage 4 gate.** Is "every unclipped threshold filled (copied and/or typed)" the right condition for 4→5, replacing OQ:8's "copied before stage 5"? Should rows of clipped fragments be required too?
6. **Stage 4 skeleton.** On entering stage 4, should the table start empty, with Copy as one click, or be prefilled from S9 when S9 happens to be fresh? Prefilling makes the default depend on timing: two tabs racing Continue could write different defaults.
7. **S22 `fragments`.** Is it acceptable to take the fragment keys from T6's output, which is not gated, so that stage 4 opens before the fit? Or must the key rule come from the fit (`best-fits.tsv`)?
8. **Thresholds editable in stages 5–6 (OQ:8 "throughout").** I read this as the whole table staying editable in stages 4–6. Correct?
9. **`biounit` in M2's basis.** Does the chain naming of the ranges depend on the biological assembly? If not, `biounit` can leave the basis and become pure display.
10. **Restart detection and cache (§10.6b, d).** Does `seamless-client.js` expose a hook for "the server restarted", or should the app infer it? Is the result cache persistent in preparic's configuration?
11. **Withdrawing queued writes (§10.6c).** Can a write handed to `seamless-client.js` be cancelled before it is retried?
12. **Equal-value PUTs (§10.6a).** Is it guaranteed that re-writing the current value never revokes the downstream cone?
13. **Read-only Alaric mounts.** Are B15 and B16 polled while the backend runs? If not, `filter_set_version` can only change across a restart, and the S16 row of §7 goes.
