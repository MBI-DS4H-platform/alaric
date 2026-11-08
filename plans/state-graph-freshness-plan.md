# W10 — Freshness from the state graph, replacing hand-designed echoes

Draft, 2026-11-07. **Relation:** a work package after [final/implementation-plan.md](final/implementation-plan.md) (W1–W9). It changes how preparic meets [oracle-contract.md](final/oracle-contract.md) P5 (freshness) and P6's pending-write rule, not what they require. It replaces the echo mechanism of [state-transitions.md](final/state-transitions.md) §6 and [contract-boundaries.md](final/contract-boundaries.md) §8.4. Those documents are final, so Phase 2 needs the author's decisions in §2.

**Baseline:** preparic `96f99da` plus its working tree, Alaric `99d6f56`. Seamless facts come from `seamless/docs/agent/contracts/shares.md` (*The state graph*, *Writing: PUT*, *The websocket*), `node-state-lifecycle.md` (*The cascade, and glitch-freedom*) and `seamless-workflow/.../share/static/seamless-client.js`. The graph facts in §3.3 come from a live dump of preparic's Context, built against a scratch mount directory on 2026-11-07.

## 1. Why

- **Today.** Each computed share carries an `echo` of the decisions it was computed from. The browser rebuilds the expected echo (`currentComputedEcho` in `seamless.svelte.ts`) and compares the two. The dependency knowledge therefore exists in three places: the transformer wiring, the echo fields the backend builds, and the expected echoes in the browser. Nothing keeps them aligned.
- **A known gap.** The fit echo (`stage4._fit_echo`, used by S9, S10, and through them S13, S15, S17, S19 and S25) is `{pdb_code, range-1 row, resids}`. It has no `biounit`, but under D7 the biounit changes the RNA coordinates. The live graph confirms that `biounit` is upstream of `best_fits`. Failing scenario:
  - same PDB code, biounit changed, and the RNA chain keeps its name but gets a different assembly operator;
  - ranges and selection are re-entered with identical rows;
  - stage 4 is reached before T5 has refitted.

  The old fit, energetics and Gaussian results then show as fresh, which breaks P5.
- **Why the state graph was rejected, and why that doesn't hold.** ST §1.4 rejected it because node states lag. The lag can be detected:
  - each snapshot is built in one controller read turn, so it is consistent;
  - each snapshot holds every named node's state and checksum, edit cells included, plus `connections`;
  - the browser knows the checksum of every edit-cell value it holds: from the `{"checksum", "marker"}` body of an accepted PUT, from websocket `update` messages, and from a GET's `ETag`, which `seamless-client.js` stores in `entry.checksum` before it calls `onchange`.

  If the snapshot's edit-cell checksums are compared with the tab's own, an old snapshot can be recognised as old. Lag then costs a delay of at most about one 0.5 s snapshot interval, not a wrong answer.
- **Glitch-freedom.** When a node leaves `complete`, its whole downstream cone re-derives to `waiting` or `blocked` synchronously, before any recompute starts (`node-state-lifecycle.md`). So a node that is `complete` in a consistent snapshot was computed from the current values of everything upstream of it.
- **Precedent.** [export-gens-plan.md](export-gens-plan.md) §7.1 already uses this check for the admission cell: it matches the snapshot checksum against the checksum of the value the tab read.

## 2. Decisions for the author

| # | Decision | Recommendation |
|---|---|---|
| F1 | Adopt the rule of §3 as the mechanism for P5. Amend ST §1.4, §2.2 and §6, CB §8.4, and OC Appendix A (rows P5 and P11) | Yes |
| F2 | Where the map from share key to node path comes from | The backend publishes it as a read-only literal share, S27 `share_index`, built from the same list as the `share()` calls. Alternative: a frontend constant, checked against the snapshot on load |
| F3 | Validator failures (T10, T19, T27) count only while they are fresh under the same rule | Yes. A passing validator is still not required, so there are no receipts (I4 stands) |
| F4 | P6's "inputs" means every upstream edit cell, not only direct pins | Yes. `TRANSFORMER_INPUTS` is removed |
| F5 | Drop the backend's cross-generation echo checks (T16, T22) and rely on glitch-freedom | Yes. Shape validation stays |
| F6 | Shares lose their `echo` fields without a schema bump | Yes. Backend and frontend are always deployed together, and a backend restart already requires a browser restart (O6). Same precedent as W8b |
| F7 | Amend export-gens-plan §4.3, §4.4, §7.1 and §9.3 (§8 of this plan) | Yes |

## 3. The rule

### 3.1 Terms

- **Snapshot:** the last state graph the tab received since its last (re)connect.
- **Node of a share:** the cell the share serves, looked up through the share index (F2). The key and the path differ for two shares: `cif.cif` is node `cif`, and `structure_id` is node `structure_id_result`. Node `structure_id` is the transformer.
- **Edit cell:** a shared cell with no incoming connection. Today these are S1–S4, S8, S11, S12, S14, S18, S24 and S26.
- **U(X):** the edit cells upstream of node X, found through `connections` in the same snapshot.
- **held(K):** the checksum of exactly the value the tab currently shows for share K. It is recorded when a server value is applied (`entry.checksum` at `onchange`) and when the tab's own write is accepted (the PUT's `checksum`). It is cleared on reconnect and resync (P9). If it is unknown, the rule fails.
- **announced(K):** `entry.checksum`, the latest checksum the server announced for K.
- **pending(K):** this tab has a debounced, queued, in-flight or retried write on K.

### 3.2 Fresh

A computed share X is fresh for a tab iff all of the following hold:

1. the tab is live and has received a snapshot since it last connected;
2. X's node is `complete` in the snapshot;
3. held(X) equals the snapshot's checksum for X's node. The bytes the tab shows are the result the snapshot describes;
4. for every edit cell E in U(X):
   - not pending(E);
   - held(E) = announced(E) = the snapshot's checksum for E.

   If an announcement arrives before its GET has completed, held ≠ announced, so the tab counts as still catching up. `cif.cif` is not read automatically, and the tab need not hold its bytes, so for it the condition is announced = snapshot checksum, with no pending write.

Consequences:

- **Sources that aren't edit cells** (`aareduce_files`, `fraglib_identities`, `gaussian_model_json`, `ring_refe_*`) are not compared. A change to one makes the nodes below it leave `complete` or change checksum, and conditions 2 and 3 catch that. ST §6 says echoes can't see such changes; this rule can, including the polled B15 and B16 mounts.
- **Reverts** give a cache hit. If the graph returns to the same digest within one interval, no new snapshot is announced, but the last snapshot still describes the state.
- **Topology changes at run time.** The per-range T13 instances come and go, so ancestry is recomputed from each snapshot. That costs little: 77 nodes and 136 connections today.

### 3.3 Reading the graph (verified on the live dump)

- **Connection sources** take three forms:
  - a node path;
  - a node path with a sub-path, such as `["range_pdb_files_aa", "range-1-aa.pdb"]`;
  - `{"node": path}`, for transformer results.
- **Connection targets** are a cell path, `[transformer, pin]`, or `[cell, key]` for a join.
- **Resolution.** Resolve every source and target to the longest prefix that is a node path. Resolve `{"symbol": s}` through `anonymous_nodes` recursively. That table is empty today; handle it anyway.
- **Null values.** Null and empty values have hex checksums in the snapshot. For example, every null marker is `38e0b9de…`.

Upstream edit cells of each computed share. This table was computed from the dump, with no protein ranges, so no T13 instances were installed:

| Share | Upstream edit cells |
|---|---|
| S21 `structure_id` | S3 |
| S5 `range_specs` | S1, S2, S4 |
| S7, S22, S23, `reference_fragment_pdbs` | S1, S2, S3, S4 |
| S9, S10, S13, S15, S17 | S1, S2, S3, S4, S8 |
| S25 | S1, S2, S3, S4, S8, S12 |
| S19 | S1, S2, S3, S4, S8, S12, S18 |
| B14 `interactions` | S1, S2, S3, S4, S8, S12, S14 |

**Granularity.** The rule works on whole edit cells, so some results go stale more often than with echoes. None of this affects a gate in today's flow:
- **S22, S9, S10 and later results depend on all of S4,** protein rows included. But S4 is editable only in stage 2, where these results gate nothing. Gate 3→4 also needs S23, which has the same ancestry.
- **Fit results depend on S8,** through `_fit_input_gate`. S8 changes only in stage 3 and on Back 3→2, where fit results gate nothing.
- **S11, S24 and S26 are upstream of no shared result.** S12 affects S25, S19 and B14; S18 affects S19; S14 affects B14. This is the same as with today's echoes.

**Design rule for later changes:** a decision that stays editable in a stage should not lie upstream of a result that gates that stage. If it must, put a projection cell between them.

### 3.4 Dependencies set from Python, outside the graph

- **`_reconcile_stage5_range_scorers`** installs and removes T13 instances from `range_specs.value`. The graph doesn't show this. The rule stays safe because `merge_energetics` has two properties, which W10 keeps and tests (§7):
  - it raises when a range in `range_specs` has no result;
  - it ignores results for ranges that no longer exist.
- **Invariant for any future code:** code that sets cells or rewires the graph at run time must preserve "complete implies consistent with everything upstream". Today the other `.set()` calls run only at build time: `pdbcode`, `biounit`, `cif` and `fraglib_identities`.

### 3.5 Validators and errors

- **Validators (F3).** A failed T10, T19 or T27 closes its gate and shows only if the validator's upstream edit cells pass condition 4. A failure reported for values the tab no longer holds is ignored. This replaces hiding failures only while writes are pending, and stops an old failure from briefly closing the gate again after the write is accepted.
- **Errors (F4).** `visibleTransformerErrors` hides a failure while any edit cell upstream of the failing node is pending. `TRANSFORMER_STAGE`, which says which stage shows an error, stays as it is.

### 3.6 Bases that the S22 echo supplied

The S22 echo `{pdbcode (upper-cased), rna}` is also used as data:
- as the basis that Continue 3→4 writes into S11, and that the clip check and S24 use;
- by the backend validators.

Replace it with a **project address** built from the decisions:
- **Frontend:** `projectAddress()` = `{pdbcode: S1 upper-cased, rna: range 1 normalised the way T0 normalises it}`. Reuse the parser behind `currentRangeSpecs()`.
- **Backend:** T10 (`stage4._validate_threshold_edits`), the clip check in `stage4.py`, T27 and `reconcile_energy_threshold_edits` take `pdbcode` and `first_rna_range` pins and build the same address.

Under the 3→4 gate, S22 is fresh, so the address equals today's echo. A test asserts that the frontend and backend addresses are equal for the same S1 and S4, including chain ids such as `NO` and integers stored as strings.

## 4. Backend changes (preparic)

1. **Share index (F2).** Collect the `share()` calls in `build_context` into one table of key, node, readonly and mount. Publish `{key: path}` as S27 `share_index` (plain, read-only, set once at build time).
2. **Remove the echoes:**
   - the `fit_echo` transformer and its pins;
   - the `echo` fields of S7 (the stacking wrapper in `backend.py`), S22 and `reference_fragment_pdbs` (`stage4.py`), S9 (`alaric_runtime.thresholds_from_tables`), S10 (`stage4._parse_best_fits`), S13 (`stage5.merge_energetics`), S15, S17, S25 and S19 (`stage6.py`), and T20's job list;
   - S23's `range_specs` and `pdb_code` fields, if nothing besides freshness reads them. The per-file `range` field stays (P11).
3. **Cross-checks (F5).** Remove the echo comparisons in `stage5.energy_thresholds` and in T22 (`stage6.py`, "T20 jobs are stale…"), and the `echo` presence checks in `stage6.py`. Keep the checks on shape and `filter_set_version`.
4. **Validators** take the address pins of §3.6.
5. **`merge_energetics` is unchanged** (§3.4).

## 5. Frontend changes (preparic)

1. **New pure module `src/lib/freshness.ts`,** with no Seamless imports: `resolveGraph`, `upstreamEditCells`, `isFresh` and `failureIsCurrent`. Fully unit-tested.
2. **`seamless.svelte.ts`:**
   - extend `StateGraphNode` and `StateGraph` with `type`, `checksum`, `connections` and `anonymous_nodes`;
   - keep the last snapshot and its resolved ancestry;
   - record held checksums (§3.1), and expose `checksum` on `SharedCell`;
   - recompute freshness on every snapshot, share update and change to pending writes.
3. **`refreshComputedFreshness`** keeps its output, the `computedFreshness` map, so gates, banners, Copy and the viewer don't change. Remove `currentComputedEcho`, `valueEcho`, `echoIsFresh` and the pending special cases for `gaussian_scores` and `gaussian_validation`, which the rule now covers.
4. **Replace the echo-as-basis uses** with `projectAddress()`: `stage4.ts` (skeleton and S11 checks), `App.svelte` (Continue 3→4 and the clip basis), `Stage4BestFits.svelte`, and `stage5.ts` (S24 basis).
5. **Errors:** `visibleTransformerErrors` takes ancestry instead of `TRANSFORMER_INPUTS` (§3.5).

## 6. Phases

Each phase is a commit that can be bisected.

- **Phase 0, verify and capture (no change in behaviour).** Add a backend integration test that builds the Context in a temporary mount directory, as `test_backend_workflow.py` does. It reads `get_graph()` and the served state graph, and asserts:
  - edit cells are sources;
  - every share key maps to a node;
  - the ancestry equals the golden table of §3.3, so any change to it shows up in review;
  - after a PUT, the 200 checksum equals the snapshot checksum;
  - a null share's `ETag` equals its snapshot checksum.

  Save a snapshot fixture for the frontend tests.
- **Phase 1, shadow.**
  - Ship `freshness.ts` and checksum tracking, and compute both verdicts. Log disagreements to the console and to a development-only counter.
  - Run the W1, W3 and W7 browser scenarios, plus the biounit scenario of §1.
  - Expected:
    - "generic stale, echo fresh" only in windows under about 1 s;
    - "generic fresh, echo stale" never for longer than about 1 s; anything longer is a bug in one of the two;
    - the biounit scenario shows a long "generic stale, echo fresh" window, which is the echo gap.
- **Phase 2, switch** (needs F1, F3 and F4). Gates, banners, Copy, the viewer, validator relevance and error hiding move to the rule. Echoes are still produced but no longer read.
- **Phase 3, remove** (needs F5 and F6). Apply §4.2–4.4 and §5.3–5.4, and update the tests (§7).
- **Phase 4, documents.** Apply §8.

## 7. Tests

- **Frontend (Vitest, `freshness.ts`, on the Phase 0 fixture):**
  - fresh;
  - an old snapshot, where an edit-cell checksum differs;
  - a pending, debounced or queued write on an upstream cell;
  - announced but not yet fetched;
  - held(X) differs from the snapshot;
  - X `waiting`, `blocked`, `failed` or `unwired`;
  - a revert inside one interval;
  - both source forms and sub-path resolution;
  - a symbol in `anonymous_nodes`;
  - a T13 instance added or removed between snapshots;
  - the `cif.cif` and `structure_id` key mappings;
  - **the biounit regression:** only S2 changed, so `best_fits` is stale;
  - validator failure relevance (F3);
  - transitive error hiding (F4);
  - `projectAddress()` equals the backend address.
- **Backend:**
  - the Phase 0 graph assertions;
  - adding a protein range makes `merge_energetics` fail until the new instance's result is merged, and removing one leaves no extra range in S13;
  - the validators with address pins;
  - the share index covers every share key the server lists.
- **Browser acceptance:** rerun the W1 (two tabs, restart, reconnect), W3 and W7 scenarios. Add the biounit scenario, using an entry whose RNA chain has a different operator in another assembly, or a synthetic CIF.
- **Existing tests that reference echoes and need updating:**
  - backend: `test_backend_workflow.py`, `test_result_backups.py`, `test_w3_stage4.py`, `test_w45_stage5_acceptance.py`, `test_w7_runtime_bindings.py`, `test_w7_stage6_backend.py`;
  - frontend: `seamless-debounced-clip`, `seamless-stacking-sync`, `stage4-active-nucleotide`, `w1-project-state`, `w1-reconnect-regression`, `w3-stage4`, `w45-stage5`, `w7-class-state`, `w7-stage6-editor`;
  - the acceptance READMEs and matrices under `tests/`.

## 8. Document changes

- **state-transitions.md:**
  - §1.4: node states alone are not a freshness test, but together with the edit-cell checksums in the same snapshot they are;
  - §2.2: the definition of fresh;
  - §6: replace the echo table with §3.2, the granularity note of §3.3 and the design rule.
- **contract-boundaries.md:**
  - §8.4: replace the echo bullets with the rule;
  - §8.2: add S27 `share_index`, and drop the echo mentions;
  - §8.6.5: S23 no longer echoes.
- **oracle-contract.md:**
  - Appendix A, rows P5 ("echoes in computed cells") and P11 ("echoing S5 and B5");
  - P6, if F4 is adopted, to say that "inputs" means upstream edit cells.
- **export-gens-plan.md:**
  - §4.3: the "freshness echoes are still checked" bullet becomes "glitch-freedom guarantees that admission's inputs are mutually consistent; decision bases are still checked";
  - §4.4: "freshness validation" is reduced to decision-basis checks;
  - §7.1: "existing stage-6 result-echo checks pass" becomes "stage-6 results are fresh by W10's rule";
  - §9.3: rewrite the test that holds back an old computed share.
- **preparic:** `STAGES.md` (the freshness paragraph) and README (execution model).
- **`plans/INDEX.txt`:** this plan.

## 9. Risks and limits

- **Seamless guarantees.** The rule relies on three: consistent snapshots, edit-cell checksums in the snapshot, and glitch-freedom. Phase 0 checks what can be observed, and Phase 1 compares the rule with the current behaviour before anything switches.
- **Confirmation delay.** After an edit, a result is confirmed fresh up to one snapshot interval (0.5 s) later, plus share pacing (about 2/3 s) when the result itself changed. Continue may enable that much later than today.
- **Coarser granularity:** none at today's gates (§3.3). The design rule keeps it that way.
- **Python-side wiring** must keep the invariant of §3.4.
- **Unchanged:** stage derivation (markers and bases), restart behaviour (O6), and the exposure of the state graph.

## 10. Out of scope

- **Provenance on delivered results.** A Seamless feature in which a share record carries the checksums of its upstream edit cells. It would remove the client-side traversal, but W10 doesn't need it.
- **Atomic multi-cell writes.** Planned in Seamless, and independent of this plan.
- **An interim fix that adds `biounit` to the fit echo.** Only worth doing if W10 is delayed. The regression test of §7 covers the case either way.
