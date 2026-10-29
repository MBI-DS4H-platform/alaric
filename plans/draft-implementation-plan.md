# Draft implementation plan: preparic stages 1–6 and Alaric deliverables

Draft, 2026-10-29. This plans implementation; it does not authorize publishing commits, change an existing contract, or design the export stage. Decisions needed from the author are enumerated at the end.

## 1. Basis and intended outcome

Read in this order: [oracle-contract.md](oracle-contract.md), [state-transitions.md](state-transitions.md), [contract-boundaries.md](contract-boundaries.md), [open-questions.md](open-questions.md), then [INDEX.txt](INDEX.txt). The oracle contract supplies observable requirements; the state-transition draft supplies the proposed state model; the boundary document supplies the T-, S- and B-numbered inventory. Explicit author decisions override proposals wherever they occur. Older mini-plans supply implementation detail only where those sources leave a gap.

The primary author read no material outside `plans/` and used no Seamless skills. Focused Luna xhigh source checks are identified below separately from document-derived requirements. Existing/planned status in the documents is a baseline to verify before coding, not a claim that every named file already exists.

The result is one durable, shared preparation project, with stages 1–6, sandbox-tested Alaric computations, editable thresholds/interactions/classes, and fresh validation results. The existing stages 1–3 must acquire the state and consistency guarantees needed by later stages. Export receives well-defined inputs and readiness information, but its page, trigger and project-directory construction remain deferred.

Out of scope: remote workers, deployment/authentication, multiple projects per backend, unbound protein structures, modified-nucleotide support, a general rewrite of stacking detection, and unrelated cleanup. The later Alaric action work is included because A12 fixes its behavior independently of export design.

### Requirements that supersede older implementation sketches

- Start the fit and its downstream work on entering stage 3; committed stage-2 range edits start no library work.
- Prepare/reference-cut the complete RNA independently of the fit. Open stage 4 once its fragments are known; fitting may continue while the user types thresholds.
- Compute all fits, tables and energies on the full RNA. Clip only derived selections, displays, jobs and downstream hand-offs.
- Stage 4 owns the clip. Stages 5–6 show it read-only. Adapt a saved matrix once on Continue 4→5; do not rewrite it on each clip edit.
- Use `null` for undecided stage decisions, keyed stacking selections, decision bases, and active/inactive markers. Computation state never moves the stage.
- Score both ligand sets in one Alaric run per protein range. No grids, JAX or summed-range energies enter preparic's scoring path.
- Derive every threshold in Alaric using the smallest multiple of 0.01 strictly above the delivered raw value. Keep raw values for validation; do not display them.
- Use one all-filter check whose jobs do not depend on editable Gaussian classes. Both anchor modes will eventually be exported.
- Keep each prefilter variant's Gaussian-only parent cutoff. Generate caches offline with an Alaric script, keep them gitignored, and fail visibly on a bad installation.

## 2. Delivery sequence and dependencies

Each work package should be reviewable and leave its relevant tests passing. Shared protocol work comes before adding wizard pages that depend on it. Alaric numerical work can be developed independently against fixtures, then connected through the declared boundary.

| Package | Owner | Main output | Depends on |
|---|---|---|---|
| W0 | both | canonical schemas, fixtures, focused feasibility probes | plans |
| W1 | preparic | durable decisions, shared stage, client synchronization and freshness | W0; D1/D7/D15/D17 where relevant |
| W2 | Alaric | A1–A4, A6, documented step interfaces and sandbox checks | W0 |
| W3 | preparic | preparation fixes and complete stage 4 | W1, W2; stage-4 decisions |
| W4 | Alaric | A5 gridless scoring step | W0, attract-jax prerequisite |
| W5 | preparic | stage 5, matrix editor, energy thresholds and hand-off | W3, W4; D2/D3/D10/D12 |
| W6 | Alaric | A7–A11 Gaussian spec, caches, shared anchor logic and check | W0, W2 |
| W7 | preparic | stage 6 and reference-anchor validation | W5, W6; D8/D9 |
| W8 | Alaric | A12 action behavior and numbering corrections | W2, W4, W6; D11/D16 |
| W9 | both | integrated acceptance and implementation documentation | W1–W8 |

Stages 4 and 5 should be delivered as one usable vertical slice: stage 4's forward gate needs fresh energetics even though its main purpose is threshold editing. Gaussian work need not delay that slice. The cascade optimization in W8 does not block preparic's gridless scoring.

## 3. W0 — settle representations and prove the protocol

### Contract fixtures and schema definitions

Create a small fixture set shared conceptually by Python and TypeScript tests: valid/undecided decisions, malformed decisions, unsupported schemas, fragment maps, threshold tables with blanks, interaction matrices, Gaussian jobs/results, and rounding/classification boundaries. Alaric owns formats it produces or consumes; preparic owns its internal decision envelopes and browser/backend formats.

Document names, types, schema, dependencies and output for each step before wiring it. Keep stable existing share/node names from the boundary inventory; add S21 `structure_id`, S22 `fragments` and T24 `list_fragments` from the state-transition draft. `range_specs` becomes computed and `range_pdbs` becomes unshared. Update the inventory when a genuinely necessary internal receipt/share is added.

Use `schema: 1` on boundary JSON, including the interaction hand-off: its abbreviated example in P5.8 omits `schema`, but G8 requires it. Reject unsupported schemas visibly without partially consuming their contents. Do not silently migrate a stored legacy decision whose basis cannot be established; load it as undecided and explain the reset. Preserve malformed-file diagnostics before startup normalization makes them disappear.

The stored matrix must have an **addressed extent**, not just a row count. A bare rectangular boolean array cannot distinguish clipping two fragments from the 5′ end from clipping two at the 3′ end. Proposed preparic envelope: schema, active flag, selected-anchor basis, explicit ordered fragment-row keys, ordered range identifiers, and the boolean grid. Its row addresses encode the old clip; no separately maintained old-clip cell is needed. Check extent against the full fragment map. Finalize this internal representation in W0, before matrix tests and UI code.

Use one canonical fragment map from A2/T24: fragment number/key, sequence, actual residue identities, existing adjacent pairs and candidate membership. Never derive a scientific fragment number from array position or `range.start` alone. D15 determines whether the UI permits holes; the map itself must agree with Alaric in either case.

### Small feasibility experiments

The focused Luna source checks found the following; these are inspected implementation facts, not newly chosen product behavior:

- The current browser client has no public restart callback or refresh-all API. It can adopt restart markers without rereading old computed values; dotted `cif.cif` is not auto-read. References reported by the agent: `seamless-client.js:115–117,152–243`; Seamless `shares.md:269–277`.
- App debounce timers can be canceled before `cell.set`. After that call there is no public cancellation API, and retries use the then-current marker. An equal-value PUT gives HTTP 200 without a websocket update; the current client can still deliver its acknowledgment through `onchange`. References: `seamless-client.js:306–414`; preparic `seamless.svelte.ts:493–505`.
- Read-only file mounts are polled by the backend mount broker. Thus an A7/A10 change can occur during a running process, contrary to ST §6's blanket restart assumption. Reference: Seamless `mounts.md:243–250,370–382`.
- Alaric's gitlink remains `12bba50`; the local attract-jax checkout is `22b23e4`, two commits ahead of its recorded origin, with three accuracy-data files untracked. A1 still requires construction-time RNA and static fitter arguments. References: agent's git inspection; `util/seamlessify_refe_best_fit.py:29,45–59,84–99`.
- The eight-file scorer closure appears viable with explicit staged paths. However, the current full attract-jax DEPS list omits `attract-par.npz`, so the contract's inclusion assertion will initially fail. Bring that full list into compliance rather than silently weakening A5's test. References: `attract-jax/util/minfor.py.DEPS.txt`; `jax_scorer.py:71,219–228,380,423`.

Turn those inspections into targeted verification tests:

1. Disconnect/reconnect and restart a backend while an open tab holds old computed values and pending writes. Establish a reliable resynchronization barrier, explicit CIF reading, and behavior of queued/retried writes.
2. Send an equal-value PUT and confirm both acknowledgment and downstream execution behavior. Do not wait for an `update` that may never arrive.
3. Edit read-only mounted code/model/ring files while running. Establish whether changes are polled or require restart; choose and document one supported update lifecycle.
4. Model adversarial Continue/Back interleavings, including an orphan written **after** an earlier cleanup, delayed writes from two stages ago, and cleanup interrupted by a conflict. Prove P3.4 before adopting the draft protocol unchanged.
5. Test folder/deepfolder connections and module attachment on the actual small A2/A4/A6 boundaries, without repeatedly loading the fragment library.

ST §3.5's ordered orphan deactivation is insufficient by itself: a delayed child Continue can retry after cleanup with a refreshed marker and reactivate that child. Entry tokens plus an app-owned write adapter address that case, but a second counterexample remains: while a stage-4 Continue is in flight, another tab can change a threshold without changing M4's activation token. The M5 PUT can then pass even though the gate was false when it was accepted. Client-side rechecking cannot close the final cross-cell race.

Strict P3.4 therefore needs an architecture correction (D17), not just an additional cleanup effect. Recommended property: transition acceptance is serialized with gate-affecting writes and verifies the expected stage, decision snapshot and fresh-result identities at commit time. Prefer a guard on existing share PUTs if an appropriate supported mechanism can be added; otherwise evaluate an authoritative atomic project-decision envelope with the necessary gate admission check. The source check found no existing multi-key primitive to assume. Preserve file persistence and decision-derived stage semantics in either design. Do not permanently bind stage 5 to the threshold values at entry: legitimate threshold edits in stages 5–6 must remain possible. Computed results must never become the source of the current stage, even though they participate in transition admission.

## 4. W1 — shared-project foundation in preparic

### Durable input and decision graph

- Mount all decisions file-authoritatively, including CIF. The backend startup loader initializes undecided markers to `null`, and clip to `(0,0)` with its basis. No running transformer writes a decision cell.
- Implement pure stage derivation and decision/read validators. M2/M3/M4/M5/M6 are ranges, keyed stacking selection, threshold table, addressed interaction matrix and Gaussian classes. Retained markers can be inactive; all consumers treat inactive or inapplicable decisions as undecided.
- Separate marker presence/well-formedness from forward-gate readiness. In particular, stage 4's valid skeleton contains blank thresholds; it must remain stage 4 while those blanks are filled. Backend validators accept undecided and distinguish permitted blanks from malformed numbers.
- Implement bases and restoration according to the oracle's decision-specific rules. Thresholds and clip belong to PDB code + range 1; matrix restoration additionally checks selected anchors and adapts its extent; classes belong to selected stacking keys. Carry the relevant project/range identity through nested lineage so identical row/key values from another project cannot reactivate unrelated decisions.
- Keep computational identity separate from restoration basis. A code/library refit preserves appropriate user edits while changing the computed results and diff marks.
- Add entry lineage separate from that semantic restoration basis: every explicit Continue activation gives its target marker a fresh activation token and records its current parent's activation token. The active prefix requires matching parent linkage. Ordinary edits preserve the token. A saved payload can be restored by Continue with new entry linkage; an old delayed child write cannot become valid merely because its parent was re-entered. Tokens are persisted decisions, not a boot-dependent stage check, so restart preserves the stage. This is a proposed implementation extension to ST, subject to W0's concurrency proof.

### Writes, navigation and synchronization

Centralize writes in the client integration layer, with the stage at issue time, original target revision, pending value and acknowledgment state. Use conditional PUTs on the existing share endpoints with the **original marker frozen per intent**, rather than handing an intent to a queue that refreshes its marker on retry. Drop unsent/retry intents on disconnect; cancel app-owned debounce work and reread before issuing new actions. Handle HTTP acknowledgment directly, including equal-value success without an update. Seamless's read/status client can remain in place. Inline edits are optimistic; losing/conflicting writes adopt the accepted server state and notify the user. Debounced edits are dropped if their stage, connection or remote target revision changes before application. Put no defaults/resets in observers of absent computed values. Test already-in-flight and ambiguous HTTP outcomes; cancellation alone is not a consistency guarantee.

Implement Continue/Back as explicit actions. Continue rechecks its gate immediately before sending, restores/merges/defaults the next marker, and performs required orphan cleanup and stale-clip reset in the proven order. Add D17's commit-time admission mechanism; the local recheck is necessary but insufficient. Back deactivates the current marker under D1. Under the per-marker design, normal transitions write one marker; exceptional cleanup is awaited and aborts on conflicting state. Never arm Continue to execute when computation later finishes. Resolve lost acknowledgments by reading the installed intent/activation token after resynchronization, not by resending with a new marker. Test that an in-flight write cannot cross a backend/context generation and accidentally pass a reused numeric marker; add a write-generation fence if the supported transport does not establish that property.

Stage 1 retains the code/CIF two-write exception: download successfully first, commit code and matching bytes, and validate the entry id read from the stored CIF via S21. Implement and test the arbiter rule, including retry after a CIF conflict only while that commit still owns the winning code and stage. Until consistency is established, close 1→2. A failed download changes neither decision.

On load/reconnect, hold a read-only overlay until every required decision has been read, including the stored structure through an explicit read when the viewer needs it. Clear trust in old computed shares on reconnect/restart and reread them before they can become fresh. A viewer showing an independent RCSB download is insufficient to prove it shows the persisted bytes; use the persisted structure for restoration and check its identity. URL parameters apply only under P9's stage-1/empty-code rules.

### Results, execution gating and errors

Use a common result envelope carrying `schema`, payload and enough input lineage to compare against the tab's current decisions, including optimistic edits. Echo actual relevant inputs through the computation; do not attach whatever decisions happen to be current when an old result arrives. Include structure identity and protein range identity wherever they affect results. Result freshness must also account for the current code/data lifecycle established in W0. In particular, carry a model/ring content identity through T17/T18/T21/T22 and compare it across related results, since read-only mounts can change live. Never combine a newly loaded browser model with an old validation merely because decisions still match. Ordinary pinned source/library updates should use the documented restart/resynchronization lifecycle.

State-graph status explains pending/failed/blocked results; it never certifies freshness or advances the stage. If a forward gate requires confirmation from a backend validator, provide a fresh validation receipt or equivalent acknowledged result. A lagging `complete` node cannot validate a just-edited value. This may require exposing a small result for T10/T19 despite the older boundary sketch leaving them unshared.

Implement the P6 stage-to-error mapping centrally, including T24 and installation errors. Suppress failures before the first status snapshot and while this tab has pending input writes. Show conversion failures as blocked. Later-stage work running on retained or orphaned decisions must not leak banners into earlier stages.

Gate the fit branch on the **valid stage-3 prefix**, not merely an old selection's `active` bit. A stale active M3 after a range change must not start a speculative library load while the derived stage is 2. Changing ticks within stage 3 must not change A1's numerical identity or repeatedly launch the fit. Use a separate cheap gate/identity projection with unchanged RNA/exclude/library inputs.

**Exit:** two tabs agree on stage/decisions after propagation; a backend restart preserves decisions and stage without browser repair; no result is trusted before resynchronization; adversarial transitions cannot skip a gate.

## 5. W2 — Alaric best-fit, reference and rounding deliverables

Keep the established A1 bash/factory path for its substantial staged dependencies. Prefer CLI + DEPS in the `alaric` environment for A3's library-dependent table computation; choose A2's module/CLI form after its small sandbox probe. Deliver A4 and A6 as standard-library modules. These are implementer choices, subject to the same sandbox and identity tests.

1. **A1/T5:** make RNA bytes and exclude code connectable project inputs; retain library identity and location separation. Preserve exact TSV columns, 1-based indices, discrete pose convention and `frag-N-SEQ-bestfit.pdb` names. Factories pin only fixed sources/data and never configure or execute Seamless.
2. **A2/T6:** cut references in template atom order, with distinct `-refe.pdb` filenames. Require exactly A1's fragment-key set, including missing-number cases. Templates remain declared inputs/dependencies; do not introduce the deferred `Reference(templates=...)` API merely for this work.
3. **A3/T7:** wrap the existing pairs/bases logic in one deliverable, taking fit TSV, RNA, exclude and library identity. Read exact library coordinates, not rounded PDB coordinates. Keep the locator's cRMSD configuration key even though matrices are not loaded.
4. **A4/T8 and A6:** implement one strictly-above rounding primitive in Alaric, used by the RMSD converter and energy-threshold function. Use decimal table text for RMSDs and shortest round-trip float64 decimal text for energies. A6 returns a two-decimal string. No extra margin or preparic rounding implementation.
5. Preserve unchanged raw table values in A4 output, with unambiguous field types and schema. Consumers must not recreate raw values by reversing rounded thresholds.

Every step gets its exact dependency declaration and a Seamless sandbox test on 1b7f. No dependency-list comments/blanks or stale `.CHECKSUM` sidecars. Block undeclared imports in the test; do not fix a failing sandbox by pinning the whole repository. Relocating a library locator must preserve identity; changing its committed identity must invalidate results.

**Exit:** golden A1 TSV and A3 tables reproduce; A2 keys/atoms agree and PDB-vs-TSV RMSD differs by at most `1e-4 Å`; clipped A3 subsets equal filtered full output; all G7 examples and the 18 existing YAML thresholds pass; inputs can be connected after construction.

## 6. W3 — preparation fidelity and stage 4

Add the single `ALARIC_DIR` resolver and launcher option in preparic; all Alaric paths derive from it. Load factory code only under the boundary's restrictions; attach computation modules inside transformations, never import installed `alaric` into the backend process.

Make RNA preparation produce `--rna --heavy --nalib --no-renumbering` output, never `--manual` or lenient flags. Protein preparation remains heavy-atom PDB2PQR with original numbering. Establish browser/backend agreement on range parsing, validity and extracted atoms under D7/D15. Preserve the byte identity of the prepared RNA for the later export seam.

Wire T0–T4, T6/T24 and shared B4/B5/B6/B7. T24 lists fragments/pairs from A2, before fitting. Then wire gated T5→T7→T8 and T9, with reference and best-fit PDB texts available through small plain results for overlays. Keep folder results backend-only.

Implement stage 4 around a keyed pairs/bases table:

- Continue 3→4 restores the retained table or creates a current fragment skeleton, and resets only a clip whose PDB/range-1 basis is stale. D5/D6 govern the initial-value and early-entry choices.
- Show rounded computed values alongside editable ones; per-fragment residues, sequence, conformer and fit RMSD; per-pair direction/trinucleotide; overlays on hover. Hide raw values.
- Copy uses only fresh A4 output and confirms before overwriting edits. Support typed values, per-cell revert, diff marks/count, finite positive values with at most two decimals, and a warning above 10 Å. Reject extra decimals rather than rounding them.
- Keep the same edit cells available in stages 4–6; stage 6's base-threshold editor is another view of those cells.
- Enforce clip integer/count/anchor constraints under D2; grey clipped fragments and any pair touching one, but leave their thresholds editable. A newly selected anchor that conflicts with the retained clip is visibly invalid until resolved.
- T10 validates against S22, so typing does not wait for the fit. T11 produces kept fragments and pairs as a subset; no clipping input reaches A1–A5.

Implement gates exactly: 2→3 needs confirmed valid rows and nonempty fresh stackings; 3→4 needs a keyed nonempty selection and fresh prepared reference fragments; 4→5 needs the required completed thresholds, valid clip and fresh energetics. A fresh computed threshold table is needed for Copy, not for advancing with typed values.

**Exit:** stage 4 opens during a held-back fit, permits typing, closes Copy while computed thresholds are stale, retains appropriate edits across refits, and waits for scoring only at 4→5. Clip edits do not change library/scorer identities.

## 7. W4/W5 — gridless energetics and stage 5

### Alaric A5 deliverable

After the documented attract-jax prerequisite is available through Alaric's pinned submodule, implement `util/energetics_seamless_support.py` and its eight-entry DEPS file from CB §4. An optional factory may encapsulate staging; it must leave ligand-folder and receptor bytes unbound.

Reduce the receptor once and every ligand once per invocation, using explicitly staged reduction rules and force-field parameters. Score identity poses with float64 gridless numpy, epsilon 15, no minimization/cutoff or JAX/scipy/compiled kernel. Return schema, bead counts, energies and per-ligand errors; fail the whole invocation on receptor reduction failure. The output contains no range index.

Test in both declared environments against the numpy identity CLI to `1e-6`, including unknown-ligand-residue handling. Use the Python float64 result; the old CLI's float32 `.npy` output is not the threshold source. Test both dependency completeness and minimality: listed edits invalidate, grid/JAX-only edits do not.

### Preparic integration and editor

T12 joins full reference/best-fit ligand folders. Reconcile one T13 instance per protein range, selecting each receptor file by content so an edit to one range leaves other scoring identities unchanged. Range numbering belongs to T14's display/merge, not to A5's computation identity. Verify that upstream extraction/preparation also preserves identical per-file content; folder-wide changes must not force scoring every range.

Implement pure interaction functions before the components: anchor collapse by `(RNA residue, range)`, lock lookup through the fragment map, Recompute, grid↔columns conversion, read validation and clip adaptation. Keep the original stacking/protein-residue keys for Gaussian rows/jobs; matrix-column collapse must not collapse distinct Gaussian anchors.

Recompute starts from locks and grows while either absolute reference/best-fit energy is at least 1, excluding the first low-energy/error/clipped row. Ranges grow independently; same-range locks/runs join under the oracle rules. Cap movement and `until next anchor` obey locks and same-range constraints. Ranges without anchors remain all false; positive clashes count as interactions.

Persist only kept addressed rows × ordered ranges. On read, validate shape, booleans, exact extent, lock coverage and anchors in every true run. Disable an invalid editor, identify the failed check and offer Recompute without writing automatically. Backend hand-off validation remains structural; the full export-time anchor validation is deferred as directed by the author.

On Continue 4→5, restore a matrix with unchanged selected anchors, read its old clip from row addresses and change only the outer ends of the first/last columns **per range**. Preserve interior caps/joins. D3 controls newly uncovered rows; a missing/inapplicable matrix uses Recompute. A racing clip edit invalidates the result's extent and leaves the project at stage 4.

T16 derives true-cell thresholds from best-fit energies using attached A6; reference energies remain informational. Write the exact P5.7 JSON with two-decimal strings and true cells only. T15 writes schema, clip, actual kept residues/ranges, runs with candidate anchors, and fragment→ranges map. D12 must be resolved before a lock or manual run with missing best-fit energy can become export-ready; never fabricate an energy, drop a lock or silently omit its threshold.

**Exit:** golden energy-threshold JSON, independent-range rescoring, matrix round trips and all refused reads pass; clip adaptation preserves every protected cell; matrix edits recompute only threshold derivation/hand-off; per-ligand failures remain row-local and their actionable matrix consequences are shown.

## 8. W6 — Gaussian specification, caches and Alaric validation

First extract `anchor_precalc.py` as a pure move: cache loading, Hopf snapping, translation/cell lookup, deduplication and capacity. Add public candidate-cell and single-cell-fill functions shared by the action and A9. Preserve legacy Gaussian anchor outputs bit for bit before changing names, dtype or filter behavior.

Implement the numpy Gaussian model/generator from the source description in [gaussian-filter-sources.md](gaussian-filter-sources.md). Pin the recovered model/report provenance and a compact test dataset when implementation obtains those sources. The branch contributes its grid/CSR algorithm and a frozen legacy model, not the new cutoffs. Do not copy the obsolete per-variant cutoff table or fallback logic from the old Gaussian mini-plan.

Generate A7's committed JSON at `$ALARIC_DIR/gaussian_stacking_filter.json`: full-precision model, six fixed base cutoffs, twelve names, masks/envelope, achieved recalls/counts, golden vectors, `filter_set_version`, legacy entry and compressed/decompressed checksums. Prefilter variants inherit their parent's cutoff exactly. Counts must be measured under those fixed cutoffs, not copied from re-derived-variant tables.

Generate A8's twelve gitignored CSR caches offline over 98,304 Hopf cells and the 0.25 Å lattice in ±5 Å, excluding the undefined origin. Use algebraic strict-mask tests at lattice boundaries; do not decide ties through `arccos`. Use bounded-memory rotation blocks and retain int32 translation indices where valid. Fix array order, serialization/compression settings and generator provenance so committed byte checksums can be reproduced; distinguish compressed-byte identity from decompressed `.npy` identity.

Test model vectors at relative `1e-12`, parent/variant cutoff equality, strict 30°/45°/envelope boundaries, CSR invariants, sampled membership and legacy reproduction. Change runtime name validation/resolution to use A7 and include cache identity in the action sigil. Preserve the legacy set separately.

Implement A9 as one sandbox-tested step over the entire job list and all filters. Use A1 pose indices, range PDBs, library/exclude identities, A7/A10 and the pinned cell-fraction constant. Derive containment from JSON, walk broad-to-strict and stop contained branches when a cell is not emitted. Keep implied results explicit: an implied rejection is not a measured zero fill and must not acquire invented count/score fields.

Verify installed compressed/decompressed caches before use; missing/mismatching files raise an installation error naming the generator command. Reuse loads/capacity work across jobs within a run. Measure verification, loading, capacity and evaluation together; only decompression is excluded from the few-seconds target. Do not silently redefine the target as warm-cache arithmetic time.

Define A11's constants in Alaric once and mirror/test them in preparic. Use `L_ov = sqrt(3)` for coverage, not the internal loop bound 1.3; retain tolerance `1e-5` and fraction `1/3`. No exported environment override may change the fraction.

**Exit:** about 20 jobs finish within a few seconds excluding decompression; A9 emission equals actual anchor cell emission; containment is monotone; every emitted pose meets capacity/fraction rules; reconstruction errors above `0.002 Å` and incomplete rings produce per-job errors.

## 9. W7 — stage 6 and both validation panels

Wire T17's checksum-pinned JSON/rings into the browser model; reject schema/name errors and run golden self-tests before 5→6. Stage-3 descriptor labels say “slip (from plane)”. T18 reuses preparic's detection code on best-fit fragments. D8 must settle the geometry alignment before claiming browser/A9 score parity: SVD/Kabsch differences and PDB coordinate precision must be checked against P6.2's `1e-6` bound, not excused by loosening it.

Compute Gaussian scores/classes and auto prefilters in browser pure logic. Use the worst candidate for each stacking under D9, choose the smallest passing base class, and show below-RRM-95 rows red while retaining class 95. The reference score is information only. Load legal filter names from A7 and share classification/name fixtures with backend validation.

Implement the retained editable class table, Copy confirmation, per-row revert/diff, flag dropping/reseeding on class changes, and 500 ms debounced writes. Code/model/refit changes update computed values and diff marks without overwriting the user's classes.

T20 creates one job per existing kept candidate, independent of class edits. T21 runs A9; T22 merges the selected filter into the class rows instantly. Display the oracle's worst-candidate states and pending/error conditions. Implied results need a truthful non-green display; if a precise fill is unavailable, show that it is implied by the containing filter rather than report 0%.

T23 uses the **same** threshold-table base cells and A4 raw fields to classify each candidate at `±1e-5`, and marks over-limit values as hard export blockers. Compare browser and A9 scores and surface discrepancies above `1e-6` as bugs. Do not introduce a second threshold copy, `anchor_modes`, or a mode selector.

**Exit:** class edits leave T20/T21 identities unchanged; threshold edits leave fit/scoring unchanged; both panels track optimistic edits and remote writes; invalid model self-tests prevent entry; validation failures never silently discard a selected anchor.

## 10. W8 — Alaric actions and the deferred export seam

Implement `energy-filter` with one fragment, one range, strict `E < T`, explicit numeric or P5.7-file thresholds, and missing-cell/schema errors. Multiple ranges are successive AND filters; never sum their energies. Make the all-gridless path the correctness oracle first. Remove `score`, `filter`, `score_add` and move or remove their scripts as directed; migrate affected fixtures/examples in the same work package rather than leaving references to removed actions.

Then address the 10–12 on-grid-node bug and cached-grid/cascade work. The proposed margins 1.0/0.1 come from finite samples; they are not a proof that arbitrary grid classifications equal gridless results. Under A12, accept/reject without refinement requires a justified error bound for that domain. Until such a bound exists, grids can order work, but unresolved decisions must receive exact scoring. Do not weaken byte-identical all-gridless equivalence to agreement on the available sample. D16 settles the remaining runtime count convention without changing numerical output.

Test strict comparisons, margin edges, injected-error refinement, exact output/provenance across chunking and process counts, and per-range AND behavior. Grid code stays outside A5's minimal dependencies. New local action templates are in scope; remote execution remains excluded.

Correct `sequence: auto` and `rmsd.py --fragment` to use Alaric's fragment mapping. Test a non-1 first residue and a numbering gap, independently of whether preparic initially allows gaps. A chain break with consecutive labels is also a case the preparation validator must examine; checking numbers alone does not establish chemical continuity.

Expose the fixed export inputs/readiness as a coherent internal bundle: exact prepared RNA bytes/checksum, common exclude/fraction, kept fragment map, edited pair/base thresholds, energy-threshold file, hand-off, class table and both validations. Keep hard blocks separate from the browser-computed “Proceed anyway” condition. Cover X1–X7 with fixture-level checks now; defer the actual export page and artifact layout. Any future export request must bind one complete decision snapshot, and an intervening edit must make its result stale.

## 11. W9 — acceptance, documentation and completion

Use the oracle clause IDs as the test matrix. Favor a small number of meaningful integration scenarios over tests that merely duplicate helper implementations.

| Area | Required evidence |
|---|---|
| Scientific outputs | 1b7f step goldens; exact table/rounding values; A2/A1 key and atom agreement; numerical parity across declared environments |
| Boundary isolation | isolated transformation for A1–A6/A9; A7/A8 offline generation/provenance checks; blocked undeclared imports; dependency-change invalidation; locator relocation cache hits; unknown-schema failures |
| Shared stage | two tabs; Continue/Back races, late orphan recreation, stale-range and clip writes, upstream change/revert, no gate skipped |
| Durability/connectivity | persisted CIF/decisions; backend restart at each reached stage; same stage before results return; disconnect/reconnect rereads and dropped deferred writes |
| Freshness/errors | hold back results while editing/reverting; protein-range changes with unchanged fragment keys; code/model restart; conversion/installation errors in correct stage scope |
| Cost of edits | execution counters/identities prove clip→no fit/scoring, classes→no A9, matrix→only T15/T16, one protein edit→one scorer |
| Matrix | stop-row exclusion, forced joins, independent overlaps, round trips, every refusal, clip shrink/expansion and race invalidation |
| Gaussian/actions | model self-test; actual anchor parity; implied results; few-seconds A9 check; byte-identical exact energy filtering; consistent fragment numbering |

Run one end-to-end 1b7f preparation through stage 6, then repeat restoration, threshold edits, matrix reuse, class edits and a two-tab conflicting transition. A golden fixture alone is insufficient for assembly-copy and missing-residue behavior; add the small synthetic/renumbered fixtures required by D7/D15 and A2/A12.

Update preparic's launcher/setup and stage documentation, Alaric step/action/dependency and cache-generation instructions, and the final share/node inventory. Correct the authorized obsolete claim in `alaric-project-layout.md`: edited preparation thresholds are the planned export input; legacy `DATA/constraints.json` is not automatically populated from best-fit tables. Keep old mini-plans as history with pointers to the overriding contract instead of trimming/reinterpreting them wholesale.

Completion means stages 1–6 meet their resolved oracle clauses, every delivered Alaric step passes its sandbox oracle, and fixed A12/export-seam obligations have evidence. Unresolved author choices remain explicitly marked; the deferred export stage is not reported as implemented.

## 12. Decisions needed from the author

D1–D16 match oracle-contract §7; D17 is an additional architecture decision exposed by the focused source check. Recommendations are draft defaults for review, not recorded author decisions. Work independent of each choice can proceed; the dependent behavior should not be finalized by inference.

1. **D1 — Back retention.** Apply deactivate-and-keep uniformly to ranges, stacking selection, threshold table and classes, as already required for the matrix. Recommended: yes. Needed to finalize W1 restoration and navigation tests.
2. **D2 — clip/anchor boundary.** Choose strict preservation of every candidate fragment (recommended), or preservation of the anchor residue with an outer candidate allowed to disappear. Separately choose visible invalid clip + closed 4→5 after a newly selected anchor conflicts with it (recommended), or automatic clamping on 3→4. Needed by W3/W5; clamping would be an explicit exception to the no-silent-correction rule.
3. **D3 — newly uncovered matrix rows.** Recommended: regrow only outer ends that touched the old clip boundary, using the energy/error-stop rule on new rows. Alternatives: extend all new rows true, or leave them false. Preserve user caps that ended before the old boundary. Needed by W5 adaptation.
4. **D4 — clipped blanks.** Require only kept threshold rows at 4→5 (recommended), or all rows including clipped ones. Under the first choice, uncovering blanks closes the gate until filled.
5. **D5 — stage-4 initial values.** Recommended: always create an empty skeleton with explicit Copy. Alternative: prefill if computed values happen to be fresh, which makes defaults timing-dependent.
6. **D6 — early stage-4 entry.** Confirm A2/T24 as the fragment-list source (recommended), rather than waiting for A1. A2/A1 key parity remains mandatory. This implements P3's stated gate without waiting minutes for fitting.
7. **D7 — assembly-copy meaning.** Choose asymmetric-unit-only ranges with copy suffixes refused and biounit display-only, or assembly-aware extraction that applies the chosen copy's operators. Recommended: assembly-aware extraction if picking copies remains part of the viewer workflow. This blocks final range identity/validation and W1's M2 basis.
8. **D8 — ring geometry.** Recommended: use Kabsch reference-ring geometry in the shared T4/T18 code and pin A10 for both, accepting changed stage-3 numbers. Otherwise resolve how the required A9/browser score agreement can hold with differing normals. Needed before W7 numerical acceptance.
9. **D9 — class granularity.** Confirm one class per selected stacking, taking the worse of its candidates (recommended), or choose per-candidate classes. Do not confuse distinct selected stackings with collapsed `(residue, range)` matrix columns.
10. **D10 — scored pose.** Confirm the discrete best-fit pose written by A1 (recommended), or choose exact superposition. The plan and current action-parity argument use the discrete pose.
11. **D11 — ovRMSD coverage.** Decide whether the hard block also covers `grow` pair thresholds (recommended: yes while it shares the same limited offset tables); confirm that over-limit anchor thresholds require lowering the value with possible “Proceed anyway”, or deselecting the anchor; choose Alaric rejection above `L_ov` versus extended tables. Recommended first delivery: explicit rejection and mirrored hard blocks; table extension is separate numerical work. Fragment 11 nucleotide 2 in 1b7f already exceeds the current guarantee.
12. **D12 — true cells without energy.** Recommended: editor/read validation refuses a true cell lacking best-fit energy. Alternative: permit it but block export. If such a cell is an anchor lock, make the conflict actionable and prevent progression that would imply complete thresholds; never silently deselect it. Needed by W5.
13. **D13 — “Proceed anyway”.** Clarify whether an energetics trigger anticipates editable energy thresholds; none are currently planned. Recommended: keep computed-only energies in this scope. Confirm amber-state treatment, especially the reference-anchor boundary zone; the current Gaussian non-green rule already includes amber. Hard stale/invalid/over-limit blocks remain non-overridable.
14. **D14 — persistent results.** Recommended: configure persistent Seamless result caching so a restart reuses expensive fits/tables. Decisions/CIF persist regardless; results still must be reread and checked for freshness. Needed for the restart performance target, not for stage restoration correctness.
15. **D15 — missing residues/gaps.** Recommended first delivery: refuse RNA ranges with unobserved residues at the stage-2 gate, with an explicit diagnostic. Alternative: support holes throughout clips, addressed rows, runs and residue mappings. Alaric's numbering fixes and A2 gap parity tests remain required either way. Consecutively numbered chain breaks require an explicit validity check as well.
16. **D16 — carried-over go-ahead/prerequisites.** Confirm the attract-jax accuracy-data commit/publication and Alaric submodule update, and the 10–12 grid bug fix already listed as awaiting go-ahead. Choose whether the “fewer than 1000 poses” test uses cascade-entry count or survivors after 5–7; recommended: survivors at the point 10–12 would run. This affects speed only; exact-output requirements remain. W4 needs the pinned numpy path, while the grid fix/cascade can follow separately.
17. **D17 — transition admission architecture.** The current share-only per-cell protocol cannot establish P3.4's gate-at-acceptance guarantee for cross-cell races. Approve an explicit correction to the proposed internal boundary: recommended, serialize transition acceptance with gate-affecting writes and check the expected snapshot/freshness at commit time, ideally through guarded existing share PUTs. If that requires unsupported server changes, evaluate one authoritative atomic decision envelope with admission validation; that alternative changes writable-share and mount layout. Keep the oracle's guarantee and later threshold editability. W0 must produce a concrete tested design before W1 is considered complete; entry tokens and frontend cleanup alone are insufficient.
