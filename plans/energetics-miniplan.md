# Stage 5 mini-plan: energetics (ATTRACT-JAX, gridless numpy) in preparic

**Purpose:** decide which RNA dinucleotides interact with which protein ranges. Every dinucleotide is scored against each protein range on its own, never against a merged receptor. Both sets are scored: the reference dinucleotides (`refe`, bound) and the best-fit conformers (`bestfit`, library). The user then edits one contiguous interaction run per stage 3 stacking anchor, starting from an energy-based Recompute. The stage depends on stage 3 (`stackings` + `stacking_selection`) and stage 4 (`refe-fragments`, `best_fit`, `best_fit_clip`).

## Inputs (all checksums)
| Input | Source |
|---|---|
| Reference dinucleotides | `ctx.refe_fragments.result`, **all** `frag-N-SEQ-refe.pdb` |
| Best-fit conformers | the `pdbs/` subfolder of `ctx.best_fit.result`, **all** `frag-N-SEQ-bestfit.pdb`, in the RNA frame. Always fitted with `--exclude-pdb-code <stage-1 code>`, so these are redocking-benchmark poses. |
| Protein ranges | `ctx.range_pdb_files_aa` without `range-1-aa.pdb`: aareduce `--heavy --pdb2pqr`, original numbering, same CIF frame as the RNA |
| Code + parameters | `alaric/parse_pdb.py`, `alaric/reduce.dat`, `util/reduce-npy.py`, attract-jax `util/{jax_scorer,pdb_io,grid_io,numpy_grid_score}.py`, `attract-original/attract-par.npz`, `epsilon` (15) |

The scoring transformers deliberately take no clips, anchors or matrix as input.

**Reduction to ATTRACT beads** happens inside the Alaric transformer, on every PDB. `parse_pdb` (Biopython) gives a ppdb, then `read_forcefield` + `reduce_ppdb` from `reduce-npy.py` give beads and types. This is the chain `score-single.sh` uses, and it needs no fraglib or templates. Gotcha: `reduce-npy.py` hard-codes `reduce.dat` next to `alaric/`, so the support script loads it with `importlib` and passes the staged path. Charges are dropped as in Alaric, so electrostatics are zero (RNA beads are uncharged).

## Transformers
**Alaric (new): `seamlessify_energetics(*, epsilon=15.0, conda_environment=None) -> Transformer`.** A bash transformer that runs `python -B energetics_seamless_support.py score ligands receptors`.
- Code, `reduce.dat` and `attract-par.npz` are checksum-pinned at construction (`files_to_checksums`). The folder pins `ligands` and `receptors` stay **unbound**, so preparic connects cells and never rebuilds the transformer.
- The worker reduces each receptor once. For each ligand it calls `JaxScoreOracle(backend="numpy", grid_free=True, energy_only=True)` with an identity pose.
- Output `energies.json`: `{schema:1, epsilon, receptors:{name:{beads}}, ligands:{name:{beads}}, pairs:[{ligand, receptor, energy, error}]}`. A reduction failure sets `error` on that pair only.
- Env: `conda_environment="alaric"` (Py3.12, numpy 1.26), as in stage 4. The `preparic` env also works.
- **Score only.** The numpy path cannot minimize, and minimizing would hide how well the discrete library pose fits.

**Preparic.**
- `_select_protein_pdbs(pdbs_aa) -> folder` drops range 1. Later it also swaps in unbound structures.
- Two factory instances, `energetics_refe` and `energetics_bestfit`, score the full sets against the same receptors.
- `_merge_energetics(refe, bestfit, best_fits, range_specs)` builds the shared plain cell `energetics`: one row per dinucleotide, in `best_fits` order, as `{fragment, resids, sequence, energy:{<range>:{refe, bestfit, error?}}}`.
- `_interaction_handoff(best_fits, best_fit_clip, interaction_matrix, range_specs)` builds the downstream output (see below).

## Shared clipping with stage 4
- **One state:** energetics uses stage 4's `ctx.best_fit_clip` (`deployment-mounts/best-fit-clip.json`, shared read-write). Energetics adds no cell, mount or copy.
- **Frontend:** stages 4 and 5 bind to the same `appState.clip` through the single `best_fit_clip` share key, with the existing pending-write and loop-avoidance logic. An edit in either stage shows up in the other, and both use the shared `ClipControls.svelte` and the validation in `best-fits.ts` (`clip1 + clip2 ≤ n-1`).
- **No rescoring:** clips are not a scoring input. Clipped rows (outside `clip1+1 … n-clip2`) are greyed and their energies stay visible.
- **Reset:** stage 4 resets the clips to `{0,0}` when `best_fits` changes. Energetics never resets clips. While `energetics` does not match `best_fits`, stage 5 shows "scoring…".
- **Clips apply only downstream,** through stage 4's `clipped_fragments` and the handoff below. **Stage 4's plan needs a fix:** its lines 3 and 42 still say energetics consumes only the unclipped dinucleotides or `clipped_fragments`.

## Interaction matrix
**Anchors** come from the stage 3 stackings the user selected. Each one pairs RNA nucleotide r with range `protein_range_pdb_id`.
- Duplicate anchors with the same (r, range) are collapsed into one.
- An anchor locks both dinucleotides that contain r: frag r-1 (as `nucleotide: second`) and frag r (as `first`). At the RNA ends only one of them exists, and clipped dinucleotides are not locked.

**Frontend model** (`src/lib/interactions.ts`, pure, Vitest): an ordered list of **columns, one per anchor**, sorted along the RNA.
- Each column is `{anchor, range, lock:[lo,hi], start, end, untilNext}`, where `start ≤ lo` and `hi ≤ end` (row indices, inclusive).
- `untilNext` is allowed only when the next anchor downstream is on the same range. It removes the cap between the two columns, so their runs merge into one run that reaches the next anchor.
- Anchors on the same range whose locks overlap or touch (nucleotides r and r+1) are merged permanently: `untilNext` is forced on and its checkbox is disabled.

**Invariants** (checked by `validateColumns`, and used by the editor to limit cap moves):
1. Each column is one contiguous run, and its lock lies inside it.
2. **No overlap within a range:** columns on the same range stay disjoint unless they are merged with `untilNext`. Columns on different ranges may overlap freely, so a dinucleotide can interact with several ranges.
3. No run includes a clipped row.

**Recompute** is a button that is always enabled. It rebuilds every column from the bound energies with this fixed rule:
- Start from the lock and grow towards 5' and towards 3'. Each step adds the next row as long as `|E_refe| ≥ 1` **or** `|E_bestfit| ≥ 1`. Growth stops at the first row where **both** are `< 1`, and that row is excluded.
- Growth also stops at a clipped row, at a row with a scoring `error`, and at the lock of another anchor on the same range.
  - If that lock is on the same range and the grown runs touch or overlap, the two columns merge and `untilNext` is ticked.
- Each range's columns grow independently, using that range's energies; other ranges' columns and locks are ignored.
- A range with no selected anchor has no column, so it gets no interactions.

**Editor** (`Stage5Energetics.svelte` + `AnchorColumn.svelte`): a list of capped columns, which replaces the 2D checkbox grid.
- Each column shows its anchor (nucleotide, protein residue, range) and a strip of dinucleotide rows with the refe | best-fit energies for its range, colour-capped at about ±20.
- The upstream and downstream caps move by dragging or with ± buttons. A cap cannot pass the lock, enter a clipped row (grey), or overlap a neighbouring column on the same range.
- An "until next anchor" checkbox appears where the next anchor is on the same range.
- A sequence overview colours each dinucleotide by its range. Hovering a row overlays the PDBs and the range in Mol*.

**Wire and storage format: a 2D grid.** `interaction_matrix` is a `list[list[bool]]`.
- Rows are **all** dinucleotides in `best_fits` order, and columns are the ranges in `range_specs` order. Clipped rows are always `false`.
- The backend treats it as a pure frontend input, following stage 3's `stacking_selection`:
  - a plain cell, mounted file-authoritative at `deployment-mounts/interaction-matrix.json` and shared read-write;
  - a startup loader that accepts only a rectangular bool matrix and rewrites anything else to `[]`;
  - writes through `cell.set`, reusing the pending-write signature, revision marker and loop-avoidance code (`writeInteractionMatrix`/`receiveInteractionMatrix` in `seamless.svelte.ts`).
- `columnsToGrid(columns, nRows, nRanges)` sets `start…end` to true in each column's range column. One row may be true in several range columns.
- `gridToColumns(grid, anchors, clip)` works on each range column of the grid on its own:
  - Each true run must contain at least one lock of that range.
  - A run with k anchors becomes k columns joined by `untilNext`. The outer ends of the run become the outer caps.
  - Two separate runs on one range mean `untilNext` is false between them.

**Reading from the backend:**
- **`[]` means undefined.** This works exactly as for the stacking selection: `receiveStackingSelection` sets `stackingSelectionNonEmpty = false` for an empty list, and `App.svelte` `complete()` then computes the defaults (`mergeStackings([], [], …)`) and writes them. In the same way, "Continue" from stage 4 runs Recompute and writes the result when `interactionMatrixNonEmpty` is false.
- **Validation before `gridToColumns` is accepted.** The read is refused if any of these fail:
  - the shape matches `energetics` × ranges, with booleans only;
  - no clipped row is true;
  - every true run contains an anchor of its range, which gives one contiguous run per column. Same-range overlap cannot occur in the grid, because a run holding two anchors is read as merged. Rows true in several ranges are allowed;
  - every unclipped anchor cell is true.
- **On failure the grid is never repaired silently.** The editor is disabled and a banner names the failed check, for example: "range 2: frag 7–9 contains no anchor. Press Recompute to rebuild from the energies; this overwrites the stored matrix." Recompute stays enabled. Nothing is written until the user presses it.

**Reset rules:** the matrix resets (`[]`, recomputed on the next stage-5 entry) **only on a stage 3 change**. That means a stacking-selection edit, or leaving stage 3 backwards: `backToRanges`/`goBack` (preparic `src/App.svelte:182-202`) also write `[]` here. In the UI, the dinucleotides, ranges and anchors cannot change any other way. Ranges are editable only in stage 2 (`App.svelte:64-74`), and reaching stage 2 from stage 3 already clears the selection. Exceptions, none of which needs a reset:
- **Clips** (stage 4): keep. The shared `ClipControls` runs `applyClip(columns)` in the same user action, in any tab with a valid matrix loaded. It clamps the caps, re-adds newly unclipped lock cells, and writes the grid.
- **Energies** change without any stage 3 change when Alaric, attract-jax or the fraglib checksums change. The factories pin them, so the change shows up on the next backend start. The matrix stays valid, because validation does not use energies. Keep it; Recompute is available.
- **Backend restart:** `cif` is unmounted and starts as `""` (preparic `backend.py:132-133`), so the upstream cells are empty until the browser refills them. The rebuilt content is identical. Treat empty or mismatched `energetics` as "scoring…", and never write `[]` because of it.
- **Out-of-band edits** to the file-authoritative `pdbcode`/`biounit`/`ranges` mounts (`backend.py:126,130,138`), or from another tab, can change the stackings without writing `[]`. `App.svelte:52-55` resets on a biounit change only in stage 2, and `mergeStackings` auto-selects new rows (`src/lib/stackings.ts:127-142`). Read-time validation (a missing anchor cell, or a run without an anchor) catches this and offers Recompute. This is a stage 3 gap and is not patched here.

**Downstream:** `_interaction_handoff(best_fits, best_fit_clip, interaction_matrix, range_specs)` writes `deployment-mounts/interactions.json`:
`{clip1, clip2, kept_residues, ranges:[{index, chain, start, end}], runs:[{range, first_fragment, last_fragment, anchors:[{fragment, nucleotide}]}], interactions:{"frag-N-SEQ": [ranges]}}`.
- It covers kept rows only and travels alongside stage 4's `clipped_fragments`.
- The backend re-checks only the shape. Anchor checks come later (open issue 3).

## Implementation steps
1. **attract-jax/alaric housekeeping:** push `9453ee6`/`22b23e4`, settle the untracked `docs/*.csv|json`, and bump Alaric's gitlink (currently `12bba50`, which has no numpy path).
2. **alaric** `util/energetics_seamless_support.py`: `input_paths()`, plus a `score` subcommand that writes `RESULT/energies.json`.
3. **alaric** `util/seamlessify_energetics.py` and `tests/1b7f/test_energetics_seamless.py`. Score `best-fits/*.pdb` and the refe fragments against `pdbs/1b7f_dom{1,2}-aa.pdb`. Check parity with `minfor.py --oracle numpy --no-grid --identity` (about 1e-6), and agreement with the grid-based `best-fits/*.score` within 0.5.
4. **preparic** `energetics.py`: `_select_protein_pdbs`, `_merge_energetics`, `_interaction_handoff`, and `_read_initial_interaction_matrix` (a copy of the stacking loader).
5. **preparic** `backend.py`: `_add_energetics_pipeline(context)`. It adds the transformers, `energetics`, `interaction_matrix`, `interactions`, and the mounts and shares, reusing `ctx.best_fit_clip`.
6. **frontend:**
   - Add `Stage = 5`, the `energetics`/`interaction_matrix` share keys with an `interactionMatrixNonEmpty` flag, and the state-graph errors.
   - In `interactions.ts`: `anchorsFromStackings`, `recompute`, `columnsToGrid`, `gridToColumns`, `validateGrid`, `validateColumns` and `applyClip`.
   - Components: `ClipControls.svelte` (shared with stage 4), `Stage5Energetics.svelte` + `AnchorColumn.svelte`.
   - Stage 4 "Continue": Recompute if the matrix is empty. Stage 2/3 back-navigation and stage 3 selection edits: write `[]`.
7. **tests:**
   - Vitest: Recompute (stop-row exclusion, clip stop, same-range merge, independent overlapping ranges), grid ↔ columns round trips, each refusal case, the reset rules, and clip sync.
   - pytest: loader rejection, clip edits leave the scoring inputs unchanged, handoff filtering.
   - An opt-in 1B7F integration test.
8. *(phase 2)* Unbound receptors: `unbound.py`, `UnboundTable.svelte`, and two more energetics instances.

## Open issues
**1. Unbound protein per range.**
- *Sources:* (a) another PDB entry (code, chain, optional residues; 3SXL for 1B7F); (b) AlphaFold DB via SIFTS, where UniProt numbering differs from author numbering; (c) the bound range itself, the default (Δ = 0).
- *Selection:* a per-range row in stage 5. The browser downloads the CIF into a share cell, so transformers stay network-free.
- *Alignment:* `_align_unbound` (Biopython) aligns sequences globally to the bound range and trims to the aligned span. It then does an iterative Kabsch fit on matched CA atoms (drop pairs over 2 Å; skip AF residues with pLDDT <70), renumbers to bound resids, reports RMSD, coverage and identity, and aareduces the result like the bound ranges.
- *Recommendation:* (c) now, (a) in phase 2, then (b). Validate against the pre-superposed `tests/1b7f/pdbs/3sxl_dom{1,2}-aa.pdb`. Unbound energies are display-only; Recompute always uses the bound energies.

**2. Clipping the RNA.** This is decided. End-clipping belongs to stage 4. Energetics shares its state, greys the clipped rows, stops growth at them, and applies the clips only downstream. The interaction matrix decides which dinucleotides touch which range.

**3. Backend anchor validation** (later): check in the backend, and in `_interaction_handoff`, that the grid contains every unclipped anchor cell. Until then, only the frontend enforces this, at read time.

**Questions for the user**
- **Ranges without an anchor:** they never get interactions. Is that intended?
- **Reset check (resolved):** the matrix resets only on a stage 3 change, so there is no question about keeping edits. Out-of-band edits are caught at read time; see the reset rules.

**Other risks**
- `reduce.dat` lacks PDB2PQR or modified residue names (HIE/HSD, MSE, …). Such pairs carry `error`, and aareduce may need a renaming step.
- Never mix grid-free scores with Alaric's grid/JAX `score` actions. Should `score-single.sh`/`score-fragments.sh` move to `--oracle numpy --no-grid`?
- The best-fit PDBs use template numbering (chain B, resid 1-2). Rows, clips and the matrix key on `frag-N-SEQ` and the row index, never on resid.
