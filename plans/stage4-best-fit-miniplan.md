# Stage 4 mini-plan: best-fit statistics, thresholds and RNA clipping in preparic

**Scope.** Stage 4 does four things:
1. Fit every dinucleotide of the RNA range to its best library conformer.
2. Derive the ovRMSD/cRMSD threshold values (the pairs and bases tables).
3. Show both.
4. Let the user clip poorly fitting dinucleotides from either end, with `clip1` (5') and `clip2` (3'). Clipped rows turn grey.

Hand-off to energetics (stage 5, see `energetics-miniplan.md`): **all** fragment PDBs from both sets, plus the shared clip state `best_fit_clip`. Energetics scores every dinucleotide, so clip edits never cause rescoring. It also edits clip1/clip2 through the same shared editor. Clipping applies only further downstream. Thresholds are computed and shown, but **`constraints.json` is not written**; that export is a later stage. Remote execution is out of scope.

## Inputs from stages 1-3
| Input | Where it lives now |
|---|---|
| RNA PDB in fitter format | `ctx.range_pdb_files_aa["range-1-aa.pdb"]` (`deployment-mounts/pdbs-aa/`) |
| PDB code | `ctx.pdbcode`. It is always passed, lower-cased, as `--exclude-pdb-code`/`--exclude`, with no toggle. All three tools must get the same code, or conformer indices point at the wrong coordinates. |
| Fragment-to-residue mapping | `ranges.yaml` row 1. Fragment N covers residues start+N-1 and start+N. |
| Protein ranges (energetics) | `pdbs-aa/range-{2..}-aa.pdb` |

**No lenient flags (verified):** `--ignore-unknown/missing/reordered` stay off. The fitter's input `range-1-aa.pdb` is produced by `range-pdbs-aareduce.sh:17-18` (`--rna --heavy --nalib --no-renumbering`, wired in at `backend.py:162-173`). That output has no unknown residues and no missing heavy atoms, and its atoms are in topology order, matching `fraglib/templates`. Caveats: aareduce fails on unknown or modified nucleotides instead of converting them, so such ranges fail before stage 4 is reached. Preparic must not pass `--manual` to aareduce.

## Backend: use the factories, not the CLIs
Use `seamlessify_refe_best_fit` (`util/seamlessify_refe_best_fit.py`). It returns a bash `Transformer` that can be assigned into preparic's `seamless_workflow.Context`, like `aareduce_ranges`. The CLIs are subprocesses that write to disk, outside the reactive graph and the shares.
- **Import:** from the local alaric checkout's `util/` (`$ALARIC_DIR`, default `~/alaric/alaric`). This is the same meaning Alaric gives it: the `alaric/` subfolder. The repo root is `Path(ALARIC_DIR).parent`, so `util/` is `Path(ALARIC_DIR).parent / "util"`.
- **Worker env:** `conda_environment="alaric"` (Py 3.12, numpy 1.26, ruamel). The `preparic` env (Py 3.15, numpy 2.5) cannot run the fitter.
- **Checksums:** sources, the 20 templates and the RNA PDB are registered through `files_to_checksums`. The library identity is the `fraglib_identities` pin, built from the 36 committed `fraglib/*/*.CHECKSUM` sidecars. `~/.alaric/fraglib.yaml` is a meta-file and does not affect identity.
- **Caching:** the backend's existing `seamless_config`, initialized before the factory call. Execution is local.
- **Blocker:** the factory takes a host file path and bakes the arguments into `transformer.code`. Fix this in Alaric (step 1): leave the inputs unbound so preparic can connect cells to them.

## PDB sets for energetics
Both sets use one PDB per dinucleotide. Scoring works on single files as positioned (`score-fragments.sh`/`score-single.sh`). Both sets are built from the same `fraglib/templates/XX-ppdb.npy`, so they match atom for atom. A free test follows: the reference-vs-best-fit RMSD must equal the TSV `rmsd`.
- **A. Reference dinucleotides (new, in Alaric).**
  - Add a `refe-fragments` support subcommand and a factory `seamlessify_refe_fragments(rna_pdb=None)`. It reuses `Reference.get_coordinates(N, 2)`, the templates, `write_pdb` and the `write_pose_pdb` pattern.
  - Inputs (checksums): `rna.pdb`, sources and templates. No library identity is needed.
  - Output: a folder `{frag-N-SEQ-refe.pdb}`.
- **B. Best-fit conformers (exists).** The fit's result folder holds `pdbs/frag-N-SEQ-bestfit.pdb` in the RNA frame, next to `best-fits.tsv`.

## Thresholds (new Alaric wrappers)
Rule (from `stage4-thresholds-howto.md`): each threshold is that pair's or nucleotide's **own** table value, **always rounded up to the next 0.01 Å**, with no maximum taken. This reproduces all 18 hand-copied `tests/1b7f-*` YAML values. By user decision, no extra margin is added, and the UI shows only rounded values. `constraints.json` is legacy and unrelated.

1. **`seamlessify_refe_best_fit_tables(best_fit_tsv=None, rna_pdb=None)`** (bash, `alaric` env)
   - Runs `refe-best-fit-pairs.py` and `refe-best-fit-bases.py` in one job, through a new `tables` subcommand in `refe_best_fit_seamless_support.py`. It extends `input_paths()` with the two scripts.
   - **Input checksums:**
     - `best-fits.tsv`, picked from the fit's result folder;
     - `rna.pdb`, the same cell as the fit (bases only);
     - the `exclude_pdb_code` pin, the same cell as the fit;
     - sources, scripts and the 20 templates;
     - the `fraglib_identities` string.
   - `fraglib.yaml` is a meta-file. `grid_spacing` stays at the default shared by all three tools and is not exposed.
   - **Output:** a folder checksum `{pairs.tsv, bases.tsv}`.
   - **Dependencies:** the conformer and rotamer library on the local host (`/opt/fraglib`; loading it dominates the runtime). The **cRMSD matrices are not read**, but the `crmsds` key must exist in `fraglib.yaml`.
   - **Why a separate transformer:** the fitter's cached result and identity stay untouched, and the tables re-run only when the fit or PDB code changes.
2. **`util/refe-thresholds.py`** (new, stdlib only). The rounding rule lives in Alaric, beside the tools.
   - Computes `ceil` to 0.01, with a guard against float noise.
   - **Input:** the tables folder. **Output:** a plain JSON checksum, `{pairs:[{down,up,sequence,ovrmsd,crmsd,ovrmsd_raw,crmsd_raw}], bases:[{fragment,sequence,nuc1_ovrmsd,nuc2_ovrmsd,…raw}]}`. The `*_raw` fields are kept only for tests and are never displayed.
   - It is wrapped as a small transformer, `seamlessify_refe_thresholds`, and is cheap.

The library-free approximation from the best-fit PDBs is rejected for thresholds: one of 22 bases values flips because PDB coordinates have only 3 decimals. All of these are computed on the **full** fit. Clipping only filters, so editing clip1 or clip2 never re-runs a library step.

## Editable thresholds
The computed values (`best_fit_thresholds`) stay read-only. What goes downstream is a separate, user-editable copy.
- **State:** `ctx.threshold_edits` is a plain cell mounted file-authoritative at `deployment-mounts/best-fit-threshold-edits.json` and shared read-write. This is the clip and `stacking_selection` pattern: the frontend supplies the value, and the startup loader resets an invalid file to `null`.
  - **Shape:** `null` (not copied yet) or `{schema:1, pairs:[{down,up,ovrmsd,crmsd}], bases:[{fragment,nuc1_ovrmsd,nuc2_ovrmsd}]}`.
  - **Keys:** rows are keyed by fragment numbers, never by row index, so chain-break gaps stay safe.
- **Copy button:** "Copy computed → editable". If the table is `null`, it copies straight away. If edits exist, it asks first ("Overwrite all N values, including M edits?") and then replaces the whole table. Each edited cell also has its own "revert to computed".
- **Validation:**
  - Each value must be a finite number > 0 with at most two decimals. Extra decimals are rejected inline, not rounded silently, and values are stored as two-decimal JSON numbers.
  - Values above 10 Å only trigger a warning.
  - The set of keys must exactly match the current computed rows.
  - The frontend checks all of this inline. A backend transformer, `validate_threshold_edits(best_fit_thresholds, threshold_edits)`, re-checks it and raises, which shows as a banner.
- **When computed values change (recommended):**
  - **Same rows** (the `best_fits` signature is unchanged; the values moved, for example after a library or code update): keep the user's edits. Mark every cell whose value differs from the newly computed one, and show a "differs from computed (x.xx)" count above the table.
  - **Different rows** (new range or PDB code; this is the same signature change that resets the clips): reset to `null` and prompt for a new copy, because the old edits refer to other dinucleotides.
- **Clipped rows:** greyed, exactly as in the computed tables, but still editable, so the edits survive if the clip changes back. Only kept rows go downstream.
- **Downstream:** `clip_fragments` reads `threshold_edits`, never the computed values, and writes the kept rows to `clipped-fragments/thresholds.json`. While the table is `null`, `thresholds.json` is absent and the later export stage refuses to proceed. Writing `constraints.json` is still not part of stage 4.

## Clipping: state, validation, downstream
- **State:** `ctx.best_fit_clip` is a plain cell `{"clip1","clip2"}`, default `{0,0}`.
  - It is mounted file-authoritative at `deployment-mounts/best-fit-clip.json` and shared read-write (the `stacking_selection` pattern). An invalid file is reset at startup.
  - The frontend `appState.clip` syncs through `seamless.svelte.ts`. `ranges.yaml` is not rewritten.
- **Semantics:** the kept fragments are `clip1+1 … n-clip2`, which corresponds to residues `start+clip1 … end-clip2`. A pair is kept only if `down ≥ clip1+1` and `up ≤ n-clip2`.
- **Validation:** non-negative integers with `clip1+clip2 ≤ n-1`.
  - The frontend checks this inline. The backend clip transformer re-checks it and raises, which shows as a banner.
  - The clips reset to `{0,0}` when the `best_fits` signature changes.
- **Downstream:** `clip_fragments(best_fits, threshold_edits, best_fit_clip)` is a Python transformer that returns a dict, which becomes a folder cell. It writes `clipped-fragments/` with:
  - `refe/frag-N-SEQ-refe.pdb` and `bestfit/frag-N-SEQ-bestfit.pdb` for the kept fragments;
  - `thresholds.json`, the kept rows of the **editable** table;
  - `clip.json`, the clips plus the kept residue range.

  The full `best_fit_thresholds` cell is also mounted at `deployment-mounts/best-fit-thresholds.json`.
- **Consumers:** energetics (stage 5) does **not** use `clipped-fragments`. It reads `ctx.refe_fragments.result` and the fit's `pdbs/` in full, plus `best_fit_clip`. `clipped-fragments` is kept only for the later export stage, after stage 5, which writes the Alaric project `DATA` and `constraints.json`. That stage reads it together with stage 5's `interactions.json`.

## Presentation
`parse_best_fits(fit result, refe result, range_specs)` produces the shared `best_fits` rows `{fragment, resids, sequence, conformer, rotamer, grid, rmsd, refe_pdb, bestfit_pdb}`. `best_fit_thresholds` is shared read-only.

- **Main table:** one row per fragment, with the columns residues, sequence, conformer, `rmsd`, base ovRMSD nuc1 and base ovRMSD nuc2. The two base columns come from bases and show the rounded threshold only.
- **Companion pairs table:** placed below the main table, with the columns `down→up`, trinucleotide, ovRMSD and cRMSD, rounded.
- **Editable threshold table:** `ThresholdEditTable.svelte` sits beside or below the computed tables, with the same bases and pairs layout, the copy button, markers on cells that differ, and per-cell revert.
- **Clipping:** the clip1/clip2 editors sit at the ends of the main table, in a `ClipControls.svelte` component shared with stage 5. Both stages bind to the same `appState.clip`. Clipped fragments and any pair touching them turn grey in both tables.
- **Viewer and status:** hovering a row overlays its reference and best-fit PDBs in Mol*. Separate "fitting…" and "computing thresholds…" states and error banners come from the state graph.

## Implementation steps
1. **alaric** `seamlessify_refe_best_fit.py`: `rna_pdb=None` leaves a `rna.pdb` pin unbound. Add an `exclude_pdb_code` pin. Extend the tests.
2. **alaric:** add an optional `templates=` argument to `Reference`, and the `refe-fragments` subcommand and factory.
3. **alaric:** add the `tables` subcommand, `seamlessify_refe_best_fit_tables`, `util/refe-thresholds.py` and `seamlessify_refe_thresholds`.
   - **Golden tests:** the tables must match `tests/1b7f/data/refe-best-fit-{pairs,bases}.tsv`, and the thresholds must reproduce the 18 YAML values.
   - **Clip test:** compare a filtered full run with a run on a sub-TSV.
4. **preparic** `best_fit.py`: `_pick_rna_pdb`, `_pick_best_fit_tsv`, `_parse_best_fits`, `_validate_threshold_edits`, `_clip_fragments`, the `threshold_edits` startup loader, and the alaric loader.
5. **preparic** `backend.py`: `_add_best_fit_pipeline(context)`. The fit, refe-fragments and tables transformers share the `rna_pdb` and `exclude_pdb_code` cells. It also adds the thresholds, `threshold_edits` and its validator, `best_fits`, `best_fit_clip`, `clipped_fragments`, the mounts and the shares.
6. **preparic** `environment.yml`/README: document the `alaric` worker env, `ALARIC_DIR` (the `alaric/` subfolder, default `~/alaric/alaric`; the repo root is its parent), `fraglib.yaml` (including the `crmsds` key) and `/opt/fraglib`.
7. **frontend:** add `Stage = 4` and `bestFits`/`thresholds`/`thresholdEdits`/`clip` state, share keys and state-graph errors.
   - `src/lib/best-fits.ts`: normalize, clip validation, clip-window and pair keep-rule helpers shared with stage 5, threshold-edit validation, copy, diff against computed, and reset on signature change. Covered by Vitest.
   - Components: `ClipControls.svelte`, `Stage4BestFit.svelte`, `BestFitTable.svelte`, `PairThresholdTable.svelte` and `ThresholdEditTable.svelte`.
   - Also: PDB overlays in `viewer.ts`, and a Stage 3 "Continue".
8. **Tests:** `tests/backend/test_best_fit.py` (parse, clip, threshold-edit validation and loader, and edits flowing into `thresholds.json`, with the 1b7f fixtures) plus an opt-in 1B7F integration test.

## Recommendations on earlier questions
- **Which pose to score:** use the discrete library pose (set B). It is what docking can sample, it matches the existing `.score` fixtures, and it needs no new code.
- **Residue numbering:** do not renumber; scoring ignores it, and renumbering would break the golden tests. If display needs it, do it in preparic.
- **Template loading:** give `Reference` an optional `templates` argument. Then `refe-fragments` needs no `fraglib.yaml`.

## Open questions for the user
- **Clipped rows:** should clipped rows in the editable table stay editable (recommended, since edits survive un-clipping), or be locked?
- **Gating:** should moving on to stage 5 require the editable table to have been copied, or should only the later export stage require it (recommended)?

## Risks
- **Runtime:** library loading runs twice, in the fit and in the tables. It runs once per RNA range and PDB code, then comes from the cache.
- **Not yet verified:**
  - whether the factory's upload works under the backend config;
  - whether a folder can feed a Python transformer (fallback: bash);
  - whether a chain break inside the range leaves gaps that pairs skip silently. The pair keep-rule must use fragment numbers, not row indices.
- **Rejected entries:** modified nucleotides are rejected at aareduce, so such entries cannot be fitted.
