# Stage 6 mini-plan: Gaussian stacking filter in preparic

> **Author decision, 2026-10-29:** installed Gaussian caches are under `precomputed-filters/gaussian/`, confirming the directory rename from `gaussian-rrm/`. Legacy-filter compatibility (r90, r95) is a non-goal; see [final/gaussian-precomputation-decisions.md](final/gaussian-precomputation-decisions.md).

Plan only; nothing is implemented. Builds on `alaric-gaussian-miniplan.md` (AlaGaussian, the "JSON contract"), `stage4-best-fit-miniplan.md` and `energetics-miniplan.md`. Numbers were checked in scratch with the `alaric` env; line numbers refer to the current `alaric/anchor.py`.

## 0. Premise and filter set
**Descriptors are computed in the backend today.** Stage 3's are produced by `detect_stacking.py`, a Seamless transformer. The browser only gets `distance`, `tilt_angle`, `slip_angle` (which is φ, from the ring normal) and the centroids, without normals. Stage 6 adds a TS geometry module that computes reference and best-fit descriptors with identical code (§2).

**12 precomputed filters (confirmed):**
- `rrm50`, `rrm60`, `rrm70`, `rrm80`: Gaussian only. Cutoffs from `envelope-analysis-5-report.md`: 3.3038, 2.9043, 2.0420, 1.1975.
- `rrm90` and `rrm95`, each × {none, `-tilt30`, `-disp45`, `-tilt30-disp45`}.

Alaric's JSON must list all 12. Preparic reads the filters from the local install, `$ALARIC_DIR/../precomputed-filters/gaussian/<name>/`, pinned by the JSON's `precalc` checksums. How Alaric produces them is AlaGaussian's concern.

## 1. Data flow
| Cell (backend) | Content | Share |
|---|---|---|
| `gaussian_model_json` | `$ALARIC_DIR/gaussian_stacking_filter.json`, mounted `mode="r"` | none |
| `gaussian_model` | built by `_client_model(json, ring_refe)`: `{version, sha256, mean, precision, normalizer, cutoffs:{50…95}, filters:{12 names → cutoff, lt_deg}, ring_references, golden}`. It raises unless `schema: 1` and all 12 names are present. | read-only |
| `range_pdbs` (exists) | `[0]` is the RNA, `[k-1]` is range k | read-only |
| best-fit PDB texts | `pdbs/frag-N-SEQ-bestfit.pdb` of `ctx.best_fit.result`, via stage 4's overlay share or a `best_fit_pdb_texts` dict | read-only |
| `gaussian_classes` | the editable table (§5) | read-write |
| `gaussian_validation` | server-side check (§7) | read-only |

**Reference rings.** `ring_references` is read from `$ALARIC_DIR/../ring-refe/refe-{A,C,PHE,HIS,ARG,TRP}.pdb`. These are the same pinned files the validator uses, so the Alaric JSON needs no change for them.

**`ALARIC_DIR` helper.** `preparic/alaric_dir.py`, shared by stages 4–6. The default is `~/alaric/alaric`, and the repo root is its parent. Settable with `preparic-start --alaric-dir`.

**Browser.** It parses PDB texts, then computes descriptors, Gaussian scores, computed classes and auto prefilters. It writes only the editable table.

## 2. Descriptors (reference and best-fit)
**Which dinucleotides.** Anchors are the selected stage-3 stackings. Fragment N covers residues `start+N-1` and `start+N`. Nucleotide r (row i = r − start) has up to two best-fit dinucleotides:
- frag i+1 as `first` (best-fit resid 1);
- frag i as `second` (resid 2).

Best-fit PDBs use chain B, resid 1–2, standard atom names, and the CIF frame. The protein ring is `protein_resid` in `range_pdbs[range-1]`, and the reference ring comes from `range_pdbs[0]`.

**Ring atom sets match training.** `stacking-prediction-plots/OLD/test-set/extract_rings.py` uses the same sets: 9-atom purines, 5-atom HIS/ARG, unweighted centroids.

**Normals do not match.**
- Training and `anchor.py:395` Kabsch-fit each ring onto `ring-refe` (G→A, U→C, TYR→PHE).
- Preparic's SVD plane deviates by up to 0.73° (tilt) on the 58 test-set PDBs.
- Kabsch onto Alaric's `ring-refe` reproduces training to float32 precision.

**Fix: `src/lib/ring-geometry.ts`.**
- Kabsch normals via the Horn quaternion method with a 4×4 Jacobi solver; atoms matched by name.
- d = |c_rna − c_prot|, φ = acos(|v·n_prot|/d) and θ = acos(|n_prot·n_rna|). φ is computed directly. Stage 3's `slip_angle` is the same φ (author, 2026-10-29), so no relabel is needed.

## 3. Gaussian, classes and auto prefilters (`src/lib/gaussian.ts`)
- **Score:** `score = exp(-0.5·Δᵀ P Δ) / normalizer`. The `golden` vectors are self-tested on load (rel tol 1e-12); a failure disables stage 6.
- **Class source:** the **best-fit** score. If an anchor has two unclipped best-fit dinucleotides, use the **worse (lower) of the two**, so either fragment can serve as the anchor action. The reference score is shown but never sets the class.
- **Class:** the smallest N in [50, 60, 70, 80, 90, 95] with `score >= cutoffs[N]`. If even RRM-95 fails, the class is 95 shown in red ("below RRM-95").
- **Auto prefilters**, for classes 90/95 only:
  - `tilt30` is checked iff every unclipped best-fit dinucleotide has θ < 30°;
  - `disp45` is checked iff every one has φ < 45°;
  - both comparisons are strict, in degrees.
  - A prefilter variant's cutoff is never higher than the Gaussian-only one, so the auto choice admits the best-fit descriptors.
- **Filter name:** `filterName(row)` = `rrm<N>` for N ≤ 80. For N = 90/95 it is `rrm<N>` + (`-tilt30` if tilt30) + (`-disp45` if disp45). This yields exactly the 12 names. A Python twin in `gaussian.py` is checked against a shared test table.

## 4. UI (`Stage6Gaussian.svelte`, `GaussianClassTable.svelte`)
**Computed table:** one row per anchor, in RNA order. It shows:
- the reference score with d/φ/θ;
- each best-fit dinucleotide's score with d/φ/θ, plus Δ against the reference;
- the class badge, red when below RRM-95;
- the auto prefilter ticks;
- the filter name.

Hovering a row overlays the best-fit PDB in Mol*.

**Editable table:** class dropdown, the two checkboxes (only for 90/95), the resulting name, a "differs from computed" marker with per-row revert, and the validation cell (§7).
- **"Copy computed → editable":** immediate if the table is `[]`. Otherwise it asks first: "Overwrite all N rows, including M edits?".
- **Switching class:** to 50–80 drops the checkboxes; back to 90/95 re-seeds them from the auto values.

**Clipping** (shared `best_fit_clip`): clipped dinucleotides are greyed and excluded from the class, the prefilter rules and validation. An anchor whose dinucleotides are all clipped has no class and no validation. Its editable row is kept, greyed, and not exported.

## 5. Editable table: storage, validation, reset, downstream
**Storage** follows the stage 4/5 pattern:
- `ctx.gaussian_classes` is mounted file-authoritative at `deployment-mounts/gaussian-classes.json` and shared read-write.
- The startup loader `_read_initial_gaussian_classes` rewrites a malformed file to `[]`.
- Writes reuse the pending-write, revision and loop-avoidance code. They are debounced by 500 ms (§7.4).
- **`[]` means undefined:** stage-6 entry copies the computed classes and writes them.

**Shape:** `[{key:[rna_resname, rna_resid, range, protein_resid], rrm, tilt30?, disp45?}]`. Rows are keyed by `stackingKey`, with one row per selected stacking, clipped ones included.

**Validation:**
- `rrm` is one of the six values;
- the flags are booleans, present iff `rrm` is 90 or 95;
- there are no duplicate keys;
- the key set equals the selected stackings.

This is checked inline and at read time. A failure disables the editor and shows a banner offering the copy; nothing is repaired silently. The backend validator `_validate_gaussian_classes` raises, which appears as a banner.

**Reset:** only a stage-3 change writes `[]`. That covers selection edits and `backToRanges`/`goBack`, at the same hooks as stage 5. Kept across:
- clip edits;
- fit changes, with differing rows marked;
- model version changes.

Out-of-band mount edits fail the key-set check at read time.

**Downstream:** only the editable table goes downstream. For each unclipped anchor, export emits `precomputed_filter_name: gaussian-rrm` and `precomputed_filter_threshold: filterName(row)`. Export refuses while the table is `[]` or invalid, or while `gaussian_validation` is stale or errored (§7.5).

## 6. Implementation steps
1. **alaric (AlaGaussian):** the 12 names in the JSON and in `anchor.py` name validation. `rrm50`–`80` are re-derived by `derive_cutoffs`.
2. **alaric:** the `anchor_precalc.py` refactor and the validation factory (§7.2).
3. **preparic:** `alaric_dir.py`, `preparic-start --alaric-dir`, README.
4. **preparic** `gaussian.py` and `backend.py:_add_gaussian_pipeline`:
   - `_client_model`, `filter_name`, the classes loader and validator;
   - `_gaussian_validation_jobs`, the 12 validator instances and `_merge_gaussian_validation`;
   - mounts and shares.
5. **frontend:**
   - libs: `pdb-atoms.ts`, `ring-geometry.ts`, `gaussian.ts` and `gaussian-classes.ts` (copy, diff, revert, validation, clip status);
   - share keys `range_pdbs`, `gaussian_model`, `gaussian_classes` and `gaussian_validation`;
   - `Stage = 6`, the components, and a stage 5 "Continue".
6. **Tests:**
   - Vitest:
     - golden vectors;
     - class and prefilter boundaries;
     - all 12 names;
     - Kabsch geometry on about 12 vendored test-set ring PDBs against `descriptors_test.csv` (≤1e-6);
     - copy, validation, reset;
     - the validation-cell states.
   - pytest: loaders, validator, job building, merge, name parity.
   - alaric: refactor parity, and a `cell_fill` cross-check (§7.2).
7. **Stage 3**: no relabel is needed, since its slip angle is φ. Optionally move `detect_stacking.py` to Kabsch normals.

## 7. Server-side validation of the class table
### 7.1 What is checked
The check runs per unclipped best-fit dinucleotide of each anchor, i.e. per candidate anchor action, with F = `filterName(row)`.
- **Best-fit pose:**
  - It is `(conformer c, rotamer k, grid g)` from the fit's `best-fits.tsv` (`conformer`, `rotamer`, `grid_x/y/z`).
  - `refe-best-fit.py:136-153` places it as `coords = conformer @ R_k + g·√3/3`, the same convention and spacing as `anchor.py:1020-1043`. So g is exactly the pose cell Alaric would emit.
  - The best fit and the anchor action must use the same library and `--exclude` pin (stage 4's cells).
- **Fill:** `fill = count / capacity` for cell g.
  - `count` comes from Alaric's own path for that single rotamer: Hopf cell snap (`:871`), accepted-translation lookup (`:885`), and rounding of `protein_center + t − ring_center` to the pose grid (`:1036-1046`). It is the number of accepted translations landing on g; 0 if the Hopf cell is not in the cache or none lands on g.
  - `capacity` is `_pose_cell_capacity` at g (`:619`).
  - `emitted = fill >= MIN_CELL_FRACTION`, which is ⅓ (`:49-51`). The value is pinned as a transformer argument.
- **Pose pass** (when fill > 0): descriptors of the actual, unsnapped pose in the protein-reference frame, computed with AlaGaussian's `gaussian_filter.descriptors` and `score`. Pass means score ≥ F's cutoff and F's strict prefilters hold. The score is returned so the frontend can compare it with its own value.
- **Sanity check:** the reconstructed pose must match the best-fit PDB within ≤0.002 Å RMSD. Otherwise the row errors, which guards against a library or exclude mismatch.

**Output per anchor:** `{key, filter, worst:{fill, emitted, pose_pass}, dinucleotides:[{fragment, nucleotide, count, capacity, fill, emitted, pose_pass, score, error?}]}`, plus the `gaussian_classes` checksum it was computed from.

### 7.2 Alaric code: a small refactor is needed
The logic is private and partly inline in the loop of `_process_conformer_precalculated`.
- **New module `alaric/anchor_precalc.py`.** Pure moves, no behaviour change:
  - `_PrecalculatedAnchor`;
  - `_load_precalculated_anchor` (`:636`);
  - the Hopf helpers (`:473-559`);
  - `_snap_cache_rotations` and `_translation_segments`;
  - `_deduplicate_owner_grid` and the capacity functions (`:585-633`).
- **New public functions:**
  - `candidate_cells(precalc, ring_coords, rotamer_matrices, protein_center) -> owners, grid, counts, capacity`, extracted from `:1008-1051`. `anchor.py` keeps only `keep = counts >= f·capacity` (`:1052`).
  - `cell_fill(precalc, ring_coords, rotamer_matrix, protein_center, grid_point) -> (count, capacity)`.
- **Guards:**
  - `tests/test_anchor_precalculated.py` and the `tests/1b7f-gaussian-anchor` outputs must stay bit-identical;
  - a new test checks that `cell_fill(...)/capacity >= ⅓` holds exactly for the poses an anchor run emits.
- **Factory.** `util/gaussian_validate_seamless_support.py` (`input_paths()`, `validate` subcommand) and `util/seamlessify_gaussian_validate.py`.
  - It builds a bash transformer with `conda_environment="alaric"`.
  - Code, the model JSON and `ring-refe/*` are pinned with `files_to_checksums`.
  - **Unbound pins:**
    - `jobs` (plain);
    - `best_fit` (folder: TSV and PDBs);
    - `proteins` (folder of range PDBs);
    - `filter_name`, plus `filter_identity`, the JSON `precalc` checksums of that filter;
    - `fraglib_identities` and `exclude_pdb_code` (stage 4's cells).
  - The worker reads `precomputed-filters/gaussian/<name>/` locally and verifies the SHA-256s before use. This is the fraglib identity pattern; nothing is staged.

### 7.3 Preparic wiring
- `_gaussian_validation_jobs(gaussian_classes, stackings, stacking_selection, best_fits, best_fit_clip, range_specs)` (Python) emits **12 job lists, one per filter name**. Each job is `{key, fragment, nucleotide, range, protein_resid}`.
- There are **12 static validator instances**, one per filter. An empty job list returns `[]` without loading the filter.
- `_merge_gaussian_validation` builds `gaussian_validation`, mounted at `deployment-mounts/gaussian-validation.json` and shared read-only.

### 7.4 When it runs and what it costs
- **Trigger:** Seamless re-runs on input change. Table writes are debounced by 500 ms, so a burst of clicks gives one run.
- **Scope of an edit:** an edit moves one anchor between two filters, so at most 2 of the 12 instances re-run. Reverting an edit hits the cache. A refit (stage 4) re-runs all non-empty instances.
- **Cost estimate (not measured; step 6 measures it):**
  - per filter run: SHA-256 and zstd decompression of `filtered-concat` (up to about 170 M pairs for `rrm95`), about 1–3 s and about 1.5 GB RAM with today's int64 cast (`:457`), halved by AlaGaussian's int32 fix;
  - the capacity table, about 1 s;
  - the library slice for c (mmap);
  - milliseconds per job.
  - Totals: about **2–6 s per edit**, and about 20–40 s cold when many filters are in use.

### 7.5 UI signalling (recommendation)
In the editable table's validation cell (anchor worst case; per-dinucleotide detail in the tooltip):

| State | Display |
|---|---|
| emitted (fill ≥ ⅓) and pose passes | green "78 %" |
| 0 < fill < ⅓ | amber "22 % < ⅓: best-fit cell not emitted" |
| fill = 0 | red "0 %: best-fit cell outside filter" |
| pose fails (fill > 0) | amber "pose fails F (score x < cutoff y)" |
| backend score ≠ frontend score (> 1e-6) | red bug banner |
| running or stale (checksum ≠ current table) | grey "validating…" |

**Blocking:**
- Export is blocked only while validation is running, stale or errored.
- Red and amber are warnings: the best-fit is a proxy, and neighbouring cells may still give near-native poses. Export lists them and asks for one confirmation.

## 8. Risks
- The cost figures in §7.4 are estimates.
- Unverified: whether Seamless cancels superseded validator runs.
- Memory peaks with `rrm95*` filters.
- Two RRM-95 prefilter variants are fallbacks below 95 % recall.
- Stage 3 and stage 6 descriptors differ by up to 0.7° until stage 3 adopts Kabsch.
- Only bound proteins are used.
- Incomplete rings give "ring incomplete" rows.

## 9. Open questions for the user
1. **Granularity:** one class per stacking anchor (recommended), or one per anchor action (fragment × first/second)?
2. **Red classes (below RRM-95):** export with a warning (recommended), block export, or prompt deselection?
3. **Validation flags:** should 0 % fill, fill < ⅓ and a pose fail stay warnings with one export confirmation (recommended), or should any of them block export?
4. **Validate all 12:** should validation run for all 12 filters per anchor once (about 20–40 s, re-run only on refit)? Edits would then be instant, and the UI could suggest the strictest filter that still emits the best-fit cell. The alternative is validating only the chosen filter, as currently specified.
5. **Stage 3 alignment:** should stage 3 also switch to Kabsch normals, so that its slip (φ) and tilt match stage 6?
