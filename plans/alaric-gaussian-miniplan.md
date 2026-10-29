# Alaric: Gaussian stacking filter, Python precalculation (mini-plan)

Plan only; nothing is implemented. Inputs: `gaussian-filter-sources.md`, `envelope-analysis-5-report.md`, branch `gaussian-filter-precalculation`, `alaric/anchor.py`. The numbers below were re-checked in scratch with the alaric env (Py 3.12, numpy 1.26).

## 1. What exists, and what is precalculated
- **Runtime consumer exists.** `anchor` with `precomputed_filter_name/_threshold` (`schema.py:18`, `resolve.py:167,256`, `backend.py:183`) runs `anchor.py --precalculated-anchor DIR`. It maps library rotamers onto Hopf cells via `ring-refe/refe-{A,C,<protein>}.npy` and emits poses from the CSR cache. `precomputed-filters/gaussian/{r90,r95,r99}` holds `rotations`, `translations`, `filtered-sizes`, `filtered-concat` (`.npy.zst`). **These are identical to the branch's `rotations/translations/A/B`** (exact match, so they come from the old Julia model). The task is therefore the **generator**. Runtime code changes only where noted.
- **Model:** score(d, φ, θ) = exp(−½ Δᵀ Σ⁻¹ Δ) / ((2π)^{3/2} √det Σ), with Δ = (d, φ, θ) − μ. Units are Å, rad and rad (normalised density, can exceed 1).
  - d = |v|, where v is the RNA ring centroid minus the protein ring centroid, in the protein-ring frame (z = protein ring normal).
  - φ (displacement) = acos(|v_z|/d), the angle of v from the protein **normal**.
  - θ (tilt/planar) = acos(|n_prot·n_rna|) = acos(|R₃₃|).
- **Grid (as in the branch):** the Hopf rotation grid, 6×16×16×64 = 98,304 cells at cell midpoints. Translation lattice x, y, z ∈ [−5, 5] in 0.25 Å steps (origin dropped). Rotations are kept by the profile bound on θ, translations by the profile bound on (d, φ), and pairs by the full score, giving the CSR layout (`filtered-sizes`, `filtered-concat`).
- **New: geometry masks**, all strict `<`, evaluated exactly on the lattice:
  - envelope d < 5.7 (d² < 32.49) and φ < 60° (3z² > x²+y²);
  - displacement < 45° (z² > x²+y²). Ties really occur on the 0.25 lattice, e.g. (0.75, 1.0, 1.25), so `arccos` must not be used for this test;
  - tilt < 30° on rotations (4R₃₃² > 3).
  - The ±5 Å box is sufficient: at every selected cutoff, max d of the kept translations is 4.94 Å and max |coord| is 4.5.
- **Offline vs runtime.** The cache depends only on (model, threshold, masks, grid constants), never on the project. The 12 caches are therefore **precalculated offline, deterministically**. Per project, the runtime only selects a threshold name, and the existing anchor code applies the cache. Preparic computes stage-3 scores client-side (§5).

## 2. Authoritative source
**Recommendation: the report.** Its cutoffs are in the units of `gaussian_model.json` (the recovered model, which reproduces every CSV score to ≤1e-13). That JSON is the model.
- From the branch, take only the **algorithm** (grid, profile prefilter, CSR format).
- The branch's model (μ = 3.871, …) and its r90/r95/r99 cutoffs are superseded. Keep them only as a frozen `legacy` entry, so that `precomputed-filters/gaussian/*` and the `tests/1b7f-gaussian-anchor*` fixtures stay valid.
- **Validation target:** recompute the cutoffs from the CSVs. I checked this in scratch: the envelope has 460 positives and 18,592 negatives, and `required = ceil(N·460/100)`. The cutoff is the `required`-th largest filtered positive score, and acceptance is `>=`. This reproduces the report bit-for-bit, plus one full-precision value the report truncates:

| RRM | filter | cutoff (density) | pos | neg | rots | trans | pairs |
|---|---|---|---|---|---|---|---|
| 90 | none | 0.27592087772968793 | 414 | 578 | 23,000 | 5,450 | 81.0 M |
| 90 | tilt30 | 0.2288694058667406 | 414 | 438 | 12,736 | 5,674 | 62.0 M |
| 90 | disp45 | 0.27007176031370117 | 414 | 541 | 23,120 | 4,714 | 73.6 M |
| 90 | tilt30+disp45 | 0.16444450868131616 | 414 | 432 | 12,736 | 5,162 | 58.6 M |
| 95 | none | 0.06141634850710175 | 437 | 1,305 | 30,560 | 8,452 | 173.5 M |
| 95 | disp45 | 0.02933111268279874 | 437 | 1,204 | 34,352 | 6,510 | 157.3 M |
| 95 | tilt30 | *fallback* 0.06141634850710175 | 425 | 718 | 12,736 | 8,452 | 93.4 M |
| 95 | tilt30+disp45 | *fallback* 0.06141634850710175 | 423 | 534 | 12,736 | 6,012 | 69.1 M |
| 50 | none | 3.303844018095503 | 230 | 31 | 8,288 | 728 | 4.1 M |
| 60 | none | 2.904318239783102 | 276 | 46 | 9,408 | 1,144 | 6.1 M |
| 70 | none | 2.04198041493334 | 322 | 90 | 11,600 | 1,640 | 12.3 M |
| 80 | none | 1.1974880057674258 | 368 | 182 | 14,936 | 2,484 | 24.0 M |

The **RRM-50/60/70/80 cutoffs are Gaussian-only** (the user's amendment) and have no prefilter variants, which makes 4 + 2×4 = **12 filters**. They follow the same rule (`required` = 230/276/322/368), and their values equal the report's "RRM-derived cutoffs evaluated on the test set" table exactly. The report prints 17 significant digits, e.g. 3.3038440180955031, which is the same double. That table's X for 50–70 uses 100,000 resampled negatives (seed 20261026), so the negative counts above (on the 18,592 envelope set) differ from it. RRM-80 (182) matches.

## 3. RRM-95 unreachable cells (**flag to user**)
After tilt<30, only 432 positives remain (428 with both filters), and 437 are required.
- **Clamp to the lowest-scoring positive:** cutoff = 5.0e-6. The Gaussian then does nothing, and the cache grows to 241 M pairs. Rejected.
- **Error:** this leaves 2 of the 8 thresholds undefined. Rejected.
- **Recommended fallback:** the report's own second table, "Added filters at fixed Gaussian-only cutoffs". Use the RRM-95 Gaussian-only cutoff with the filter added: 425/460 = 92.4 % and 423/460 = 92.0 % recall.
  - In the JSON, mark these entries `cutoff_rule: "gaussian-only-fallback"` with `achieved_recall`.
  - The derivation code raises on unreachability unless the fallback is explicitly enabled. The generator prints a warning.

## 4. Python module and API (numpy only)
`alaric/gaussian_filter.py`. It imports nothing beyond numpy/json, so it is Py ≤3.12 and numpy 1.x/2.x neutral. Hand-written quaternion→matrix as in the branch; no scipy.
- `load_spec(path=None) -> GaussianSpec`: reads `$ALARIC_DIR/gaussian_stacking_filter.json` (default: next to the module).
- `score(d, phi_rad, theta_rad, model) -> ndarray`
- `descriptors(rotation, translation) -> (d, phi, theta)`
- `hopf_rotations() -> (R[98304,3,3], theta)`
- `lattice_translations() -> (t, d, phi)`
- `precalculate(spec, name) -> dict[str, ndarray]`: the four arrays. The pair stage iterates over rotation blocks, because a dense RRM-95 matrix would be about 2 GB. A depends only on θ, so it is computed per unique θ.
- `write_precalculation(arrays, outdir)`: zstd `.npy.zst`. `filtered-concat` is stored as int32 (max index 8,451).
- `derive_cutoffs(positives, negatives, spec) -> dict`: the report rule, used by tests and to regenerate the JSON.
- CLI: `alaric/util/gaussian-precalc.py NAME --output DIR` and `--derive-cutoffs CSV...`.

The legacy r90 run from this port matched the shipped arrays (4.3 s): translations and A are exact, and rotations agree to 1.7e-15. That pins the algorithm.

## 5. Data file and preparic interface (stable contract)
Stable path: `$ALARIC_DIR/gaussian_stacking_filter.json`. `ALARIC_DIR` keeps its existing meaning, the `alaric/` package subfolder (INSTALLATION.md). Relative to the repo root (`$ALARIC_DIR/..`, which is what preparic derives), the path is `alaric/gaussian_stacking_filter.json`. The arrays live at `<repo root>/precomputed-filters/gaussian-rrm/<name>/`. Plain JSON, `schema: 1`, `version: "2026.1"`:
- `model`: `descriptor_order` `["distance_A","displacement_rad","tilt_rad"]`, the definitions as text (§1), `mean`, `covariance`, `precision` (Σ⁻¹, precomputed), and `normalizer` = 0.17639457343866755. Also `formula`: `score = exp(-0.5*Δ·P·Δ)/normalizer`, `Δ = x − mean`, angles in radians, plus the source SHA-256 of `gaussian_model.json`.
- `envelope`: `{distance_lt_A: 5.7, displacement_lt_deg: 60}`.
- `filters`: `{tilt30: {descriptor: tilt, lt_deg: 30}, disp45: {descriptor: displacement, lt_deg: 45}}`.
- `thresholds`: 12 entries:
  - `rrm50`, `rrm60`, `rrm70`, `rrm80` (Gaussian-only);
  - `rrm90`, `rrm90-tilt30`, `rrm90-disp45`, `rrm90-tilt30-disp45`;
  - the same four for `rrm95`.

  Name grammar: `rrm<N>[-tilt30][-disp45]`. Filter suffixes are valid only for N ∈ {90, 95}. Each has `cutoff` (repr float), `accept: ">="`, `filters`, `cutoff_rule`, `rrm_target`, `achieved_recall`, the pos/neg counts, and `precalc`. For each of the four files, `precalc` holds `sha256` of the `.npy.zst` bytes on disk (what preparic verifies) and `npy_sha256` of the decompressed array (stable across zstd versions).

  Preparic mounts this JSON read-only in its backend `Context`, and needs only plain JSON for that. Its `preparic` env never imports alaric. No ring data goes in the JSON: preparic reads `ring-refe/`.
- `legacy`: the branch model, r90/r95/r99, and the identity of the shipped dirs.
- `golden`: input (d, φ°, θ°) → score, at rel tol 1e-12 (JS and Python agree on IEEE doubles):
  - (3.778988652846167, 19.246627257798725, 12.439090799908035) → 5.669108638127693 (the mode)
  - (3.5, 0, 0) → 1.5350965274736001
  - (4.0, 30, 20) → 3.7720844966258613
  - (4.5, 45, 30) → 0.4299308606304266
  - (5.5, 59, 60) → 1.0036849461953622e-05
  - Three `descriptors_test.csv` rows, e.g. (4.213033715048206, 24.24718615414934, 3.9380656228423447) → 0.8340775434593926 (the CSV value is 0.8340775434593833, from Julia)
  - Plus pass/fail vectors at the 45°/30° boundaries.
  - For each of the 12 thresholds, one accepted and one rejected input on either side of the cutoff, e.g. scores just above and below 3.303844018095503 for `rrm50`.

**Descriptor match with stage 3 (`preparic/detect_stacking.py`):**
- `distance` and `tilt_angle` match d and θ (in degrees).
- **`slip_angle` is not φ.** It is asin(|n·v|/d), the angle from the ring **plane**, so **φ = 90° − slip_angle**. Then disp<45° ⇔ slip>45°, and envelope φ<60° ⇔ slip>30°.
- Resolved by `preparic-gaussian-miniplan.md` §2: stage 6 computes φ directly, after a Kabsch fit onto `ring-refe`, as training and `anchor.py` do. Remaining caveat: stage 3's default selection uses `<=`, while the Gaussian filters are strict.

## 6. Plugging into Alaric
No new action. The `anchor` config stays as it is:
```yaml
action: anchor
precomputed_filter_name: gaussian-rrm   # new set; `gaussian` = legacy r90/r95/r99
precomputed_filter_threshold: rrm95-disp45   # one of the 12 names in the JSON (rrm50..rrm80 have no suffixes)
```
Changes needed:
- `anchor.py`: `PRECALCULATED_THRESHOLDS` and the argparse `choices` are hard-coded to r90/r95/r99. They become name-validated against the JSON plus legacy. Keep `filtered-concat` as int32 instead of `astype(int64)` (RRM-95 would otherwise cost 1.4 GB per process).
  - Both changes land in `alaric/anchor_precalc.py` after PrepaGaussian's pure-move refactor (loader, validation, cell lookup, capacity). Do them as a separate commit after that refactor's bit-identical parity tests pass.
  - `seamlessify_gaussian_validate` (preparic plan §7.2) consumes `gaussian_filter.descriptors`/`score` and the JSON from this plan. It reads the caches locally and runs in conda env `alaric`.
- `resolve.py`: validate the name, and put the cache checksum into the params so it feeds the sigil (today only the path enters).
- Remote: the existing `$ALARIC_REMOTE_ALARIC_DIR/../precomputed-filters/<name>/<thr>` path.
- **Relation to the stacking anchors:** each anchor action stacks one nucleotide on one protein residue. The threshold sets how tolerant the stack geometry is. Preparic picks the name per anchor (§8, Q3).

## 7. Seamless and shipping
Following f8372ae/685b6d0:
- `util/gaussian_precalc_seamless_support.py`: `input_paths()` returns the module, CLI and JSON.
- `util/seamlessify_gaussian_precalc.py`: a Transformer factory. It pins the inputs with `files_to_checksums` and takes `name` as a str argument; the result is the 4-array directory.
- `util/gaussian-precalc-SEAMLESS.sh NAME --output DIR`.
- **Ship:** the JSON (with checksums) goes in git. The arrays do **not** go in git. Total is about 814 M pairs, or about 350 MB zst for all 12, scaled from the legacy ratio of 35 MB per 82.6 M pairs. RRM-50..80 add 46.5 M pairs (about 20 MB, +6 %), and each generates in seconds. `alaric-gaussian-materialize [NAME…]` fills `precomputed-filters/gaussian-rrm/<name>/` from the Seamless cache, or recomputes (seconds to minutes, single core), and verifies the checksums.
- It copies the exact cached `.zst` buffers, so `sha256` holds. A recompute is checked by `npy_sha256` and then re-pinned.
- Preparic and the Alaric install share one machine and read the files in place. Nothing is staged, and remote workers are out of scope.

## 8. Tests and steps
1. `alaric/gaussian_filter.py`, with `tests/gaussian/test_model.py`: the golden vectors, and agreement with the shipped legacy r90/r95/r99 (exact translations/A/B, rotations to 1e-12).
2. Vendor the envelope descriptors (460 pos + 18,592 neg, with source SHA-256) as `tests/gaussian/envelope-descriptors.csv.zst`. `test_cutoffs.py` reproduces all 12 cutoffs in §2 bit-for-bit, including RRM-50..80 against the report's test-set table. It also checks the report's second table and the pos/neg counts on the 18,592 envelope set (RRM-80: 182).
3. Generate `alaric/gaussian_stacking_filter.json` with the CLI. Commit the generator command in its `provenance`.
4. Generate the 12 caches, and test the CSR invariants and the masks. A brute-force check of 10⁵ random (rotation, translation) membership samples against `score`.
5. Seamless support, factory and `.sh`, with `tests/test_seamlessify_gaussian_precalc.py`.
6. The `anchor.py`, `resolve.py` and `schema.py` changes. Add a `tests/1b7f-gaussian-rrm-anchor` yaml, and a golden pose count on frag6 for `rrm90`.
7. `documents/gaussian-filter.md`. Hand the JSON contract to preparic.

## 9. Open questions
1. Is the RRM-95 tilt30/both fallback (fixed Gaussian-only cutoff + filter, 92.4 %/92.0 %) acceptable, or should all RRM-90/95 variants use the fixed-cutoff reading?
2. Should the envelope (d<5.7, φ<60) be applied as a hard mask? It is non-binding for d at all 12 cutoffs (max d 4.94).
3. Who chooses the threshold per anchor: a preparic default (e.g. `rrm95`) or the user?
4. (Resolved on the preparic side: φ is computed directly, and stage 3's `slip_angle` is only relabelled.) Should `rrm95-tilt30*` be hidden from the UI, since they are fallbacks below 95 % recall?
5. Distribution: Seamless materialisation (recommended), or commit about 350 MB of arrays like the legacy set?
