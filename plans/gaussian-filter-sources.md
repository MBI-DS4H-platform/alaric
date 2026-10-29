# Gaussian filter: source description

Sources: `~/shared/stacking-prediction-plots` (same directory as `/srv/shared/stacking-prediction-plots`, via `readlink -f`) and `~/shared/stacking-interactions-classifying` (branch `gaussian-filter-precalculation`). Nothing was checked out, modified or run in either repository. Branch `.npy.zst` files were extracted to the scratchpad and read only for headers and statistics.

## Part 1. `*gauss*` in stacking-prediction-plots

Top-level glob matches (9): one CSV, eight PNGs, one directory. Naming and sizes in bytes.

### `descriptors_synthetic_negatives_gaussian.csv` (7,367,214)
- CSV, 43,400 rows. Columns: `distance_A, displacement_angle_deg, planar_angle_deg, gaussian_score, rf_score, adaboost_score, logreg_score, naive4_score, gaussian_score_python`.
- All original columns match field-for-field `descriptors_synthetic_negatives.csv-BAK` (the original sample, SHA-256 `90e6ff01...`), plus the added `gaussian_score_python` column. The latter is the Python recomputation (`recover-gaussian/gaussian_score.py --output`).
- Ranges: distance 0.088-5.0 Å, displacement 0.6-90 deg, planar 0.05-30 deg. `gaussian_score` is 1.95e-61 to 5.64. Other score columns are populated here, but are empty in the regenerated `descriptors_synthetic_negatives.csv`.
- Units: Å and degrees. Score is the normalized 3-D Gaussian density in (Å, rad, rad) and can exceed 1. Python agrees to relative error <= 9.96e-14.

### Eight `envelope-analysis-*-gaussian*.png` (matplotlib, 300 dpi)
All are plots of `gaussian_score` for RRM positives (`label` train+validation in `descriptors_rrm.csv`; 1090 rows in total) against the synthetic negatives. Filters `<` are strict. The generating scripts are in the same directory.

| File (bytes) | Script | Content |
|---|---|---|
| `3a-gaussian` (135,466), `3b-gaussian` (133,971) | `envelope-analysis-3.py` | Cumulative fraction within class vs Gaussian-score percentile among synthetic negatives (high to low). Classes: positives vs `duplicate` negatives. 3a: distance <5 Å (453 pos, 2 neg). 3b: 5-6 Å (9 pos, 7 neg). Displacement <60 deg throughout. |
| `4a`..`4c-gaussian` (148,099 / 148,462 / 148,094) | `envelope-analysis-4.py` | Same percentile axis, log y: positives 1-CDF, negatives CDF. Distance <6 / <5.7 / <5 Å, displacement <60 deg. Positives n=462 / 460 / 453; duplicates n=9 / 2 / 2. Reference is the filtered synthetic CSV (21,682 / 18,592 / 12,551 rows). |
| `4d-gaussian` (152,308) | `envelope-analysis-4.py` | Same, with the old `-BAK` reference. Distance <5 Å, tilt <30 deg, displacement <60 deg (431 pos, 1 neg, 21,604 ref). |
| `5-gaussian-roc` (257,320), `5-gaussian-roc-test-set` (216,353) | `envelope-analysis-5.py` | Threshold-sweep ROC. X = % of 18,592 envelope synthetic negatives accepted, Y = % of positives accepted (460 RRM, or 58 test). Five curves: Gaussian only, +distance <5, +tilt <30, +displacement <45, +both. Insets: x 0-2% and 0-10%. |

Reports `envelope-analysis-4{a,b,c,d}-report.md` accompany the 4-series (they include KS checks of the generator marginals).

### `recover-gaussian/` (directory, 72 KB; 11 files + `__pycache__/`)
- `gaussian-score.md` (4,939): narrative. The Julia Gaussian density is ported and reproduces all 43,400 original scores (max rel. error 9.97e-14). The original Julia parameter CSV `datasets/generated/stacking_gaussian.csv` is absent, and the older hardcoded parameters in git history do not match. Parameters were therefore recovered by quadratic least squares on log-density using the first 1,000 rows, and verified on the other 42,400.
- `gaussian_model.json` (896): `feature_order` (distance_A, displacement_angle_rad, planar_angle_rad). `mean` = [3.778988653, 0.335917016, 0.217103090]. `covariance` = [[0.134026434, 0.048747465, 0.036616721], [0.048747465, 0.050512458, 0.018930689], [0.036616721, 0.018930689, 0.039514510]]. Plus a provenance block. sqrt(det) = 0.0112, so the normalizer (2pi)^1.5 sqrt(det) = 0.176395. The same model reproduces the `gaussian_score` of `descriptors_rrm.csv` (1090 rows) and `descriptors_test.csv` (58 rows) to <= 4e-13 (my check).
- `gaussian_score.py` (6,573): `GaussianModel` (Cholesky, `score(distance_A, disp_deg, planar_deg)`), CLI validator. `recover_gaussian_model.py` (3,899): the recovery. `test_gaussian_score.py` (3,107): unittests.
- `gaussian-thresholds.py` (2,731) with outputs `gaussian-thresholds.txt` / `-OLD.txt` (270 each) and `-with-test.txt` (497). Table of RRM recall 90/95/99/100% against Gaussian-score cutoff 0.8300 / 0.3414 / 0.04928 / 0.006515 (density units). Synthetic-negative acceptance is 7.5/12.9/25.2/37.6% (OLD 7.7/13.4/25.3/37.9%). Test-set recall is 94.8/98.2/100/100%. The script's active filter is displacement <50 deg and distance <5.5 Å (the planar <30, distance <5 filter is commented out). Its default input `descriptors_rrm_positives.csv` is absent. The .txt files predate the script.
- `gaussian-score-{original,new}-validation.txt` (305 / 335): validator output, 43,400/43,400 match.
- Related, outside the glob: `generate_synthetic_negatives.py` (seed 20261024; distance = 6 U^(1/3), both angles = arccos(1-U)). Other matches: `OLD/duplicate-threshold-gaussian*.png` (3), `__pycache__/*gaussian*.pyc` (3).

### `envelope-analysis-5-report.md` (5,997) and `...-test-set-report.md` (2,610)
Both are written by `envelope-analysis-5.py`. The git working tree there shows the report and script as modified and uncommitted. I recomputed the cutoffs from the CSVs and they match.

- **Envelope** (`in_envelope`, script line 29): distance <5.7 Å and displacement <60 deg. This gives 460 RRM positives (label train or validation) and 18,592 of 43,400 synthetic negatives. Test set: 58, all in the envelope.
- **Filters** (`add_filter`, line 34): tilt <30 means `planar_angle_deg < 30`. Displacement <45 means `displacement_angle_deg < 45`. Distance <5 Å is a fifth, extra variant. The fixed denominators (460 / 18,592) are kept for all curves.
- **RRM-N cutoff** (lines 158-165, 210-212, 239-240): `required = ceil(N*460/100)` = 414 (N=90) or 437 (N=95). The cutoff is the `required`-th largest `gaussian_score` among the filtered positives. Contacts are accepted if score >= cutoff (ties are included). The cutoff is therefore an observed score, in normalized density units of the recovered model.
- **Two readings of "filtered" cutoffs.** Table "Fixed positive recovery targets" re-derives the cutoff under each filter. Table "Added filters at fixed Gaussian-only cutoffs" keeps the Gaussian-only cutoff and adds the filters.

| Filter | RRM-90 re-derived | RRM-95 re-derived |
|---|---|---|
| Gaussian only | 0.27592087772968793 | 0.061416348507101752 |
| tilt <30 | 0.228869406 (432 pos max) | unreachable (432 < 437) |
| displacement <45 | 0.27007176 | 0.0293311127 |
| both | 0.164444509 | unreachable (428 < 437) |

- With the fixed Gaussian-only cutoffs, the eight combinations are all defined. At RRM-90 (cutoff 0.27592): negatives 578 / 414 / 533 / 375 and RRM positives 414 / 411 / 413 / 410 (unfiltered / tilt / disp / both). At RRM-95 (cutoff 0.06142): negatives 1,305 / 718 / 972 / 534 and positives 437 / 425 / 432 / 423.

## Part 2. Branch `gaussian-filter-precalculation`

**Branch point.** Branches present: `main`, `gaussian-filter-precalculation`, `paper/paper-text` (checked out), and others. There is no `master`. Local `main` (0ad49d2) and `origin/main` (1b5623d) differ.

- `git merge-base main gaussian-filter-precalculation` = `0ad49d2`. Local `main` already holds the first two commits below, so only the last is "unique" against it.
- `git merge-base origin/main gaussian-filter-precalculation` = `78899a8` (2026-07-02). Against `origin/main`, the branch has three unique commits, all by Sjoerd de Vries on 2026-07-27. This is the full set.

| Hash | Message | Summary |
|---|---|---|
| `b54f157` | Gaussian filter precalculation: planning | Adds `plans/gaussian-precalculation-{prompt,plan,handoff-plan}.md` (719 lines): spec, design (Hopf rotation grid), verification and a snapping-vs-nearest-neighbour analysis. |
| `0ad49d2` | Gaussian filter precalculation: implementation | Edits `scripts/gaussian.jl` (+168), `notebooks/Project.toml` (adds NPZ) and `Manifest.toml`. Adds `scripts/gaussian_grid.py`, `scripts/check_grid_parity.py`, `scripts/export_gaussian_grid.jl` (3 lines). |
| `e218bc8` | Gaussian filter precalculation: dataset | Adds 12 binary files `datasets/gaussian_grid/{A,B,rotations,translations}_{r90,r95,r99}.npy.zst`. No text diff. |

### Code
- **Languages and entry points.** Julia: `julia --project=notebooks scripts/export_gaussian_grid.jl` calls `export_grids` (`gaussian.jl:224`). `gaussian.jl` run directly still performs the TPR/FPR evaluation of `interactions.arrow` (line 307). Python: `scripts/gaussian_grid.py` (writes `datasets/gaussian_grid/py/`) and `scripts/check_grid_parity.py`.
- **Model** (`gaussian.jl:38-54`): 3-D Gaussian over (d in Å, phi in rad, theta in rad). Hardcoded `STACKING_MEAN` = (3.87095, 0.418246, 0.273519) and `STACKING_COV` = [[0.0940985, 0.0407767, 0.0290554], [., 0.0480912, 0.0176934], [., ., 0.0356662]]. There is no sigma parameter; the shape comes entirely from this covariance. `classify` (l.17) is the unnormalized exp(-Δ'Σ⁻¹Δ/2), with a maximum of 1. Normalizer `_STACKING_NORM` = 0.13545 (l.48). Thresholds are density values times the normalizer (l.52-54): `THRESHOLD_90` = 1.30 × norm = 0.1761, `_95` = 0.88 × norm = 0.1192, `_99` = 0.38 × norm = 0.0515. In the entry point, ground truth is `com_dis < 4.5 Å && plane_angle < 20°` (l.300).
- **Feature definitions** (`preprocess_gaussian`, l.65-70): d = |pos|, phi = acos(|pos_z/d|) (displacement angle), theta = acos(|R33|) (tilt, planar angle). Both angles lie in [0, pi/2].
- **`prefilter`** (l.26-33) is the marginal Gaussian over a subset of dimensions, an upper bound of `classify`. Rotation depends on theta only (`ids_rot=(3,)`), translation on (d, phi) only (`ids_pos=(1,2)`).
- **Algorithm** (per threshold r90 / r95 / r99):
  1. `sample_rotations` (l.145): Hopf grid with 6 cube faces × 16 × 16 equiangular cells (midpoints, `tan` spacing) × 64 fiber angles psi = 98,304 rotations. Loop order face, p, q, m. Quaternion at l.115-129 is converted by a hand-written `quat_to_matrix` (l.103). Keeps rotations with `prefilter(theta) >= threshold`. Output: (N,3,3) matrices and theta.
  2. `sample_translations` (l.165): x, y, z from -5 to 5 Å in steps of 0.25 (41³ = 68,921). The origin gives NaN and is rejected. Keeps points with `prefilter(d, phi) >= threshold`.
  3. `pair_indices` (l.186): for every kept rotation i and kept translation j, evaluates the full `classify(d_j, phi_j, theta_i) >= threshold`. `A[i]` is the count of passing j. `B` is the concatenation of passing j as 0-based indices into the saved translation array, so `cumsum(A)` gives offsets (CSR layout).
- **Python** (`gaussian_grid.py`): reimplements steps 1-2 (l.92-141) with the same constants and loop order, and not `A`/`B`. `check_grid_parity.py` asserts Julia and Python rotations and translations agree to `atol=1e-12`, checks CSR consistency (int64, sum(A)==len(B), 0<=B<M) and a 3×3 sample of rotation-translation memberships. I did not run it.
- **Dependencies.** Julia: StaticArrays, LinearAlgebra, Arrow, DataFrames, DataFrameMacros, Chain, Rotations, NPZ (new). Python: numpy.
- **Formats.** Reads `datasets/interactions.arrow` only in the evaluation entry point. Writes `.npy` via `npzwrite`. The arrays have `fortran_order: True` in the header (checked on `rotations_r90`; numpy loads (N,3,3) correctly). Committed files are zstd-compressed `.npy.zst`.

### Dataset contents (decompressed, checked)

| | rotations | translations | A | B |
|---|---|---|---|---|
| r90 | (18104,3,3) f8 | (2880,3) f8 | (18104,) i8, 0-2632, 56 zero rows | (33,246,528,) i8, max 2879 |
| r95 | (20416,3,3) | (3762,3) | 0-3384, 128 zeros | 47,110,912, max 3761 |
| r99 | (24872,3,3) | (5094,3) | 0-4720, 208 zeros | 82,574,416, max 5093 |

Uncompressed `B_r99.npy` is 660,595,408 bytes (`.zst` 35 MB; total about 73 MB). Rotation matrices are proper (det = 1; first 2000 rows checked). Kept translations have |coordinate| <= 4.0 Å and d of about 3.25-4.61 Å. Kept theta is up to 0.625 / 0.663 / 0.733 rad (35.8 / 38.0 / 42.0 deg).

### Relation to the part 1 files
- Same quantities: Julia (d, phi, theta) map to CSV `distance_A`, `displacement_angle_deg`, `planar_angle_deg` (in radians). The classify function is the Julia Gaussian that `gaussian-score.md` ports. The CSV `gaussian_score` equals `classify / _STACKING_NORM`.
- Different parameters: the branch hardcodes the older fit (mean 3.871 Å, 24.0 deg, 15.7 deg) with density cutoffs 1.30 / 0.88 / 0.38. `gaussian_model.json` is the recovered newer model (mean 3.779 Å, 19.25 deg, 12.44 deg), and `gaussian-score.md` states that the older hardcoded values do not match the CSV. The recovered mean is close to that of the d<5 Å, volume-weighted refit in commit `423b875` (origin/fix/gaussian-fit; message gives mean 3.77 Å, 19.3 deg, 12.6 deg), whose `gaussian.jl` loads its Gaussian and thresholds from `datasets/generated/stacking_gaussian*.csv` (the absent file).
- Different thresholds: the branch has r90/r95/r99 only, with no tilt or displacement filters, no distance <5.7 / displacement <60 envelope, and no RRM-set-derived cutoffs. The part 1 cutoffs (0.83 / 0.34 / 0.049 / 0.0065, and 0.27592 / 0.06142 in report 5) are in normalized density units of the recovered model, not of the branch's model.
- Filter axes: tilt <30 acts on theta (the rotation axis), and displacement <45 acts on phi (the translation axis), so each filter touches one side of the rotation/translation split in the branch's algorithm. This is stated as a relationship; the branch does not implement them.
