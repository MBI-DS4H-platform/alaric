# Alaric Gaussian filter precomputation: implementation plan

Deliverables **A7** (the committed filter specification `alaric/gaussian_stacking_filter.json`) and **A8** (an offline generator plus gitignored CSR caches under `<repo root>/precomputed-filters/gaussian-rrm/<name>/`). This document is self-contained. Sources are cited only for provenance.

Precedence of sources:
1. the author's corrections of 2026-10-29: hard mask d < 5 Å; cutoffs rounded down to five decimals; no cutoff re-derivation;
2. `plans/oracle-contract.md` §3.7–3.8 and G3/G4/G8/G9;
3. `plans/contract-boundaries.md` §1, §5.1 and §6;
4. `plans/open-questions.md`;
5. `plans/draft-implementation-plan.md` §8;
6. `plans/alaric-gaussian-miniplan.md`, which is outdated wherever it conflicts with the above.

## 1. Fixed facts and rules

**Filter names (12).** `rrm50`, `rrm60`, `rrm70`, `rrm80`, plus `rrmN`, `rrmN-tilt30`, `rrmN-disp45` and `rrmN-tilt30-disp45` for N ∈ {90, 95}. The grammar is `rrm<N>[-tilt30][-disp45]`. Suffixes are allowed only for N ∈ {90, 95}.

**Model.** The source is `/srv/shared/stacking-prediction-plots/recover-gaussian/gaussian_model.json`, sha256 `0bfb8acf…7db8`.
- Descriptor order: (d in Å, φ in rad, θ in rad).
- mean = [3.778988652846167, 0.3359170155526752, 0.21710309041293038].
- covariance = [[0.13402643421739668, 0.048747464673082525, 0.03661672105978793], [0.048747464673082525, 0.05051245758442225, 0.018930688724029816], [0.03661672105978793, 0.018930688724029816, 0.039514509672839315]].
- P = Σ⁻¹.
- normalizer = (2π)^{3/2}·√det Σ = 0.17639457343866755 (measured).
- score = exp(−½·q)/normalizer, where Δ = x − mean and q = P00Δ0² + P11Δ1² + P22Δ2² + 2(P01Δ0Δ1 + P02Δ0Δ2 + P12Δ1Δ2).
- Degrees convert as rad = deg·(π/180).

P and the normalizer are **hard-coded literals** in the module, computed once at bootstrap. A test checks them against `np.linalg.inv` and the formula to 1e-14. This keeps LAPACK differences between environments out of the identity (G3).

**Descriptor definitions.** These are written into the JSON as text.
- v is the RNA ring centroid minus the protein ring centroid, in the protein ring's frame (z is the protein normal).
- d = |v|.
- φ = arccos(|v_z|/d), the angle from the protein normal. Stage 3's slip angle is measured from the ring plane, so φ = 90° − slip.
- θ = arccos(min(1, |R₃₃|)), the angle between the two ring normals.
- Normals come from a Kabsch fit of each ring onto the reference rings (G→A, U→C, TYR→PHE).

**Applied mask.** This is used for the caches and by every consumer, for example A9 `pose_pass` and the browser. It is d < 5 Å and φ < 60°, both strict (author's correction).
- The cutoff-derivation envelope (d < 5.7 Å, φ < 60°: 460 RRM train+validation positives, 18,592 synthetic negatives) is **provenance only**.
- To avoid misuse, it never appears as a top-level key.

**Prefilters.** `tilt30` is θ < 30° and `disp45` is φ < 45°, both strict.

**Exact tests.** Masks are never decided through `arccos`.
- Lattice point t = (i, j, k)·0.25 Å with integers i, j, k ∈ [−20, 20]:
  - d < 5 ⇔ i²+j²+k² < 400;
  - φ < 60° ⇔ 3k² > i²+j²;
  - disp45 ⇔ k² > i²+j².
- Rotation: tilt30 ⇔ 4·R₃₃² > 3, evaluated in float64 on the stored matrix entry.

**Grid.**
- Rotations: the Hopf grid of 6 faces × 16 × 16 × 64 = 98,304 cells, at cell midpoints. The cell index is ((face·16 + p)·16 + q)·64 + m, with m varying fastest.
  - a = tan(π/4·(2(p+½)/16 − 1)) and b likewise with q.
  - The face direction s is (±1, a, b), (a, ±1, b) or (a, b, ±1) for faces 0/1, 2/3, 4/5; even faces take +, then s is normalised.
  - θs = arccos(s_z), φs = atan2(s_y, s_x), ψ = 2π(m+½)/64.
  - q = (cos(θs/2)cos(ψ/2), cos(θs/2)sin(ψ/2), sin(θs/2)cos(φs+ψ/2), sin(θs/2)sin(φs+ψ/2)) as (w, x, y, z).
  - R is the standard hand-written quaternion-to-matrix, R[row, col] (branch `gaussian.jl:103-129`, `anchor.py:473-508`).
- Translations: the 41³ lattice over ±5 Å in 0.25 Å steps, in C order (x slowest, z fastest), with the origin excluded.

**Cutoffs.** Each value is the report's full-precision value, **rounded down** to the largest multiple of 1e-5 that is ≤ raw. Variants use exactly their parent's rounded cutoff.

| name(s) | report row (`envelope-analysis-5-report.md`) | raw cutoff | cutoff |
|---|---|---|---|
| rrm50 | l.42 | 3.3038440180955031 | 3.30384 |
| rrm60 | l.43 | 2.9043182397831022 | 2.90431 |
| rrm70 | l.44 | 2.0419804149333398 | 2.04198 |
| rrm80 | l.45 | 1.1974880057674258 | 1.19748 |
| rrm90 (+3 variants) | l.46 (and l.59) | 0.27592087772968793 | 0.27592 |
| rrm95 (+3 variants) | l.51 (and l.63) | 0.061416348507101752 | 0.06141 |

- The report has sha256 `11f854aa…2edc`. Its working tree is modified and uncommitted in that repository, so a 6 KB copy is vendored (§3).
- **Rounding happens in one place**, `gaussian_filter.floor_cutoff(raw, decimals=5) -> str`:
  - take the shortest round-trip decimal `repr(raw)`, the same convention G7 uses for energies;
  - apply `Decimal.quantize(1e-5, ROUND_FLOOR)`;
  - assert `float(s) <= raw`.
- This is the mirror image of the RMSD/energy rule (smallest multiple strictly above raw, for strict `<`).
- The re-derived variant cutoffs (0.229, 0.270, 0.164, 0.029) and any "fallback" MUST NOT appear in code, JSON or tests.

**Acceptance.** A (rotation cell, translation) pair is kept iff all four hold:
- `score(d, φ, θ) >= cutoff`, with score computed by `gaussian_filter.score`;
- the applied mask holds;
- the name's prefilters hold;
- the translation is not the origin.

**Containment.** Filter A ⊆ filter B iff A's masks include B's masks and cutoff_A ≥ cutoff_B. This gives:
- rrm50 ⊂ rrm60 ⊂ rrm70 ⊂ rrm80 ⊂ rrm90 ⊂ rrm95;
- for N ∈ {90, 95}: rrmN-tilt30-disp45 ⊂ rrmN-tilt30 ⊂ rrmN, and rrmN-tilt30-disp45 ⊂ rrmN-disp45 ⊂ rrmN;
- each 90 filter ⊂ its 95 counterpart.

Rounding down is monotone, so containment is unchanged.

## 2. Verification results (measured 2026-10-29, alaric env, read-only)

**Cutoff values.**
- All six report values parse, via `float()`, to the doubles in the contract:
  - `float('3.3038440180955031') == 3.303844018095503`;
  - `float('0.061416348507101752') == 0.06141634850710175`;
  - the same holds for 60, 70 and 80.
- Rounding down gives exactly 3.30384, 2.90431, 2.04198, 1.19748, 0.27592 and 0.06141, and `float(rounded) <= raw` holds for all six.
- As a cross-check, applying the script's rule (`envelope-analysis-5.py:208-212`: required = ⌈N·460/100⌉, the cutoff is the required-th largest CSV `gaussian_score`) to the CSVs reproduces all six bit for bit. This check is not part of the plan.

**d < 5 against d < 5.7: no difference at any of the 12 filters.**
- **Report:** the "+ distance < 5 Å" rows give the same cutoff and the same counts as "Gaussian only" (l.23 for 90 %: 0.275920878, 414/460, 578; l.28 for 95 %: 0.0614163485, 437/460, 1,305; l.47 for test-set 90 %).
- **Analytic bound:** the largest score possible at any d ≥ 5 is exp(−½(5−μ_d)²/Σ_dd)/normalizer = 0.021779 < 0.06141, the smallest cutoff. No pair with d ≥ 5 can be accepted. The ±5 Å box therefore also loses nothing, and with d < 5 it covers the mask by construction.
- **Lattice computation:** pairs, kept rotations and kept translations are identical under no distance mask (box only), d < 5.7 and d < 5, at all 12 filters and for both raw and rounded cutoffs. The largest kept d is:
  - 4.146 Å (rrm50), 4.168 (60), 4.279 (70), 4.402 (80);
  - 4.677 (rrm90 family), 4.861 (rrm95 family).
  - The miniplan's "4.94" belonged to the superseded 0.029 cutoff.
- **Lattice points at exactly d = 5:**
  - (0,0,±5), (0,±3,±4) and (±3,0,±4) pass φ < 60°, but their score is ≤ 0.0218, so they are rejected by the cutoff as well as by the mask;
  - (3,4,0) and (5,0,0) already fail φ < 60°.
- **Conclusion:** the mask is a no-op. It is kept as the author's contract.

**Effect of the five-decimal rounding.**
- No dataset count changes: positives and negatives on the 460/18,592 set are identical for raw and rounded cutoffs, with and without d < 5 (CSV check).
- Cache sizes grow slightly: rrm80 +640 pairs, rrm95 +4,800, rrm95-tilt30 +1,280, rrm95-disp45 +2,816, rrm95-tilt30-disp45 +256.
- rrm50/60/70 and the rrm90 family are unchanged. No kept-rotation or kept-translation count changes.

**Determinism margins.**
- The smallest relative gap |score/cutoff − 1| over every lattice pair is 1.29e-7 (rrm95 rounded). All others are ≥ 1.2e-6.
- The smallest |4R₃₃² − 3| over the 98,304 cells is 6.06e-4.
- d² ties occur only at d = 5, which is covered above. φ = 60° ties are impossible on the lattice: i²+j² = 3k² has no nonzero integer solution.
- φ = 45° ties do occur, for example (0.75, 1.0, 1.25).
- The cache content is therefore insensitive to ULP-level differences in `exp`/`arccos` between environments.

**Legacy r90.**
- An in-memory run of this algorithm with the legacy model reproduces the legacy pair set exactly: 33,246,528 pairs, 2,880 translations with identical values and order, and Hopf cell order.
- Rotations agree to 1.75e-15.
- Legacy keeps 56 zero-size rows that the canonical layout drops (18,104 against 18,048 rows).

**Environment and legacy file format.**
- alaric env: Python 3.12.13, numpy 1.26.4, `zstandard` 0.25.0 (cext, libzstd 1.5.7); the `zstd` CLI is v1.5.7.
- `alaric/npy_io.py` writes a v1.0 header with `fortran_order: False` and C order, and compresses with `ZstdCompressor()` at default level 3, with the content size declared and no frame checksum (`npy_io.py:308-333`, `save_npy` at `:373`).
- Legacy `.npy.zst` files were written by Julia NPZ: `fortran_order: True`, `<i8` sizes and concat. They were compressed by the zstd CLI with an XXH64 frame checksum.
- The runtime reads either form (`npy_io.load_npy`). It prefers `.npy` over `.npy.zst` when both exist (`find_npy`, `npy_io.py:61-72`).

## 3. Design decisions

1. **Canonical CSR layout.** The new caches are a pure function of the accepted pair set S:
   - `rotations.npy`: float64 [R,3,3], C order. The Hopf cells with ≥ 1 accepted pair, in ascending cell index.
   - `translations.npy`: float64 [M,3]. The lattice points with ≥ 1 accepted pair, in lattice C order, with values exactly i·0.25.
   - `filtered-sizes.npy`: int32 [R].
   - `filtered-concat.npy`: int32 [Σ sizes]. Per row, ascending indices into `translations`.

   int32 is valid because M ≤ 8,452 and sizes ≤ 16,734. No zero rows are stored: a missing row and a zero row give the same count 0 in `anchor.py`. The layout is therefore independent of any acceleration prefilter, and containment is visible at array level.

   The runtime loader accepts this layout:
   - it casts to int64 (`anchor.py:448,457`);
   - it infers the lattice origin and step from the kept translations (`:562-582`);
   - it requires distinct Hopf centres within 1e-5 (`:541-559`).
2. **Algorithm without numeric prefilters.**
   - The score depends only on (|R₃₃|, i²+j², k²). There are 34,808 distinct |R₃₃| values and 2,302 distinct translation keys with φ < 60°.
   - Compute the score table S[θ-key, t-key] once per model (≈ 80 M evaluations, about 640 MB float64, computed in θ blocks). Each key's inputs are bitwise identical to every member's, so the table equals per-pair evaluation exactly.
   - Pass 1 builds the accept table (score ≥ cutoff ∧ masks). It yields per-cell sizes as accept @ per-key multiplicity, plus the kept rows and kept translations.
   - Pass 2 streams `filtered-concat` in blocks of 1,024 cells into `npy_io.NpyWriter`, whose shape is declared up front from pass 1. The ~694 MB rrm95 concat is never held whole.
   - Peak memory is about 1.2 GB. There is no branch-style profile bound, because the table makes it unnecessary.
3. **Determinism audit.** The generator records:
   - `min_rel_margin` = min |score/cutoff − 1| over the mask region;
   - `min_tilt_gap` = min |4R₃₃² − 3|.

   It fails if either is < 1e-10. A failure would mean the cache depends on ULP behaviour; the measured values are 1.29e-7 and 6.06e-4.
4. **Checksums.**
   - `npy_sha256` is the sha256 of the complete decompressed `.npy` file bytes (header plus payload). It is **authoritative**: it is the identity of the data and depends only on the array.
   - `sha256` is the sha256 of the reference run's `.npy.zst` bytes: `zstandard` 0.25.0 / libzstd 1.5.7, level 3, content size declared, 4 MiB chunks via `save_npy`. It is a fast path.
   - Verification rule (`gaussian_cache.verify_file`), in the order the loader resolves files:
     1. If `X.npy` exists, hash it. It must equal `npy_sha256`.
     2. Otherwise, if `X.npy.zst` exists and its bytes hash to `sha256`, it is OK.
     3. Otherwise, stream-decompress it and hash. If the result equals `npy_sha256`, it is OK with status `recompressed`; this is how another zstd version's output is accepted.
     4. Anything else, including a missing file, raises `GaussianCacheInstallationError`.
   - The error message names the file, both digests and the command, for example `python <root>/util/gaussian-precalc.py rrm95 --output <root>/precomputed-filters/gaussian-rrm/rrm95` (or `util/gaussian-precalc-all.py`). Nothing is regenerated silently (G9).
   - The helper returns the resolved paths, so the caller loads exactly the verified files.
   - It uses only the standard library plus `zstandard`, imported lazily for the fallback, so A9 can attach it in either environment.
   - Downstream identities (A9, the anchor sigil) should use `npy_sha256`, so that recompression never changes a cache key.
5. **Bootstrap order.** The spec constants live in the module, so caches do not depend on the JSON and there is no cycle.
   1. `gaussian-precalc-all.py` builds the 12 caches. Each folder gets a gitignored `manifest.json` with: schema 1, name, cutoff, per-file sha256/npy_sha256/shape/dtype, counts, the audit values, and the environment (python, numpy, zstandard, libzstd, level). Its digests come from re-reading the published files with the helper.
   2. `util/gaussian-filter-json.py` assembles the JSON from the module constants, the report table, computed golden vectors, the legacy digests and the 12 manifests.
   3. Commit the JSON.

   Regeneration on another machine:
   - `gaussian-precalc-all.py --check` builds into a temporary directory and compares against the JSON.
   - An `npy_sha256` mismatch is a hard failure (algorithm or environment drift), and the JSON is never overwritten.
   - A `sha256`-only mismatch is reported as "recompressed, arrays identical" and exits 0.

   A deliberate change of model, cutoff, mask or grid bumps `filter_set_version` and re-runs steps 1–3.
6. **Spec digest.** `spec_digest` is the sha256 of the canonical JSON of {model, mask, filters, grid, names with cutoffs}. The test holds a table {filter_set_version: spec_digest}, so a semantic edit without a version bump fails.
7. **Counts come from the report, not from recomputation.** §2 shows that rounding changes no dataset count.
   - **Variants and rrm90/95:** the "Added filters at fixed Gaussian-only cutoffs" table, l.59–66.
     - RRM positives: 414/411/413/410 and 437/425/432/423.
     - Negatives on the 18,592 set: 578/414/533/375 and 1,305/718/972/534 (order: none/tilt/disp/both).
   - **rrm50–80:** the report's l.42–45. That table prints the cutoff, X and test Y, but not RRM Y.
     - Positives are the rule's required count 230/276/322/368. The CSV check confirmed exactly that many, with no ties.
     - Negatives are copied with their own denominator and set: 183, 246 and 493 of 100,000 resampled envelope negatives (seed 20261026) for 50/60/70, and 182/18,592 for 80.
   - **Test Y:** recorded only for Gaussian-only rows: 27, 34, 46, 52, 57 and 58 of 58.
   - **Never** copy l.47–50 or l.22–30: they hold the re-derived variant cutoffs.
   - `achieved_recall` = positives/460, a float.
   - Cache counts (rotations, translations, pairs) come from the manifests.
8. **Vendored data.** Only the 6 KB report goes into `tests/gaussian-data/envelope-analysis-5-report.md`, checked against its sha256. No descriptor dataset is vendored and there is no derivation module.

## 4. JSON layout (`alaric/gaussian_stacking_filter.json`)

The JSON is written with `json.dumps(indent=1, ensure_ascii=True)`, fixed key order, float `repr`, a trailing newline and no timestamps. Regenerating from the same inputs is byte-identical.

```
{schema:1, filter_set_version:"2026.1", spec_digest,
 model:{descriptor_order:["distance_A","displacement_rad","tilt_rad"], definitions:{…text §1},
        mean, covariance, precision, normalizer, formula, quadratic_form, degrees_to_radians:"deg*(pi/180)",
        source:{path, sha256}},
 mask:{applies_to:"caches, pose_pass, browser", distance_lt_A:5, displacement_lt_deg:60,
       exact:{distance:"x²+y²+z² < 25", displacement:"3z² > x²+y²"}},
 filters:{tilt30:{descriptor:"tilt", lt_deg:30, exact:"4·R33² > 3"},
          disp45:{descriptor:"displacement", lt_deg:45, exact:"z² > x²+y²"}},
 grid:{hopf:{faces:6,n:16,npsi:64,cells:98304,index:"((face*16+p)*16+q)*64+m"},
       lattice:{min:-5,max:5,step:0.25,order:"C (z fastest)",origin:"excluded"},
       layout:"canonical: rows/translations with ≥1 accepted pair; int32 sizes/concat"},
 thresholds:{"rrm95-disp45":{rrm_target:95, parent:"rrm95", filters:["disp45"], accept:">=",
       cutoff:0.06141, cutoff_decimal:"0.06141", raw_cutoff:0.06141634850710175,
       cutoff_rule:"parent Gaussian-only RRM-N cutoff of the report, floored to 1e-5; prefilters on top",
       source:{report_line:65, table:"Added filters at fixed Gaussian-only cutoffs"},
       achieved:{rrm_positives:{count:432,of:460}, achieved_recall:0.93913…, negatives:{count:972,of:18592,set:"envelope synthetic"}, test_positives:null},
       precalc:{dir:"precomputed-filters/gaussian-rrm/rrm95-disp45",
                files:{rotations:{sha256,npy_sha256,shape,dtype}, translations:…, "filtered-sizes":…, "filtered-concat":…},
                counts:{rotations,translations,pairs}, audit:{min_rel_margin, min_tilt_gap}}}, …12 entries},
 golden:{tolerance_rel:1e-12, scores:[{d,phi_deg,theta_deg,score}], decisions:[{name,d,phi_deg,theta_deg,accept}],
         lattice_decisions:[{name, xyz, accept}]},
 legacy:{model:{mean,covariance,normalizer_formula}, scale:"unnormalized exp(-q/2) >= density*norm",
         thresholds:{r90:1.30,r95:0.88,r99:0.38}, dir:"precomputed-filters/gaussian/{r90,r95,r99}",
         files:{…sha256,npy_sha256}, format:"fortran_order True, <i8, zstd CLI + XXH64", reproduced:"r90 pair set"},
 provenance:{report:{path,sha256,note:"uncommitted working-tree version; copy vendored in tests"},
             cutoff_envelope:{distance_lt_A:5.7, displacement_lt_deg:60, positives:460, negatives:18592},
             cutoff_rounding:"floor to 1e-5 of repr(raw)", generator:{commands}, precalc_environment:{…}}}
```

**Golden content.**
- **Scores:**
  - the mode;
  - (3.5, 0, 0), (4.0, 30, 20), (4.5, 45, 30) and (5.5, 59, 60);
  - three `descriptors_test.csv` rows, whose Julia values agree with ours to ≈1e-14.
- **Cutoff decisions:** for each of the six rounded cutoffs c, two points on the ray d > μ_d (φ, θ at the mean, which pass both prefilters), with score c(1 ± 1e-9). They are found by bisection, and each lists its accept/reject verdict for every name.
- **Prefilter boundaries:**
  - at d = μ_d: φ = 45° exactly (reject for `*-disp45`) and 44.99999°;
  - at d = μ_d: θ = 30° exactly and 29.99999°;
  - at these points the score is asserted to be ≥ 0.27592, so only the prefilter decides.
- **Lattice decisions:**
  - (0.75, 1.0, 1.25): the disp45 tie, which rejects;
  - (0,0,5) and (0,3,4): d = 5, which rejects;
  - (3,4,0) and (5,0,0): these fail φ < 60°;
  - (0,0,4.75): this passes the mask.

## 5. Work steps

**S1. Model module `alaric/gaussian_filter.py`.** It uses only numpy, math, decimal and json, has flat imports and is numpy 1.x/2.x neutral, so A9 can attach it in the preparic environment.
- Constants: model literals, the legacy model, the CUTOFFS table (raw value, report line, table), names, the mask and the filters.
- Functions:
  - `score(d, phi_rad, theta_rad, model=MODEL)` with the explicit q;
  - `descriptors(R, t)`;
  - `hopf_rotations() -> (R[98304,3,3], absR33)`;
  - `lattice() -> (ijk int, t float)`;
  - `mask_translation(ijk, disp45)` and `mask_rotation(R33, tilt30)`;
  - `floor_cutoff`;
  - `threshold_spec(name)` and `containment_edges(spec)`.
- *Exit:* `test_gaussian_model.py` and `test_gaussian_masks.py` pass (§7 rows 1–4, 7).

**S2. Generator core `alaric/gaussian_precalc.py`.**
- Functions:
  - `score_table(model)`;
  - `accept_pairs(spec_entry)`, which returns sizes, the kept cells and translations, a concat block iterator, and the audit;
  - `canonicalize_csr(arrays)`, which drops zero rows and unused translations and remaps indices (used for legacy);
  - `write_cache(dir, …)` through `npy_io.save_npy`, publish by rename, then the manifest.
- The legacy mode compares unnormalized exp(−q/2) ≥ density × norm, with no masks.
- *Exit:*
  - `test_gaussian_legacy.py`: generated r90 equals canonicalized shipped r90, with equal cells, rotations ≤ 1e-12, translations, sizes and concat exact;
  - `test_gaussian_generator.py`: rrm50 is built twice in-process with identical arrays and bytes, and the audit thresholds hold.

**S3. Verification helper and CLIs.**
- `alaric/gaussian_cache.py`: `verify_file`, `verify_filter_cache(spec, name, root)`, `verify_legacy`, and `GaussianCacheInstallationError`.
- `util/gaussian-precalc.py NAME --output DIR [--check]`.
- `util/gaussian-precalc-all.py [--root R] [--check] [--decompress]`. `--decompress` installs `.npy` instead of `.npy.zst`.
- `.gitignore`: add `precomputed-filters/gaussian-rrm/`.
- *Exit:* `test_gaussian_verify.py` uses temporary directories and rrm50 to check:
  - the `.npy` form passes;
  - the `.zst` form passes;
  - a `.zst` recompressed at level 19 passes as `recompressed`;
  - a flipped byte, a truncated file and a missing file each raise an error naming the generator command;
  - when both forms exist, the `.npy` is checked.

**S4. Bootstrap: build all 12, generate the JSON, commit.**
- `util/gaussian-filter-json.py --root R --output alaric/gaussian_stacking_filter.json [--check]`.
- *Exit:* `test_gaussian_spec_json.py` and `test_gaussian_caches.py` pass (§7). `--check` is byte-identical.

**S5. Documentation.** Write `documents/gaussian-filter.md` covering:
- the model and masks;
- the name grammar and containment;
- the cache layout;
- the install, `--check` and decompress commands;
- the checksum rule;
- the bootstrap and version-bump procedure.

Then hand the JSON contract to preparic.

## 6. Budget (measured counts; sizes and times are estimates, marked ~)

Measured with rounded cutoffs and the d < 5 mask. Concat is decompressed int32. The compressed size is ~0.45 B/pair, scaled from legacy r90 (15.0 MB for 33.2 M int64 pairs).

| name | rotations | translations | pairs | concat .npy | ~.zst |
|---|---|---|---|---|---|
| rrm50 | 7,920 | 728 | 4,111,232 | 16 MB | ~2 MB |
| rrm60 | 8,824 | 1,144 | 6,105,920 | 24 MB | ~3 MB |
| rrm70 | 11,368 | 1,640 | 12,303,744 | 49 MB | ~6 MB |
| rrm80 | 14,776 | 2,484 | 23,988,416 | 96 MB | ~11 MB |
| rrm90 | 22,928 | 5,450 | 80,969,280 | 324 MB | ~36 MB |
| rrm90-tilt30 | 12,736 | 5,450 | 57,538,752 | 230 MB | ~26 MB |
| rrm90-disp45 | 22,928 | 4,658 | 72,710,144 | 291 MB | ~33 MB |
| rrm90-tilt30-disp45 | 12,736 | 4,658 | 50,635,520 | 203 MB | ~23 MB |
| rrm95 | 30,560 | 8,452 | 173,472,656 | 694 MB | ~78 MB |
| rrm95-tilt30 | 12,736 | 8,452 | 93,382,960 | 374 MB | ~42 MB |
| rrm95-disp45 | 30,560 | 6,012 | 128,908,176 | 516 MB | ~58 MB |
| rrm95-tilt30-disp45 | 12,736 | 6,012 | 69,059,632 | 276 MB | ~31 MB |
| **total** | | | 773.2 M | 3.09 GB | ~350 MB |

These counts replace the miniplan's variant counts, which were computed for the superseded re-derived cutoffs.

**Time and memory.**
- The score table takes ~3 s; a grouped evaluation of all 12 filters at two cutoffs took 14 s.
- Per filter: ~1–3 s compute, ~2 s zstd for rrm95, and ~1 s re-hash.
- All 12 take ~1–3 min on one core.
- Peak RSS is ~1.5 GB.
- The legacy r90 test takes ~5 s with the table method. Brute force over all 6.8e9 pairs took 350 s, so the test must not use it.

## 7. Test matrix

Cache-dependent tests are marked `gaussian_caches`. They skip with the install command when the caches are absent, and fail instead when `ALARIC_GAUSSIAN_CACHES_REQUIRED=1`.

| # | Test (file) | Oracle clause |
|---|---|---|
| 1 | Golden scores reproduce at rel 1e-12 from the JSON's own parameters; P/normalizer literals agree with inv/formula to 1e-14 (`test_gaussian_model`) | A7 golden 1e-12 |
| 2 | `floor_cutoff`: the six values; exact multiple; 0.29 → "0.29000"; `float(s) <= raw` on 10⁵ random raws | A7 cutoffs (author's rounding) |
| 3 | Report check: vendored report sha256 matches; parse l.42–46, 51 → raw == CUTOFFS raw bit for bit; floor equals the JSON `cutoff_decimal`; the JSON has no value in {0.2288…, 0.2700…, 0.1644…, 0.0293…} | A7 cutoff provenance; forbidden values |
| 4 | Each variant's cutoff == its parent's; exactly the 12 names; grammar; `schema == 1`; containment edges derived from the JSON equal §1 | A7 parent==variant, containment |
| 5 | JSON regeneration byte-identical; spec_digest/version table; counts equal the report rows of §3.7; consumer refuses `schema: 2` | A7 content, G8 |
| 6 | Decision goldens: accept/reject around every cutoff, φ=45°/θ=30° boundaries, for all 12 names | A7 golden pairs |
| 7 | Exact-mask ties: lattice cases of §4 including d = 5; exhaustive 41³ check that the integer formulas equal the generator's mask; tilt predicate on stored R₃₃ | A8 strict, exact, no arccos |
| 8 | Installed caches verify against JSON `sha256`/`npy_sha256` (`test_gaussian_caches`) | A8 checksums == JSON |
| 9 | CSR invariants: dtypes int32/float64, C order, len(sizes) == R, Σ sizes == len(concat), 0 ≤ concat < M, rows strictly ascending, sizes > 0, cells distinct and ascending, translations on the 0.25 lattice inside the mask; loads through `anchor._validate_precalculated_arrays`, `_build_cell_lookup`, `_lattice_geometry` | A8 CSR invariants, runtime compatibility |
| 10 | Membership: 10⁵ uniform (cell, lattice point) pairs, 10⁵ drawn from members, and 10⁵ near the cutoff (score within ×2); CSR membership == model predicate via `score(descriptors(R, t))` | A8 random membership |
| 11 | Containment on caches: for every edge A ⊂ B, cells, translations and per-row sets satisfy A ⊆ B | A7/A8 containment lattice |
| 12 | Legacy r90 reproduction (row 1 of S2's exit) | A8 legacy |
| 13 | Verify helper failure modes (S3 exit); error text names the generator command | A8/A9 installation check, G9 |
| 14 | Audit thresholds hold in the manifests; build twice gives identical `npy_sha256` | G3 determinism |

## 8. Files

- **New:**
  - `alaric/gaussian_filter.py`, `alaric/gaussian_precalc.py`, `alaric/gaussian_cache.py`;
  - `alaric/gaussian_stacking_filter.json` (generated, committed);
  - `util/gaussian-precalc.py`, `util/gaussian-precalc-all.py`, `util/gaussian-filter-json.py`;
  - `tests/test_gaussian_{model,masks,generator,legacy,verify,spec_json,caches}.py`;
  - `tests/gaussian-data/envelope-analysis-5-report.md`;
  - `documents/gaussian-filter.md`.
- **Modified:** `.gitignore`.
- **Read, not modified:** `alaric/npy_io.py`, `alaric/anchor.py`, `precomputed-filters/gaussian/r90`.

## 9. Follow-ups (out of scope)

- The `anchor_precalc.py` pure-move refactor. After it, keep the concat as int32 rather than int64, which saves ~1.4 GB per process for rrm95.
- Validate names against the JSON in `anchor.py`, `resolve.py` and `schema.py`, and put the cache identity (`npy_sha256`) into the sigil.
- The A9 all-filter check, which calls `gaussian_cache.verify_filter_cache` and reads `mask`/`filters`/`cutoff` from the JSON.
- Preparic T17/T19 integration: read `mask`, not an `envelope` key.

## 10. Documents to correct

- **`oracle-contract.md`:**
  - l.212 "envelope: d < 5.7 Å" becomes `mask` d < 5 (5.7 is provenance only);
  - l.214 "cutoff (full-precision float)" becomes the five-decimal floor plus `raw_cutoff`;
  - l.215 "Gaussian-only RRM-N cutoff" should say "rounded down to 1e-5";
  - l.224–233: the "bit for bit" clause and table become the raw values bit for bit plus the rounded table;
  - l.246 "inside the envelope" becomes "inside the mask (d < 5, φ < 60°)".
- **`contract-boundaries.md`:**
  - l.159 "envelope mask (d < 5.7 Å…)" becomes d < 5;
  - l.159 "`rrm90`'s 0.2759…" becomes 0.27592.
- **`alaric-gaussian-miniplan.md`:**
  - l.13 (d < 5.7, d² < 32.49) becomes d² < 25;
  - §2 table l.23–38: full-precision cutoffs, re-derived variant cutoffs and counts;
  - l.67 `envelope: {distance_lt_A: 5.7…}`;
  - l.74 "cutoff (repr float)";
  - l.119 vendored descriptors and `test_cutoffs`;
  - l.128 (5.7 and "max d 4.94");
  - §3 fallback and §7 Seamless factory and materialiser.
- **`open-questions.md` l.32:** the author's own text says 5.7. Annotate it as superseded rather than editing the decision.
- **`draft-implementation-plan.md` l.182:** "six fixed base cutoffs… full-precision model" should note five-decimal cutoffs and the d < 5 mask.

## 11. Open questions

None blocking. For information: the report is an uncommitted working-tree file in its repository. Provenance therefore relies on its sha256 and the vendored copy. Committing it there would let the JSON also cite a commit.

### Critical Files for Implementation
- /home/agent/alaric/alaric/npy_io.py
- /home/agent/alaric/alaric/anchor.py
- /srv/shared/stacking-prediction-plots/envelope-analysis-5-report.md
- /srv/shared/stacking-prediction-plots/recover-gaussian/gaussian_model.json
- /home/agent/alaric/precomputed-filters/gaussian/r90/
