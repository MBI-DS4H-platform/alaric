# Gaussian stacking filter

The Gaussian stacking filter decides which ring-level placements (a rotation of
the nucleotide ring relative to an aromatic protein ring, plus a translation)
count as a plausible stacking interaction. The anchor action uses it through
precomputed CSR caches. Other consumers (preparic `pose_pass`, the browser) use
the same definition through the committed JSON contract.

| what | where | in git |
|---|---|---|
| Model, masks, grids, cutoff table | `alaric/gaussian_filter.py` | yes |
| Generator core | `alaric/gaussian_precalc.py` | yes |
| Installation check | `alaric/gaussian_cache.py` | yes |
| JSON contract (generated) | `alaric/gaussian_stacking_filter.json` | yes |
| CLIs | `util/gaussian-precalc.py`, `util/gaussian-precalc-all.py`, `util/gaussian-filter-json.py` | yes |
| Caches (generated) | `precomputed-filters/gaussian/<name>/` | no (gitignored) |
| Tests | `tests/test_gaussian_*.py`, `tests/gaussian-data/` | yes |

Everything that defines the filter is a literal in `gaussian_filter.py`. The
caches are generated from those literals alone, and the JSON is generated from
the literals plus the cache manifests. Neither the caches nor the module depend
on the JSON.

## 1. Model

Three descriptors, in this order:

- **v** is the RNA ring centroid minus the protein ring centroid, expressed in
  the protein ring frame (z is the protein ring normal).
- **d** = |v|, in Å.
- **φ** (displacement) = arccos(|v_z| / d), the angle between v and the protein
  normal, in radians. Stage 3's slip angle is the same φ.
- **θ** (tilt) = arccos(min(1, |R₃₃|)), the angle between the two ring
  normals, in radians.
- Ring normals come from a Kabsch fit of each ring onto the reference rings
  (G→A, U→C, TYR→PHE).

The score is a trivariate Gaussian density:

```
D     = (d, φ, θ) − mean
q     = P00·D0² + P11·D1² + P22·D2² + 2·(P01·D0·D1 + P02·D0·D2 + P12·D1·D2)
score = exp(−q/2) / normalizer
```

with

```
mean       = [3.778988652846167, 0.3359170155526752, 0.21710309041293038]
covariance = [[0.13402643421739668,  0.048747464673082525, 0.03661672105978793 ],
              [0.048747464673082525, 0.05051245758442225,  0.018930688724029816],
              [0.03661672105978793,  0.018930688724029816, 0.039514509672839315]]
P          = covariance⁻¹
normalizer = (2π)^{3/2}·√det(covariance) = 0.17639457343866755
```

Degrees convert as `deg*(pi/180)`. The model comes from
`/srv/shared/stacking-prediction-plots/recover-gaussian/gaussian_model.json`
(sha256 `0bfb8acf…7db8`).

`P` and the normalizer are hard-coded literals (`PRECISION`, `NORMALIZER`), not
computed at import time. This keeps LAPACK differences between environments out
of the filter identity. `tests/test_gaussian_model.py` checks the literals
against `np.linalg.inv` and the formula to 1e-14.

`gaussian_filter.score(d, phi_rad, theta_rad)` evaluates the score (it
broadcasts), and `gaussian_filter.descriptors(R, t)` computes (d, φ, θ) from a
rotation and translation.

## 2. Mask and prefilters

**Applied mask.** Every consumer (the caches, `pose_pass`, the browser) applies
the same mask: **d < 5 Å and φ < 60°**, both strict.

The cutoffs were derived on a wider envelope (d < 5.7 Å, φ < 60°: 460 RRM
positives, 18,592 synthetic negatives). That envelope is provenance only. It
appears in the JSON under `provenance.cutoff_envelope` and never as a top-level
key, so it cannot be mistaken for the mask. In practice d < 5 is a no-op: the
largest score possible at d ≥ 5 is 0.021779, below the smallest cutoff
(0.06141).

**Prefilters.** Two optional prefilters, both strict:

- `tilt30`: θ < 30°;
- `disp45`: φ < 45°.

**Exact tests.** On the grid, masks are never decided through `arccos`.
For a lattice translation t = (i, j, k)·0.25 Å:

| predicate | integer test |
|---|---|
| d < 5 Å | i² + j² + k² < 400 |
| φ < 60° | 3k² > i² + j² |
| disp45 (φ < 45°) | k² > i² + j² |

For a rotation, tilt30 is `4·R33² > 3`, evaluated in float64 on the stored
matrix entry. These are `gaussian_filter.mask_translation` and
`gaussian_filter.mask_rotation`.

**Acceptance.** A pair is accepted by filter `name` iff all of these hold:

- `score(d, φ, θ) >= cutoff` (note `>=`);
- the applied mask holds;
- the name's prefilters hold;
- the translation is not the origin.

`gaussian_filter.accept_descriptor(name, d, phi_rad, theta_rad)` is the
reference predicate on continuous descriptors.

## 3. Names, cutoffs and containment

**Grammar.** `rrm<N>[-tilt30][-disp45]`, with N ∈ {50, 60, 70, 80, 90, 95}.
The suffixes are allowed only for N ∈ {90, 95}. This gives exactly 12 names:

```
rrm50  rrm60  rrm70  rrm80
rrm90  rrm90-tilt30  rrm90-disp45  rrm90-tilt30-disp45
rrm95  rrm95-tilt30  rrm95-disp45  rrm95-tilt30-disp45
```

`gaussian_filter.parse_name` validates a name; `all_names()` lists them.

**Cutoffs.** `rrmN` is the Gaussian-only cutoff that keeps N % of the 460 RRM
positives, taken from `envelope-analysis-5-report.md`. Each cutoff is the
report's full-precision value **rounded down** to the largest multiple of 1e-5
that is ≤ the raw value. A variant uses exactly its parent's cutoff, with the
prefilters applied on top. The cutoffs are not re-derived for the variants.

| name(s) | raw cutoff | cutoff |
|---|---|---|
| rrm50 | 3.3038440180955031 | 3.30384 |
| rrm60 | 2.9043182397831022 | 2.90431 |
| rrm70 | 2.0419804149333398 | 2.04198 |
| rrm80 | 1.1974880057674258 | 1.19748 |
| rrm90 and its 3 variants | 0.27592087772968793 | 0.27592 |
| rrm95 and its 3 variants | 0.061416348507101752 | 0.06141 |

Rounding happens in one place, `gaussian_filter.floor_cutoff(raw, decimals=5)`:
it takes `repr(raw)`, quantizes with `Decimal(..., ROUND_FLOOR)`, and asserts
that the result is ≤ raw. The rounding changes no dataset count from the
report; it adds a few thousand pairs to some caches.

**Containment.** Filter A ⊆ filter B iff A's prefilters include B's and
cutoff_A ≥ cutoff_B. This gives:

- rrm50 ⊂ rrm60 ⊂ rrm70 ⊂ rrm80 ⊂ rrm90 ⊂ rrm95;
- for N ∈ {90, 95}: rrmN-tilt30-disp45 ⊂ rrmN-tilt30 ⊂ rrmN and
  rrmN-tilt30-disp45 ⊂ rrmN-disp45 ⊂ rrmN;
- each rrm90 filter ⊂ its rrm95 counterpart.

`gaussian_filter.containment_edges(thresholds)` derives the edges from the JSON
entries. The containment also holds at the array level in the caches (same
cells, same lattice), and `tests/test_gaussian_caches.py` checks it.

## 4. Grid

**Rotations.** A Hopf grid of 6 faces × 16 × 16 × 64 = 98,304 cells, evaluated
at cell midpoints. The cell index is `((face*16 + p)*16 + q)*64 + m`, with m
varying fastest.

- a = tan(π/4·(2(p+½)/16 − 1)), and b likewise with q.
- The face direction s is (±1, a, b), (a, ±1, b) or (a, b, ±1) for faces 0/1,
  2/3, 4/5. Even faces take +. Then s is normalised.
- θs = arccos(s_z), φs = atan2(s_y, s_x), ψ = 2π(m+½)/64.
- q = (cos(θs/2)cos(ψ/2), cos(θs/2)sin(ψ/2), sin(θs/2)cos(φs+ψ/2),
  sin(θs/2)sin(φs+ψ/2)) as (w, x, y, z), converted to R[row, col] by the
  standard quaternion-to-matrix formula.

**Translations.** The 41³ lattice over ±5 Å in 0.25 Å steps, in C order (x
slowest, z fastest), with the origin excluded.

`gaussian_filter.hopf_rotations()` and `gaussian_filter.lattice()` build them.

## 5. Cache layout

Each filter has a directory `precomputed-filters/gaussian/<name>/` with four
arrays and a manifest. The arrays are a pure function of the accepted pair set:

| file | dtype, shape | content |
|---|---|---|
| `rotations.npy[.zst]` | float64 [R,3,3], C order | Hopf cells with ≥ 1 accepted pair, ascending cell index |
| `translations.npy[.zst]` | float64 [M,3] | lattice points with ≥ 1 accepted pair, lattice C order, values exactly i·0.25 |
| `filtered-sizes.npy[.zst]` | int32 [R] | number of accepted translations per row (always > 0) |
| `filtered-concat.npy[.zst]` | int32 [Σ sizes] | per row, ascending indices into `translations` |
| `manifest.json` | | name, cutoff, per-file digests/shape/dtype, counts, audit, build environment |

Row r accepts translations
`concat[sizes[:r].sum() : sizes[:r+1].sum()]`. No zero-size rows are stored: a
missing row and an empty row mean the same thing to `anchor.py`. Because only
accepted cells and points are stored, the layout does not depend on any
acceleration used by the generator.

The anchor loader (`anchor.py`) casts sizes/concat to int64, infers the lattice
origin and step from the stored translations, and requires the Hopf centres to
be distinct. The `.npy` files use the v1.0 header with `fortran_order: False`.
When both `X.npy` and `X.npy.zst` exist, `npy_io.find_npy` prefers `X.npy`.

Measured sizes:

| name | rotations | translations | pairs | on disk (.zst) |
|---|---|---|---|---|
| rrm50 | 7,920 | 728 | 4,111,232 | 0.2 MB |
| rrm60 | 8,824 | 1,144 | 6,105,920 | 0.3 MB |
| rrm70 | 11,368 | 1,640 | 12,303,744 | 0.4 MB |
| rrm80 | 14,776 | 2,484 | 23,988,416 | 0.9 MB |
| rrm90 | 22,928 | 5,450 | 80,969,280 | 3.8 MB |
| rrm90-tilt30 | 12,736 | 5,450 | 57,538,752 | 1.7 MB |
| rrm90-disp45 | 22,928 | 4,658 | 72,710,144 | 3.0 MB |
| rrm90-tilt30-disp45 | 12,736 | 4,658 | 50,635,520 | 1.3 MB |
| rrm95 | 30,560 | 8,452 | 173,472,656 | 9.5 MB |
| rrm95-tilt30 | 12,736 | 8,452 | 93,382,960 | 2.6 MB |
| rrm95-disp45 | 30,560 | 6,012 | 128,908,176 | 5.3 MB |
| rrm95-tilt30-disp45 | 12,736 | 6,012 | 69,059,632 | 1.4 MB |

The concat is 4 bytes per pair once decompressed (694 MB for rrm95; 3.09 GB for
all 12). Decompress only if you have the disk space and need faster loading.

## 6. Installing and checking the caches

The caches are not in git. Build them once per checkout (in the `alaric` env):

```bash
# all 12 filters into <root>/precomputed-filters/gaussian/<name>/ (~1–3 min, ~1.5 GB peak RSS)
python util/gaussian-precalc-all.py [--root R]

# one filter
python util/gaussian-precalc.py rrm95 --output precomputed-filters/gaussian/rrm95
```

Add `--decompress` to either command to install `X.npy` instead of
`X.npy.zst`. Both forms verify against the same JSON entry (§7).

To check that this machine reproduces the committed caches without touching
the installed ones:

```bash
python util/gaussian-precalc-all.py --check          # all 12
python util/gaussian-precalc.py rrm95 --check        # one filter
```

`--check` builds into a temporary directory and compares the new manifest with
the JSON:

- an `npy_sha256` mismatch is a hard failure (exit 1): the arrays differ, so
  the algorithm or environment has drifted;
- a `sha256`-only mismatch prints `recompressed, arrays identical` and exits 0:
  a different zstd produced different compressed bytes for the same arrays.

**Using a filter in a project.** In the anchor action of `alaric.yaml`:

```yaml
precomputed_filter_name: gaussian
precomputed_filter_threshold: rrm95       # any of the 12 names
```

The middle layer looks for `precomputed-filters/gaussian/<name>/` under the
project root first, then under the alaric repository. Remote runs use
`$ALARIC_REMOTE_ALARIC_DIR/../precomputed-filters/gaussian/<name>/`. On the
command line, `anchor.py --precalculated-anchor DIR` takes the directory
directly.

**Determinism audit.** The generator records two margins in each manifest and
fails if either is below 1e-10:

- `min_rel_margin` = min |score/cutoff − 1| over the masked region (measured
  minimum 1.29e-7, for rrm95);
- `min_tilt_gap` = min |4·R33² − 3| over all cells (6.06e-4).

A failure would mean that the cache depends on ULP-level differences in
`exp`/`arccos` between environments.

## 7. Checksum rule

Each file entry in the JSON (`thresholds.<name>.precalc.files.<file>`) has two
digests:

- **`npy_sha256`**: the sha256 of the complete decompressed `.npy` file (header
  and payload). This is **authoritative**: it identifies the array and does not
  depend on the compressor.
- **`sha256`**: the sha256 of the reference build's `.npy.zst` bytes
  (`zstandard` 0.25.0, libzstd 1.5.7, level 3, content size declared). It is
  only a fast path.

`gaussian_cache.verify_file` checks a logical file `X.npy` in the order the
loader resolves it:

1. If `X.npy` exists, hash it. It must equal `npy_sha256`.
2. Otherwise, if `X.npy.zst` exists and its bytes hash to `sha256`, it is OK
   (status `zst`).
3. Otherwise, stream-decompress `X.npy.zst` and hash the result. If it equals
   `npy_sha256`, it is OK (status `recompressed`). This accepts output from
   another zstd version.
4. Anything else (wrong digest, corrupt or truncated stream, missing file)
   raises `GaussianCacheInstallationError`.

The error message names the file, both expected digests and the command that
regenerates it. Nothing is regenerated silently.

`gaussian_cache.verify_filter_cache(spec, name, root)` verifies all four files
of a filter and returns the resolved paths, so the caller loads exactly the
files that were verified. The module uses only the standard library, plus
`zstandard` imported lazily for step 3, so it can be used from other
environments (such as preparic).

Downstream identities (cache keys, the anchor sigil) should use `npy_sha256`,
so that recompression never changes them.

## 8. The JSON contract

`alaric/gaussian_stacking_filter.json` is the single description of the filter
for consumers outside `gaussian_filter.py`. Top-level keys:

| key | content |
|---|---|
| `schema` | `1`. A consumer must refuse any other value. |
| `filter_set_version` | `"2026.1"` |
| `spec_digest` | sha256 of the canonical JSON of `model`, `mask`, `filters`, `grid` and each threshold's `rrm_target`/`filters`/`cutoff` |
| `model` | descriptor order and definitions, mean, covariance, precision, normalizer, formula, source |
| `mask` | `distance_lt_A: 5`, `displacement_lt_deg: 60`, and the exact integer forms |
| `filters` | `tilt30`, `disp45`: descriptor, `lt_deg`, exact form |
| `grid` | Hopf and lattice parameters, cell index formula, layout |
| `thresholds` | one entry per name (below) |
| `golden` | reference scores, accept/reject decisions around every cutoff and prefilter boundary, lattice mask decisions |
| `provenance` | report path and sha256, the cutoff-derivation envelope, the rounding rule, generator commands, build environment |

Each `thresholds.<name>` entry has `rrm_target`, `parent` (null for base
names), `filters`, `accept: ">="`, `cutoff` (float), `cutoff_decimal` (string),
`raw_cutoff`, `cutoff_rule`, `source` (report lines and table), `achieved`
(RRM positives of 460, recall, negatives with their set and denominator, test
positives of 58 for base names) and `precalc` (directory, per-file
`sha256`/`npy_sha256`/`shape`/`dtype`, counts and audit).

The `achieved` counts are copied from the report, not recomputed.

`gaussian_filter.load_spec(path=None)` loads the JSON and validates it through
`parse_spec`, which checks `schema` and recomputes `spec_digest`.

Consumers should read `mask`, `filters` and `thresholds.<name>.cutoff` from the
JSON. They should not hard-code the 5.7 Å envelope, and should not compute
their own cutoffs.

The JSON is written with `indent=1`, a fixed key order, float `repr`, a
trailing newline and no timestamps, so regenerating it from the same inputs is
byte-identical.

## 9. Bootstrap and version bumps

The order of the first build:

1. `python util/gaussian-precalc-all.py` builds the 12 caches. Each directory
   gets a `manifest.json` whose digests are read back from the published files.
2. `python util/gaussian-filter-json.py --root . --output alaric/gaussian_stacking_filter.json`
   assembles the JSON from the module constants, the goldens and the 12
   manifests. It reads three rows of
   `/srv/shared/stacking-prediction-plots/descriptors_test.csv` for score
   goldens (`--descriptors-test` overrides the path). It refuses manifests
   built in different environments.
3. Commit the JSON.

`python util/gaussian-filter-json.py --root . --check` regenerates the JSON in
memory and requires it to be byte-identical to the committed file.

**Changing the filter.** Any change to the model, a cutoff, the mask, a
prefilter or the grid is a new filter set:

1. Edit the literals in `alaric/gaussian_filter.py`. For a new model, recompute
   `PRECISION` and `NORMALIZER` once and paste them as literals. Update
   `CUTOFFS` and `REPORT_COUNTS` from the new report, and the vendored report
   copy and `REPORT_SOURCE` with it.
2. Bump `FILTER_SET_VERSION`.
3. Rebuild all caches: `python util/gaussian-precalc-all.py`.
4. Regenerate the JSON: `python util/gaussian-filter-json.py --root . --output alaric/gaussian_stacking_filter.json`.
5. Add the new `filter_set_version → spec_digest` pair to `DIGESTS` in
   `tests/test_gaussian_spec_json.py`. Keep the old pairs.
6. Run the tests with the caches required (below), then commit the module, the
   JSON and the test table together.

`spec_digest` covers only the semantic part of the JSON. A semantic edit
without a version bump fails `test_spec_digest_table`. A rebuild that changes
only the compressed bytes (`sha256`) does not need a version bump.

## 10. Tests

```bash
/home/agent/miniforge3/envs/alaric/bin/python -m pytest tests/test_gaussian_*.py
```

Tests that need installed caches are marked `gaussian_caches`. They skip with
the install command when the caches are absent. Set
`ALARIC_GAUSSIAN_CACHES_REQUIRED=1` to make them fail instead, which is what a
release or version bump should use.

| file | covers |
|---|---|
| `test_gaussian_model.py` | precision/normalizer literals, golden scores, `floor_cutoff`, cutoffs against the vendored report |
| `test_gaussian_masks.py` | lattice and Hopf grid, exact integer masks (exhaustive on the lattice), tie cases, tilt predicate, decision goldens at every cutoff and prefilter boundary |
| `test_gaussian_spec_json.py` | names and grammar, containment, `spec_digest` table, counts against the report, schema refusal, byte-identical regeneration, forbidden cutoff values |
| `test_gaussian_generator.py` | build twice gives identical bytes, canonical layout, audit failure is raised |
| `test_gaussian_verify.py` | checksum rule: `.npy`, `.zst`, recompressed, corrupt, truncated, missing, both forms present; error names the generator command |
| `test_gaussian_caches.py` | installed caches verify against the JSON; CSR invariants and anchor loader compatibility; random membership against the model; containment on the arrays; manifest audit thresholds |
