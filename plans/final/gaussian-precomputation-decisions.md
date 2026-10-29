# Gaussian filter precomputation: decisions and their effect on the source plans

Date: 2026-10-29. Scope: the Alaric-side precomputation of the 12 Gaussian stacking filters, deliverables A7 (the committed `alaric/gaussian_stacking_filter.json`) and A8 (the generator script and its gitignored caches). The implementation plan is `gaussian-precomputation-plan.md` in this folder.

Read this document together with `contract-boundaries.md` (CB, in this folder) and `../draft-implementation-plan.md` (DIP). Where they disagree with it, **this document wins**. It also overrides `../oracle-contract.md` (OC) §3.7–3.8 and `../alaric-gaussian-miniplan.md` (AG) on the same points. CB was brought in line with it when CB was finalized (2026-10-29); the other three documents have not yet been edited to match.

## 1. Readiness

The precomputation can be implemented now. None of DIP's open decisions D1–D17 blocks it:
- D8 (Kabsch ring normals in stage 3) changes stage 3 numbers and browser/A9 parity, not the caches.
- The cache content depends only on the model, the cutoffs, the masks and the grid constants, never on a project.

The source data is on disk:
- `/srv/shared/stacking-prediction-plots/recover-gaussian/gaussian_model.json` (the model);
- `/srv/shared/stacking-prediction-plots/envelope-analysis-5-report.md` (the cutoffs);
- `precomputed-filters/gaussian/{r90,r95,r99}` (the legacy set).

## 2. The filter set (unchanged)

There are 12 filters:
- 4 Gaussian-only filters: `rrm50`, `rrm60`, `rrm70`, `rrm80`;
- 2 × 4 filters: for N ∈ {90, 95}, `rrmN`, `rrmN-tilt30`, `rrmN-disp45` and `rrmN-tilt30-disp45`.

Every prefilter variant uses exactly its parent's cutoff, with the prefilters added on top, so its recall falls below N by definition. The re-derived per-variant cutoffs (0.229, 0.270, 0.164, 0.029) and the "RRM-95 fallback" must not appear. This is what CB §5.1, CB §6 and DIP §1 and §8 already say.

## 3. Decisions made on 2026-10-29

### 3.1 The hard distance mask is d < 5 Å, not d < 5.7 Å (author)
- The precomputation applies the mask d < 5 Å and φ < 60°, both strict. On the lattice they are tested exactly as d² < 25 and 3z² > x² + y². Lattice points at exactly d = 5, such as (3, 4, 0) and (0, 0, 5), are excluded.
- The author states that d < 5 and d < 5.7 give the same result. The planning agent verified this (plan §2):
  - **The report:** its "+ distance < 5 Å" rows have the same cutoff and counts as "Gaussian only".
  - **An analytic bound:** the largest possible score at d ≥ 5 is 0.0218, below the smallest cutoff, 0.06141.
  - **The lattice computation:** at all 12 filters, the pairs, rotations and translations are identical under no distance mask, d < 5.7 and d < 5. The largest kept d is 4.861 Å, for the rrm95 family.

  The mask is therefore a no-op, but it stays d < 5 Å as the contract.
- The JSON records the mask that is applied (d < 5, φ < 60). The report's envelope (d < 5.7, φ < 60, the set on which the cutoffs were derived) is recorded as provenance only.

### 3.2 The cutoffs are taken from the report, not re-derived (author)
- They were already derived in `envelope-analysis-5-report.md` (SHA-256 `11f854aa8e4e8a3ade18dcde84394f770bed08dea175796ae20e3f8953722edc`), written by `envelope-analysis-5.py`.
- Alaric does not need a cutoff-derivation module, a derivation CLI or a copy of the descriptor data.
- The JSON records the report's path, its SHA-256 and the rows the values came from.
- Achieved recall and positive/negative counts are copied from the report. The variants' counts come from its table "Added filters at fixed Gaussian-only cutoffs". The plan verified that the five-decimal rounding (§3.3) changes no positive or negative count.
- The rounding does add a few cache pairs, at most 4,800 (rrm95), out of 69–173 M pairs per affected filter. Kept rotation and translation counts do not change.

### 3.3 Cutoffs have five decimals, rounded down (author)
The JSON and the generator use five-decimal cutoffs. The full-precision report value is kept as `raw_cutoff` in provenance. The rounding happens in one place in Alaric's code.

| Name | Report value (full precision) | Cutoff (five decimals) |
|---|---|---|
| rrm50 | 3.303844018095503 | 3.30384 |
| rrm60 | 2.904318239783102 | 2.90431 |
| rrm70 | 2.04198041493334 | 2.04198 |
| rrm80 | 1.1974880057674258 | 1.19748 |
| rrm90 | 0.27592087772968793 | 0.27592 |
| rrm95 | 0.06141634850710175 | 0.06141 |

The values are rounded **down**, to the largest multiple of 1e-5 that is ≤ the raw value:
- Acceptance is `score >= cutoff`, and each raw cutoff is the exact score of the N-th positive.
- Rounding to nearest would raise rrm60, rrm80 and rrm95 (to 2.90432, 1.19749 and 0.06142) above that positive and drop it, so recall would fall below N.
- Rounding down keeps every positive the raw cutoff accepted.
- This mirrors the strictly-above rule used for RMSD and energy thresholds, where acceptance is the strict `<`.

The author confirmed rounding down on 2026-10-29.

Rounding preserves the order of the cutoffs, so the filter hierarchy (CB §5.2, OC A7 "Containment") is unchanged.

### 3.4 Scope of the implementation plan (Claude)
- **In scope:** A7 and A8. That means the numpy model, the Hopf rotations and the lattice, the exact masks, the generator and a wrapper that builds all 12 filters, deterministic serialization, the JSON with its checksums, golden vectors and legacy entry, the `.gitignore` entry, a reusable checksum-verification helper and the A7/A8 oracle tests.
- **Out of scope (follow-ups):**
  - the `anchor_precalc.py` pure-move refactor;
  - A9 (the all-filter check, T21);
  - the `anchor.py`/`resolve.py`/`schema.py` name validation and sigil changes;
  - preparic integration.
- **Left to the implementer, and resolved in the plan:**
  - The JSON stores `sha256` of the `.zst` bytes. A regeneration with a different zstd version can give different bytes for an identical array. The plan makes `npy_sha256` authoritative:
    - a `.zst` file whose bytes do not match `sha256`, but whose decompressed array matches `npy_sha256`, is accepted as `recompressed`;
    - downstream cache keys should use `npy_sha256`.
  - The order of bootstrap steps: generate the caches, write their checksums into the JSON, then commit.
  - A bounded-memory pair stage.
  - Cache sizes and runtimes measured freshly for the variants at their parents' cutoffs.

## 4. Effect on `contract-boundaries.md`
- **§1, "Large immutable data is identified, not staged":** unchanged.
- **§5.1, Generator:**
  - "The envelope mask (d < 5.7 Å, φ < 60°) is applied at this stage" becomes **d < 5 Å**, φ < 60° (§3.1).
  - "`rrm90-tilt30` uses `rrm90`'s 0.2759…" becomes 0.27592: cutoffs have five decimals (§3.3).
  - The rest is unchanged: parent cutoffs, the generator script, the gitignored folder, the ban on the re-derived values.
- **§5.1, compressed or not / the JSON is the contract / installation check:** unchanged. The plan only makes precise how a `.zst` mismatch with a matching `npy_sha256` is handled.
- **§5.2, the check tool (T21):**
  - Out of scope of this plan.
  - Its hierarchy and lattice still hold.
  - `pose_pass` must use the five-decimal cutoffs and the applied mask (d < 5), as the JSON records them.
- **§6, overrides table:** add rows for d < 5 instead of 5.7, for five-decimal cutoffs, and for "cutoffs come from the report, no re-derivation".

## 5. Effect on `draft-implementation-plan.md`
- **§1, last bullet ("Keep each prefilter variant's Gaussian-only parent cutoff…"):** unchanged.
- **§2, W6:** the precomputation part of W6 (A7, A8) now has its own plan in this folder. A9 to A11 and the `anchor_precalc` refactor remain W6 work.
- **§8 (W6):**
  - "Pin the recovered model/report provenance and a compact test dataset": pin the model and the report by SHA-256. No test dataset is needed for the cutoffs (§3.2).
  - "Generate A7's committed JSON …: full-precision model, six fixed base cutoffs, …":
    - the model (mean, covariance, precision, normalizer) stays full precision;
    - the six base cutoffs are the five-decimal values of §3.3, with `raw_cutoff` kept as provenance.
  - "Counts must be measured under those fixed cutoffs, not copied from re-derived-variant tables": counts come from the report's fixed-cutoff table (§3.2), except where five-decimal rounding changes them.
  - "Test … strict 30°/45°/envelope boundaries": the envelope boundary is d < 5 Å (d² < 25), plus φ < 60°.
  - "Test model vectors at relative 1e-12": unchanged for the score. The accept/reject golden pairs sit either side of the **five-decimal** cutoffs.
  - The rest of the A8 paragraph is unchanged: Hopf grid, lattice, algebraic masks, bounded memory, int32 indices, reproducible serialization, compressed versus decompressed identity.
- **§11, W9 table, "A7/A8 offline generation/provenance checks":** unchanged in kind. The cutoff check is now "the JSON equals the report rounded down", not a re-derivation.
- **§12, decisions:** no new author decision is needed. The rounding direction (§3.3) is confirmed.

## 6. Other documents still to correct (not edited)
- **`oracle-contract.md` §3.7 (A7):**
  - "envelope: d < 5.7 Å" becomes d < 5 Å.
  - The cutoff table and the clause "equal the report's values bit for bit" become five-decimal values that equal the report's values rounded down.
- **`oracle-contract.md` §3.8 (A8):** the "inside the envelope" condition becomes d < 5 Å.
- **`alaric-gaussian-miniplan.md`:**
  - §1: "envelope d < 5.7 (d² < 32.49)".
  - §2: full-precision cutoffs; variant rows; derivation from CSVs.
  - §3: the fallback.
  - §4: `derive_cutoffs`.
  - §5: `envelope` in the JSON.
  - §8 step 2: the vendored descriptors and `test_cutoffs.py`.
