# Amendment: golden φ/θ boundary points must use the conditional mean

Amends `plans/final/gaussian-precomputation-plan.md` §4 "Golden content", "Prefilter boundaries" (l.237–240).

## Current plan text

> - **Prefilter boundaries:**
>   - at d = μ_d: φ = 45° exactly (reject for `*-disp45`) and 44.99999°;
>   - at d = μ_d: θ = 30° exactly and 29.99999°;
>   - at these points the score is asserted to be ≥ 0.27592, so only the prefilter decides.

## Intent

These points should test only the strict prefilters `disp45` (φ < 45°) and `tilt30` (θ < 30°). The score must clear the rrm90 family cutoff (0.27592, and therefore also 0.06141), so that a `*-disp45`/`*-tilt30` name rejects at 45°/30° and accepts at 44.99999°/29.99999° *only* because of the prefilter. Those two families are the only names with prefilters.

## The plan's points violate that assertion (for φ)

The other descriptors are held at the model mean μ = (3.778989 Å, 19.2466°, 12.4391°). Values were computed in the alaric env with `gaussian_filter.score`:

| point | score | ≥ 0.27592? |
|---|---|---|
| (μ_d, 45°, μ_θ) | 0.234541 | **no** |
| (μ_d, 44.99999°, μ_θ) | 0.234541 | **no** |
| (μ_d, μ_φ, 30°) | 1.093996 | yes |
| (μ_d, μ_φ, 29.99999°) | 1.093998 | yes |

At φ = 44.99999° the rrm90 names reject on score, so the "inside" case cannot show that disp45 is the deciding predicate. The plan's own assertion fails. The θ points happen to pass, but the same rule should be used for both.

## Conditional mean

Fix descriptor i at value v. The joint Gaussian is maximised over the other two descriptors at their conditional mean:

  x_j(v) = μ_j + Σ_ji / Σ_ii · (v − μ_i),  for j ≠ i,

and there q reduces to (v − μ_i)²/Σ_ii, so

  score = exp(−½ (v − μ_i)² / Σ_ii) / normalizer.

This is the largest score attainable at that φ (or θ). It is therefore the point that best isolates the prefilter. It is also the same marginal bound the legacy Julia `prefilter` used (`gaussian.jl:26-33`).

| fixed | d (Å) | φ (°) | θ (°) | score |
|---|---|---|---|---|
| φ = 45° | 4.212764 | 45 | 22.090751 | 0.767356 |
| φ = 44.99999° | 4.212764 | 44.99999 | 22.090747 | 0.767357 |
| θ = 30° | 4.063008 | 27.659742 | 30 | 1.726955 |
| θ = 29.99999° | 4.063007 | 27.659737 | 29.99999 | 1.726957 |

All four values are ≥ 0.27592, and they lie inside the mask (d < 5, φ < 60°). The rrm50–80 names reject on score at these points. They have no prefilters, so this is expected, and the decisions are still recorded for them.

## Proposed replacement text

> - **Prefilter boundaries:** each point fixes one descriptor and puts the other two at their conditional mean, x_j = μ_j + Σ_ji/Σ_ii·(v − μ_i), which maximises the score at that value:
>   - φ = 45° exactly (reject for `*-disp45`) and 44.99999°;
>   - θ = 30° exactly (reject for `*-tilt30`) and 29.99999°;
>   - at these points the score is asserted to be ≥ 0.27592 (measured 0.767 and 1.727), so for the rrm90/rrm95 families only the prefilter decides.

## Implementation status

The implementation already does this. The function `golden()` in `util/gaussian-filter-json.py` (where the golden generation lives, not in `gaussian_precalc`/`gaussian_filter`) uses a helper `cond(fixed, value_rad)` with exactly this formula. It asserts `score >= 0.27592` before recording each point, and `tests/test_gaussian_masks.py::test_descriptor_predicate_boundaries_from_json` checks the verdicts for the rrm90/95 families. Only the plan text needs amending.

## Review verdict

The amendment identifies a valid test-fixture correction. The prefilters remain straightforward strict geometry comparisons applied alongside the Gaussian cutoff. The fixed-cutoff analysis in `~/shared/stacking-prediction-plots/envelope-analysis-5-report.md`, implemented by `envelope-analysis-5.py`, already measures their effect; its conclusions are unaffected.

I independently reproduced the boundary scores. With distance and tilt held at their model means, displacement 44.99999° scores approximately 0.234541, below the rrm90 cutoff of 0.27592. Both sides of the proposed displacement boundary therefore reject for rrm90-disp45, preventing that fixture from demonstrating the strict comparison. The conditional-mean construction raises the score to approximately 0.767356 and correctly isolates the geometry predicate for the rrm90/95 families.

However, “must use the conditional mean” is too strong. Any fixture comfortably above the cutoff and satisfying the other geometry limits is sufficient. For example, fixing distance at 4.2 Å and tilt at 20° gives scores around 0.75145 on both sides of displacement 45°. Conditional means are a systematic, reasonable choice, rather than a requirement of the prefilter itself.

One statement above needs correction: rrm50–80 do not all reject every proposed point. The conditional θ boundary scores approximately 1.726955, exceeding the rrm80 cutoff of 1.19748, so rrm80 accepts it. Keep the implemented construction, soften the necessity claim, and correct that statement.
