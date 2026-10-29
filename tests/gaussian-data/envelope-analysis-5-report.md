# Envelope analysis 5: Gaussian score with added geometry filters (RRM train + validation)

The common envelope is distance <5.7 Å and displacement <60°.
It contains 460 RRM train + validation positives and 18,592 synthetic negatives.

Every curve accepts contacts with Gaussian score ≥ a swept threshold, together with its listed geometry filters. Scores tied at a threshold are accepted together. X is the number of accepted synthetic negatives divided by 18,592; Y is the number of accepted RRM train + validation positives divided by 460. These envelope denominators stay fixed for all five curves, including curves with added filters. Displacement <45° is shown alone and together with tilt <30°; neither includes the distance <5 Å filter. The first point rejects everything; the final point accepts every contact passing the added geometry filters.

| Added filter | RRM train + validation positives passing | Positive endpoint | Synthetic negatives passing | Negative endpoint |
| --- | ---: | ---: | ---: | ---: |
| Gaussian only | 460 | 100.00% | 18,592 | 100.00% |
| + distance < 5 Å | 453 | 98.48% | 12,551 | 67.51% |
| + tilt < 30° | 432 | 93.91% | 2,577 | 13.86% |
| + displacement < 45° | 449 | 97.61% | 10,903 | 58.64% |
| + tilt and displacement < 45° | 428 | 93.04% | 1,510 | 8.12% |

## Fixed positive recovery targets

For each target, the cutoff is the highest observed Gaussian score that selects at least that fraction of the original envelope positives. Unreachable means the geometry filter removes too many positives.

| Target | Added filter | Gaussian cutoff | Positives selected | Synthetic negatives selected |
| ---: | --- | ---: | ---: | ---: |
| 90% | Gaussian only | 0.275920878 | 414/460 | 578/18,592 (3.109%) |
| 90% | + distance < 5 Å | 0.275920878 | 414/460 | 578/18,592 (3.109%) |
| 90% | + tilt < 30° | 0.228869406 | 414/460 | 438/18,592 (2.356%) |
| 90% | + displacement < 45° | 0.27007176 | 414/460 | 541/18,592 (2.910%) |
| 90% | + tilt and displacement < 45° | 0.164444509 | 414/460 | 432/18,592 (2.324%) |
| 95% | Gaussian only | 0.0614163485 | 437/460 | 1,305/18,592 (7.019%) |
| 95% | + distance < 5 Å | 0.0614163485 | 437/460 | 1,305/18,592 (7.019%) |
| 95% | + tilt < 30° | Unreachable | 432/460 maximum | — |
| 95% | + displacement < 45° | 0.0293311127 | 437/460 | 1,204/18,592 (6.476%) |
| 95% | + tilt and displacement < 45° | Unreachable | 428/460 maximum | — |

The main plot displays Y from 50–100%. The left inset shows X from 0–2% and Y from 50–100%; the right inset shows X from 0–10% and Y from 85–100%, using the same envelope denominators.
Plot: `envelope-analysis-5-gaussian-roc.png`.

## RRM-derived cutoffs evaluated on the test set

Each cutoff is the highest observed Gaussian score that recovers at least the stated fraction of the 460 envelope RRM positives under the row’s filter. The same cutoff and filter are then applied to the 58 envelope test positives. The RRM target divides by 460; test Y divides by 58. For 50%, 60%, and 70%, X uses 100,000 independently sampled synthetic negatives already conditioned on distance <5.7 Å and displacement <60°, with seed 20261026. Sampling follows the existing generator: distance = 5.7 × U^(1/3), displacement = arccos(1 − U/2), and tilt = arccos(1 − U), with independent U values; scores use the same `recover-gaussian/gaussian_model.json` model. All other X values use the original 18,592 filtered synthetic negatives, as do both plots. Gaussian only and + distance <5 Å share the same 90% cutoff.

| RRM target | Filter | Gaussian cutoff (accept ≥) | X | Test Y |
| ---: | --- | ---: | ---: | ---: |
| 50% | Gaussian only | 3.3038440180955031 | 183/100,000 (0.183%) | 27/58 (46.55%) |
| 60% | Gaussian only | 2.9043182397831022 | 246/100,000 (0.246%) | 34/58 (58.62%) |
| 70% | Gaussian only | 2.0419804149333398 | 493/100,000 (0.493%) | 46/58 (79.31%) |
| 80% | Gaussian only | 1.1974880057674258 | 182/18,592 (0.979%) | 52/58 (89.66%) |
| 90% | Gaussian only | 0.27592087772968793 | 578/18,592 (3.109%) | 57/58 (98.28%) |
| 90% | + distance < 5 Å | 0.27592087772968793 | 578/18,592 (3.109%) | 57/58 (98.28%) |
| 90% | + tilt < 30° | 0.22886940586674059 | 438/18,592 (2.356%) | 57/58 (98.28%) |
| 90% | + displacement < 45° | 0.27007176031370117 | 541/18,592 (2.910%) | 56/58 (96.55%) |
| 90% | + tilt and displacement < 45° | 0.16444450868131616 | 432/18,592 (2.324%) | 57/58 (98.28%) |
| 95% | Gaussian only | 0.061416348507101752 | 1,305/18,592 (7.019%) | 58/58 (100.00%) |

## Added filters at fixed Gaussian-only cutoffs

For each target, the Gaussian-only cutoff remains fixed while tilt <30° and/or displacement <45° are added. X divides by 18,592 envelope synthetic negatives and RRM Y divides by 460 envelope positives. X changes are percentages of the Gaussian-only selected-negative count; RRM Y changes are percentage points.

| Gaussian-only target | Cutoff (accept ≥) | Added filter | X | Change in X vs Gaussian only | RRM Y | Change in RRM Y |
| ---: | ---: | --- | ---: | ---: | ---: | ---: |
| 90% | 0.27592087772968793 | Gaussian only | 578/18,592 (3.109%) | +0.00% | 414/460 (90.00%) | +0.00 pp |
| 90% | 0.27592087772968793 | + tilt < 30° | 414/18,592 (2.227%) | -28.37% | 411/460 (89.35%) | -0.65 pp |
| 90% | 0.27592087772968793 | + displacement < 45° | 533/18,592 (2.867%) | -7.79% | 413/460 (89.78%) | -0.22 pp |
| 90% | 0.27592087772968793 | + tilt and displacement < 45° | 375/18,592 (2.017%) | -35.12% | 410/460 (89.13%) | -0.87 pp |
| 95% | 0.061416348507101752 | Gaussian only | 1,305/18,592 (7.019%) | +0.00% | 437/460 (95.00%) | +0.00 pp |
| 95% | 0.061416348507101752 | + tilt < 30° | 718/18,592 (3.862%) | -44.98% | 425/460 (92.39%) | -2.61 pp |
| 95% | 0.061416348507101752 | + displacement < 45° | 972/18,592 (5.228%) | -25.52% | 432/460 (93.91%) | -1.09 pp |
| 95% | 0.061416348507101752 | + tilt and displacement < 45° | 534/18,592 (2.872%) | -59.08% | 423/460 (91.96%) | -3.04 pp |
