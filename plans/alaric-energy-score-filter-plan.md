# Alaric: merged energy score+filter, one threshold per (dinucleotide, range)

Spin-off for preparic stage 5 (`energetics-miniplan.md`). Plan only; nothing is implemented.

## 1. Current state
- **Two actions.** `score` (`alaric/middle/schema.py:23`, `nb_kernel` compiled|jax at `:158-160`) runs `alaric/score.sh:146-163`: `minfor.py --oracle jax`, **no `--grid`**, so every run (and every chunk, `templates/local-chunk/score.sh`) regenerates a 10-12 grid just in time. It writes float32 `score.npy`. `filter` (`schema.py:27,169-182`) keeps `score < threshold`, strictly (`pose_filter.py:146-162`, `filter-poses.py:41`). Only `score` is chunkable (`middle/deploy.py:34`).
- **Thresholds** are hand-pasted numbers. Example: `tests/1b7f-anchor-refe/frag6-merged-filter` uses -16.59, which is the grid score of the best-fit, -16.5913 (`tests/1b7f/best-fits/frag-6-1b7f-dom1.score`), rounded up (ceil). That matches the RMSD rule in `stage4-thresholds-howto.md` §3. Some filters use summed ranges (`frag5-merged-score-all` = `score_add` dom1+dom2, threshold -22.28). The new design forbids summing.
- **No grid caching or Seamless** for scoring. The in-house `generate_grid` defaults to plateau 10 / neighbour 12 Å at 0.9 Å (`attract-jax/util/grid_generator.py:180-183`). The `--generate-grid` CLI cannot set them (`minfor.py:1786`).

## 2. The accuracy analysis, and checking the claims
The analysis is **not** in Alaric HEAD/HEAD~1 (685b6d0, f8372ae are best-fit Seamless work). It is in attract-jax's unpushed `22b23e4` (`docs/numpy-scoring.md`, "Larger sample") and `9453ee6`. Its data (`docs/numpy-grid-free-2048{.csv,-summary.json}`) is **untracked**.

**"5-7" and "10-12"** are the grid's plateau distance and neighbour (pairlist) distance in Å. Short-range pairs within the neighbour distance are summed explicitly, and the rest is interpolated, at 0.9 Å spacing. Dataset: 1,221 rigid perturbations (E ≤ 100) of 1B7F `frag5r` (18 beads) on `receptorr` (863 beads), with legacy grids.

| Grid − gridless, absolute error | median | p95 | p99 | max | > 0.05 | > 0.5 | signed range, fraction > 0 |
|---|---|---|---|---|---|---|---|
| 5-7 | 0.104 | 0.336 | 0.430 | **0.535** | 912 | **4** | −0.535…+0.438, 57 % |
| 10-12 | 0.0047 | 0.014 | 0.020 | **0.025** | 0 | 0 | −0.015…+0.025, 59 % |

- **The user's labels are swapped.** "10-12 accurate to 0.5" is true but 20× too loose. "5-7 accurate to 0.05" is **false**: 75 % of poses exceed 0.05. The data say 5-7 ≈ ±0.5 (and that bound is itself violated, by up to 0.535) and 10-12 ≈ ±0.05 (it holds, with a 2× margin).
- **Sign:** mixed, with negligible bias (+0.008 and +0.002). Margins must be symmetric. The error does not depend on the energy bin (5-7 max is 0.44-0.53 in every bin from −40 to 100).
- **Coverage:** one receptor and one ligand, near-native perturbations only. Nothing about per-range bounds, unbound ranges (3SXL) or dinucleotide ligands is established.
- **My re-run with in-house grids** (scratch, numpy backend, single core). The 5-7 grid matches the legacy grid within 8e-6. The 10-12 grid matches except the **identity pose: error 0.230**. There a ligand bead sits exactly on a grid node; offsetting it by 1e-9 Å restores 0.0003. This is an attract-jax edge-case bug. Identity/best-fit poses are exactly what thresholding scores, so this needs fixing before any cascade.
- **Cost per pose (1B7F):**

  | 5-7 grid | 10-12 grid | gridless |
  |---|---|---|
  | 92 µs | 382 µs (4.1×) | 790 µs (8.5×, scales with range size) |

  Grid generation: 23 s / 275 MB (5-7) and 31 s / 505 MB (10-12).

## 3. Cascade order (question for the user)
**The data support the user's order (5-7 → 10-12 → gridless), not the order implied by the stated numbers.** 5-7 is both the cheapest and the least accurate grid, so it is the coarse first pass. Refining with 5-7 after 10-12 could never resolve anything. The labels are the error.
- **Margins:** the cascade is exact only if the margin exceeds the true error. A 0.5 margin for 5-7 is unsafe (4/1,221 poses exceed it). Proposed: m₁ = 1.0 (5-7) and m₂ = 0.1 (10-12), with no mismatches on the dataset at any tested threshold.
- **Modelled cost** at thresholds −25…−10: 3-stage ≈ 105-128 µs/pose, versus 790 all-gridless and 383 for 10-12 → gridless. 5-7 → gridless (skipping 10-12, one grid per range) costs 115-160. It is a simpler option worth offering.

## 4. Merged action `energy-filter`
```yaml
action: energy-filter
input: frag5-merged
sequence: auto      # as score
exclude: auto
fragment: auto      # new; from the input pool
protein: range-2    # exactly ONE range (DATA/range-2-aa.pdb); lists rejected
threshold: auto     # or a number; auto -> DATA/energy-thresholds.json
cascade: [5-7, 10-12]   # [] = all gridless; margins fixed in code: {5-7: 1.0, 10-12: 0.1}
```
- **Output:** a POSE pool, the same format and provenance as `filter`, plus `stats.json` with per-stage counts.
- **No mixed energy array.** Kept poses' energies would mix grid and gridless values. Use a plain gridless `score` if energies are needed.
- **Thresholds file:** `energy-thresholds.json` is written by a preparic transformer (§6), alongside `interactions.json`:
  `{schema:1, method:"gridless", epsilon:15, source:"bestfit", rounding:"…", thresholds:{"frag-5-UU":{"range-2":"-16.60"}}}`.
  - Only true matrix cells appear. `auto` for a missing cell is a `ResolveError`, so there is no filter for a non-interacting pair.
  - A dinucleotide may interact with several ranges. Each (dinucleotide, range) pair is still its own action, scored against that range alone, and energies are never summed. **Decided (user): AND.** A pose must pass the filter for every range its dinucleotide interacts with, so the per-range actions are chained (filter A, then filter B).
- **Decision**, per pose, with grid energy g and margin m: keep if g < T−m, reject if g > T+m, else pass to the next stage. The last stage is gridless E vs T. Band edges always refine.
  - **Result:** the output equals the all-gridless result whenever |g−E| < m. It is then independent of grid backend, chunking and nprocs (deterministic). The gridless numpy kernel is float64 with fixed chunking.
  - T is parsed from its decimal string to float64. All comparisons are in float64, never against float32 `score.npy`.
- **Strict `<` and rounding.** Round half away from zero: −3.141 → −3.15, 0.123 → 0.13 (`Decimal(repr(E)).quantize(0.01, ROUND_UP)`).
  - For negative E this makes T **more negative** than the best-fit energy. With `E < T` the best-fit pose itself is then always rejected (−16.5913 → −16.60).
  - Today's ceil with `<` keeps it, except at exact multiples of 0.01. Ceil with `<=` always keeps it.
  - For positive (clash) thresholds, away-from-zero loosens. See Q2.
- **Legacy:** `score`/`filter`/`score_add` stay for old projects and fixtures. New projects use `energy-filter` only.

## 5. Grids and Seamless
- **One grid per (range, grid type).**
  - The alphabet is all RNA bead types from `reduce.dat`, so one grid serves every dinucleotide.
  - The grid is a pure function of: the reduced range, `attract-par.npz`, plateau, neighbour, spacing, extension, alphabet and epsilon.
- **Cached Seamless CLI**, following f8372ae/685b6d0: `util/energy-grid-SEAMLESS.sh RANGE_PDB --type 5-7|10-12 --output DIR`, plus `util/energy_grid_seamless_support.py` (`input_paths()`) and the factory `util/seamlessify_energy_grid.py`.
  - Code, `reduce.dat` and the `.npz` are pinned with `files_to_checksums`. The range-PDB folder pin stays **unbound**, so preparic can connect `pdbs-aa/` to it without rebuilding the transformer, as in the energetics miniplan.
  - Conda env `alaric` (Py3.12, numpy 1.26, scipy for `grid_generator`).
  - Preparic does not need grids, because thresholds are gridless. The factory is optional, for tests or a future preview.
  - Result `grid.npz` is in `SEAMLESS_CACHE`.
- `energy-filter`'s `run.sh` calls the CLI once per job, so chunks share the cached grid instead of regenerating it. The grid checksum goes into the action's parameters, and so into the sigil.

## 6. Threshold computation
- **Input: gridless best-fit energies** (`energies.json` from the planned `seamlessify_energetics`, `energetics-miniplan.md`), identity pose, one range.
- **Why best-fit, not refe.** The best-fit is a library conformer on the pool's own discretization, scored with the PDB code excluded. It is the near-native pose the pool can actually contain, and today's thresholds already use it. The refe energy is usually more favourable, so it would reject the best-fit pose. Show refe as information only.
- **Output:** the JSON above. Values are strings with two decimals, rounded by `alaric/energy_threshold.py:round_energy_threshold`, the single implementation used by preparic too.
- **Preparic wiring.** The computation is a Seamless transformer in preparic's dev `Context` (not plain backend code), so its result is cached.
  - **Inputs:** the cells `energetics`, `interaction_matrix`, `best_fits` and `best_fit_clip`.
  - **Output:** the cell `energy_thresholds`, mounted file-authoritative at `deployment-mounts/energy-thresholds.json` and shared read-only to the frontend.
  - **Code:** transformers must be self-contained, so `energy_threshold.py` has no imports beyond the standard library (`decimal`, `json`). Preparic loads it from `$ALARIC_DIR/energy_threshold.py` (same machine) and pins it by checksum, as a Seamless module or as inlined code. Factories in `util/` are found at `$ALARIC_DIR/../util`.
  - **Not designed yet:** copying the mount into an Alaric project's `DATA/`, which is the project-export stage.

## 7. Prerequisites
1. In attract-jax, commit the untracked `docs/numpy-grid-free-*`. Fix the on-node outlier (`grid_generator.py` neighbour assignment, or the lookup in `numpy_grid_score.py`). Add `--plateaudis/--neighbourdis` to `--generate-grid`.
2. Push `9453ee6`/`22b23e4`/fix, then bump Alaric's gitlink from `12bba50`.
3. Re-run the 2,048-pose comparison per range with in-house grids, on 1b7f dom1/dom2 and 3sxl dom1/dom2 with the dinucleotide ligands. Accept the margins only if max error < m/2.

## 8. Implementation steps
1. `alaric/energy_threshold.py` (rounding, JSON load), plus unit tests (−3.141, 0.123, −3.14, 0, −0.001).
2. The grid Seamless CLI and factory (§5), and `tests/test_seamlessify_energy_grid.py`.
3. `alaric/energy_filter.py`, the cascade engine on `pose_filter.py`'s chunk streaming. The selector interface (`pose_filter.py:88-93`) sees only index ranges, so extend it to receive the pose chunk. It uses an in-process `JaxScoreOracle`: JAX/compiled for grid stages, numpy `grid_free=True` for the last stage.
4. In `schema.py`, `resolve.py` and `deploy.py` (CHUNKABLE), add the `energy-filter` action and `threshold/fragment: auto`. Add `templates/{local,local-chunk,remote,remote-chunk}/energy-filter.sh`.
5. Tests:
   - **Golden 1b7f:** `tests/1b7f-anchor-refe` frag6/frag7 pools vs dom1. The cascade output must be byte-identical (arc and provenance) to `cascade: []`, with chunked and unchunked runs equal.
   - An injected-error test proving that band poses are refined.
   - Golden `energy-thresholds.json` for the 11 best-fits × 2 domains.
6. Document the action in `documents/`, and hand preparic the JSON contract.

## 9. Questions for the user
1. **Accuracy labels:** the data give 5-7 ≈ ±0.5 (max 0.535) and 10-12 ≈ ±0.025. Is it OK to keep your order 5-7 → 10-12 → gridless with margins 1.0/0.1, or would you rather drop 10-12 (5-7 → gridless)?
2. **Rounding vs `<`:** away-from-zero on negative thresholds rejects the best-fit pose. Options: ceil + `<` (today), ceil + `<=`, or your rule as stated.
3. **Threshold source:** best-fit (recommended) or refe, or the min/max of the two?
4. **Old actions:** keep `score`/`filter`/`score_add` for legacy projects, or remove them?
