# How Alaric derives ovRMSD / cRMSD thresholds (for preparic stage 4)

Paths are relative to `/home/agent/alaric`. I re-ran all commands in the `alaric` conda env and got byte-identical output.

## 1. Command chain (tests/1b7f)
No Makefile or README records it. It is reconstructed from the golden tests (`tests/1b7f/test_rmsd_refe_best_fit.py:45-79`, `test_refe_best_fit_seamless.py:35`).
```
cd tests/1b7f
../../util/refe-best-fit.py       pdbs/rna.pdb --exclude-pdb-code 1b7f > data/refe-best-fit.tsv
../../util/refe-best-fit-pairs.py data/refe-best-fit.tsv --exclude 1b7f > data/refe-best-fit-pairs.tsv
../../util/refe-best-fit-bases.py data/refe-best-fit.tsv pdbs/rna.pdb --exclude 1b7f > data/refe-best-fit-bases.tsv
```
- **`refe-best-fit.tsv`:** one row per dinucleotide. Columns: `fragment` (1-based, covers residues f and f+1), `sequence`, 1-based `conformer` and `rotamer`, `grid_x/y/z` (units of sqrt(3)/3 Å), `rmsd`. **`best-fits.tsv`** is not in `data/`. It is this same file inside the Seamless fitter's result folder, next to `pdbs/frag-N-SEQ-bestfit.pdb`.
- All three tools read the fragment library via `~/.alaric/fraglib.yaml` (`/opt/fraglib`: rotaconformers 8.2 GB). `--exclude` must equal the fitter's PDB code, or conformer indices silently point at wrong coordinates.
- The cRMSD matrices (`/opt/fraglib/crmsd`) are **not** read. The YAML key only has to exist (`documents/refe-best-fit-seamless.md:33`).

## 2. Meaning
- **pairs.tsv:** one row per adjacent pair (down=N, up=N+1), 10 rows for 11 fragments. `sequence` is the trinucleotide. The shared nucleotide is the second of N and the first of N+1 (`util/refe-best-fit-pairs.py:208-223`).
  - **ovRMSD:** RMSD of that nucleotide as placed by the two fitted poses (no superposition). It is what `grow --ov-rmsd` bounds (`alaric/grow.py:1182`).
  - **cRMSD:** Kabsch RMSD between the two conformers' versions of it. It equals the cRMSD matrix entry (cross-checked at `test_rmsd_refe_best_fit.py:82-111`) and is bounded by `grow --crmsd` (`grow.py:544`).
- **bases.tsv:** one row per fragment. `nucleotideK_ovRMSD` is the RMSD of the fitted pose's base atoms (2-character atom names, `library.py:488-503`) against the reference, for nucleotide K=1 or 2 (`util/refe-best-fit-bases.py:146-214`). An inner nucleotide appears twice with different values.

## 3. Threshold rule (verified)
**Threshold = table value rounded UP to 0.01, per pair or nucleotide. There is no maximum.** It reproduces all 18 hand-copied numbers in `tests/1b7f-anchor-refe/*/alaric.yaml` and `tests/1b7f-single-conformer/*/alaric.yaml`.
- `grow`: use row (min, max) of source and target fragment, either direction. `frag6-fwd` uses pair 5→6: 1.201289 gives `ovrmsd: 1.21`, 0.825784 gives `crmsd: 0.83`.
- `anchor-refe`: use bases row N, column 1 for `nucleotide: first` and column 2 for `second`. `frag10-anchor` is 0.469029, so 0.47. The same nucleotide differs by action: nucleotide 5 is 0.61 as frag4 second and 0.25 as frag5 first.
- The comparisons are strict `<`, so rounding up lets the best-fit pose pass (unless the value is already a multiple of 0.01).
- The values are hand-pasted `alaric.yaml` snapshots. The pairs table was regenerated in `a663c6e` ("fix stale refe RMSDs") and the YAMLs edited by hand to match.
- **`constraints.json` is unrelated.** The `tests/1b7f-project*/DATA` copies are legacy hand-made files (`8dc89b7`, 3 decimals, different values, e.g. 5→6 is 0.76/0.25). They are read only for `crmsd/ovrmsd: auto` (`alaric/middle/resolve.py:143-164`). `1b7f-anchor-refe/DATA` has none.

## 4. Seamless status
The fitter is wrapped (`f8372ae`, `685b6d0`). **Pairs and bases are not.** To wrap them:
- Add a support subcommand and input list (`util/refe_best_fit_seamless_support.py:13,40-48` covers only the fitter) and a factory.
- Pins: flat sources (`library`, `superimpose`, `parse_pdb`, `reference`, ...), the script, the 20 templates (`config()` loads all, `library.py:1053`), `fraglib_identities`, the `fraglib.yaml` meta-file, `exclude_pdb_code`, `grid_spacing`, the fitter's TSV, and `rna.pdb` (bases only).
- Hard-coded `ALARIC_DIR` (pairs `:14`) and `~/.alaric/fraglib.yaml` are harmless with the existing `support.py:51-70` pattern.
- Runtime is library loading (non-mmap `np.load`, 16 libraries). The arithmetic is trivial.
- Env: the `alaric` conda env (Py 3.12, numpy 1.26). The preparic env cannot run the fitter (miniplan `:20`).
- **Library-free alternative (my experiment):** compute from the best-fit PDBs. Pairs: ovRMSD is the RMSD of nucleotide 2 of `frag-N-bestfit.pdb` against nucleotide 1 of `frag-N+1-bestfit.pdb`, and cRMSD is Kabsch on the same pair. This matches within 1.4e-4 Å, and all 20 rounded thresholds are identical. Bases (with the reference PDB set) has one flip in 22: fragment 5, nucleotide 2 gives 0.559941 against 0.560037, so 0.56 instead of 0.57. Cause: PDBs have 3 decimals (`write_pdb.py:43`). It is fine for display but not exact for thresholds.

## 5. Clipping
Fits are per-fragment, so clipping is a **pure subset**. Keep fragments `clip1+1 … n-clip2` and pairs with `down ≥ clip1+1` and `up ≤ n-clip2`. I ran the unmodified tools on a TSV of fragments 3-10: the rows were byte-identical to the filtered full outputs. Recommended: compute on the full fit and filter by clip, so clip edits never rerun the library step. Clipping matters: the worst values sit at the ends (ovRMSD 1.69, 1.56 and 1.34; fragment 11 nucleotide 2 is 2.246).

## 6. Gotchas and open questions
- Chain breaks leave gaps in `fragment`. Pairs skips non-adjacent rows silently (`:241`). Bases re-parses `rna.pdb`, so `--ignore-*` flags must match the fitter's.
- Nothing implements the rounding rule. Put it in Alaric beside the tools, not in preparic.
- Thresholds are tight, with exactly the best-fit value. Should users get a margin?
- Bases rows are needed only for anchor nucleotides (`tests/1b7f/data/anchors.txt`: 4, 5, 6, 11). What if an anchor lies in the clipped region?
- Can a fitter folder result feed a Python transformer? Unverified (miniplan `:50`). The factory also bakes its arguments into `transformer.code` (miniplan `:23`), the same blocker.
