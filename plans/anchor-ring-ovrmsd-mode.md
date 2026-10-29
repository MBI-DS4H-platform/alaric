# The `anchor-refe` action: anchor poses filtered by ovRMSD to a reference

Paths are relative to `/home/agent/alaric`. The mode is named `anchor-refe`, not "test" or "demo". Three details of the brief differ from the code. It fits **nucleobase heavy atoms**, not the stacking ring sets. Its reference comes from the **RNA PDB**, not from `ring-refe/*.npy`. It is a separate **action**, not a flag on `anchor`.

## 1. Name and selection
- Selected by `action: anchor-refe` (or `type:`) in `alaric.yaml` (`schema.py:107-113`). Required keys: `fragment`, `sequence`, `exclude`, `nucleotide`, `ovrmsd` (`schema.py:20,35`). Optional: `reference`, default `reference.pdb` (`:167-168`).
- `nucleotide` is `first` or `second` (`:123-127`). `ovrmsd` is a float > 0 (`:162-166`). `auto` is allowed for `fragment`, `sequence`, `exclude`, `ovrmsd` (`:82`).
- Unknown keys are rejected (`:116-118`): `angle`, `dihedral`, `margin`, `protein`, `resid`, `precomputed_*` and `conformer(s)` are all errors here.
- Neighbours (`schema.py:18-19`):
  - `anchor` + `angle`/`dihedral`: brute-force stacking (`resolve.py:289-294`).
  - `anchor` + `precomputed_filter_name`/`_threshold` (both or neither, `resolve.py:256-264`): the Gaussian cache. The threshold is `r90|r95|r99` (`anchor.py:38`) and the rings come from `ring-refe/` (`resolve.py:199-224,277-288`).
  - `anchor-test`: `anchor` plus exactly one of `nconformers`/`conformer` (`schema.py:130-144`). It is a conformer-limited stacking variant, **not** this mode.

## 2. What it computes
- **Atoms.** The base heavy atoms of the one selected nucleotide: template atoms with 2-character names (`library.py:499-503`; `only_base=True`, `anchor_refe.py:231`). That is 10 for A, 11 for G, 8 for C and U, including N1/N9 and the exocyclic N/O. They are **not** the 9/6-atom ring sets of `stack_geometry.py:6-11`. The other nucleotide is unconstrained.
- **Reference coordinates.** `DATA/reference.pdb` is passed as `--reference` (`resolve.py:311-320`, `backend.py:221-233`). `load_reference_fragment` (`anchor_refe.py:177-210`) parses the first model and matches atoms by name to the A/C/G/U templates (`reference.py:10-95`). It takes residues N and N+1, counted from the first residue (`reference.py:155-170`), and the atom mask is applied at `anchor_refe.py:308`. It errors on missing atoms, unknown residues or a sequence mismatch. The protein PDB and `ring-refe/` are not used.
- **ovRMSD.** For each library (conformer, rotamer, grid translation): `sqrt(sum|coords@R + t*sqrt(3)/3 - ref|^2 / n)` over those atoms, **no superposition**, in the PDB frame. The reference is wrapped as a single identity-pose source (`anchor_refe.py:241-268`) and run through grow's kernel (`:341-355`, `grow.py:904-1239`). Values within 1e-5 Å of the threshold are recomputed directly (`grow.py:35,1055-1081,1137-1183`).
- **Comparison: strict `<`** (`grow.py:1081,1135,1182`; oracle `tests/1b7f/test_anchor_refe.py:67`).
- **Granularity.** One scalar per action dir, i.e. per (fragment, nucleotide), not per atom, ring or residue. The same nucleotide differs by fragment: nucleotide 5 is 0.61 (fragment 4, second) and 0.25 (fragment 5, first).

## 3. Threshold inputs today
- `ovrmsd:` is a plain number in Å, with no rounding in code. It reaches the CLI through `str()` (`backend.py:228`) and enters the sigil, together with the checksum of the whole `reference.pdb` (`sigil.py:36-42`).
- The six values in `tests/1b7f-anchor-refe/*/alaric.yaml` (0.25, 0.57, 0.61, 0.61, 0.86, 0.47) are hand-pasted. I re-checked that each equals the matching `tests/1b7f/data/refe-best-fit-bases.tsv` entry rounded **up** to 0.01.
- `ovrmsd: auto` reads one project-wide value from `DATA/anchor.yaml` (`resolve.py:309-310`). That does not fit per-anchor thresholds. The fixtures' `anchor.yaml` lacks the key, so `auto` fails there. Only the synthetic `tests/middle/test_anchor_refe_action.py:45` uses it.

## 4. Deriving thresholds
- **Existing code.**
  - `util/refe-best-fit.py` picks the best conformer by whole-dinucleotide superposition RMSD, then the best rotamer and grid (`:118-153`). It writes `best-fits.tsv` and `frag-N-SEQ-bestfit.pdb`.
  - `util/refe-best-fit-bases.py` computes exactly the needed value. It uses the same `only_base` mask and unaligned RMSD, with the pose rebuilt from the TSV (`:146-214`). Output is `nucleotide1_ovRMSD` (for `first`) and `nucleotide2_ovRMSD` (for `second`), 6 decimals. It needs the fraglib, the TSV, `rna.pdb` and the same `--exclude`.
- **Rule** (`plans/stage4-thresholds-howto.md` §3): table value rounded up to 0.01, no margin. No code implements it. Since the test is `<`, a value that is an exact multiple of 0.01 would reject the best-fit pose itself (`test_anchor_refe.py:220-254` brackets it).
- **Seamless.** Only the fitter is wrapped: `util/seamlessify_refe_best_fit.py`, `util/refe-best-fit-SEAMLESS.{py,sh}`, `documents/refe-best-fit-seamless.md`. There is **no** wrapper for `bases`; `plans/stage4-best-fit-miniplan.md` only plans one.
- **Library-free route from the best-fit PDBs.** Compare the 2-character-name atoms of residue 1 or 2 of `frag-N-SEQ-bestfit.pdb` with the same-named atoms of `rna.pdb` residue N or N+1. I ran this on 1b7f: all 22 values agree within 2e-4 Å, but one rounded threshold flips. Fragment 5, nucleotide 2 gives 0.559941 (→ 0.56) from the PDB against 0.560037 (→ 0.57) exact, and 0.56 rejects the best-fit pose. The cause is the 3-decimal PDB format (`write_pdb.py:43`).
- **Consistency.** The fit reads `rna.pdb` but the action reads `reference.pdb`; they must hold the same coordinates (byte-identical in the fixtures, checked with `cmp`). The fit's `--exclude` must equal the action's `exclude`.

## 5. Grids
- None of the Gaussian machinery applies: no Hopf rotation cells, no cache, no ring-frame superposition, no ⅓ rule (`MIN_CELL_FRACTION`, `anchor.py:51,1052`). `anchor_refe.py` imports only `_select_conformers` from `anchor` (`:22`).
- Rotations are the library's own rotamers. Translations are the int16 pose grid in √3/3 Å units (`grow.py:31`). The test runs on the actual discrete pose, so a cell is simply in or out. A best-fit pose is on the grid, so it is emitted iff threshold > its base RMSD.
- Output is a source pool without provenance (`anchor_refe.py:353`).

## 6. Tests, limits, loose ends
- **Tests.** `tests/1b7f/test_anchor_refe.py` (needs the fraglib): brute-force equality (`:124,149`), the bracket (`:220`), the sequence mismatch (`:257`). `tests/middle/test_anchor_refe_action.py`: `auto`, sigil, local and chunk deploy, schema errors. `tests/1b7f-anchor-refe/` is YAML-only, with `DATA` symlinked to `../1b7f-single-conformer/DATA`: six anchor-refe dirs feeding grow, identity, score and filter, and no pose-count golden.
- **Limits.**
  - Chunking is by `--conformer-range` only (`deploy.py:30-34`), and `--nprocs` defaults to 1 on purpose.
  - Offset tables are built for thresholds up to `MAX_OVERLAP_RMSD = 1.3` (`grow.py:32,268-310`). Larger radii silently clamp (`:319-320`), nothing validates `ovrmsd`, and coverage above about 1.7 Å is untested. The 1b7f end values reach 2.246 (fragment 11, nucleotide 2).
  - Protein clashes are not considered; the fixtures follow with `score` and `filter`.
- **Unfinished.** No threshold generator, no margin policy, no per-anchor `auto`. `plans/alaric-project-layout.md:44` still calls the link between the best-fit tables and the YAML numbers "inferred".
