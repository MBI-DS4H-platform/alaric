# attract-jax: gridless, numpy-only scoring (input for preparic "energetics")

## Entry points
- CLI: `attract-jax/util/minfor.py --oracle numpy --score --energy-only --no-grid ...`
- Python: `util/jax_scorer.py: JaxScoreOracle(backend="numpy", grid_free=True, energy_only=True, ...)`, then `score_batch(ens, dofs)`. Kernel: `util/numpy_grid_score.py` (`_direct_energies`). Docs: `attract-jax/docs/numpy-scoring.md`.
- Related modes: `--oracle numpy --grid X.grid|X.npz` (grid interpolation, numpy), and the old `--oracle jax`.

## Inputs
- Reduced (coarse-grained) ATTRACT beads only. Receptor: `--receptor-coordinates` (N,3) plus `--receptor-atomtypes` (+ optional `--receptor-charges`), or `--receptor-ens-list` / `--ligand-pdb` pointing at already-reduced ATTRACT PDBs (atom type in cols 55-59, charge 60-67, parsed by `util/pdb_io.py`). Ligand: `--ligand-ensemble` (C,N,3) plus `--ligand-atomtypes` (+ optional `--ligand-charges`).
- Force field: `--attract-par-npz attract-original/attract-par.npz` (tracked in attract-jax). `--epsilon` defaults to 15. rdie only (cdie raises).
- Poses: `--identity` scores the ligand where it sits (one pose, as the Alaric scripts do), or `--input-npy` (N,6) with `--input-format euler|rotvec`.
- All-atom to beads is done by Alaric, not attract-jax: `alaric/parse_pdb.py` (Biopython) to a ppdb.npy, then `util/reduce-npy.py` (rules in `alaric/reduce.dat`) to `*-coor.npy` and `*-atomtypes.npy`. `util/score-single.sh RECEPTOR_PDB LIGAND_PDB` is the template (identity transform, prints energy). Library ligand types come from `fraglib/templates/ATTRACT/<SEQ>-ppdb.npy` (`util/ligand-atomtypes.py`).

## Outputs
- One total energy per pose (kcal/mol-style ATTRACT units), no per-atom or LJ/electrostatic split, no gradients. CLI with `--output-npy` writes float32 (N,); without it, legacy-style `Energy:` lines on stdout. Python API returns float64 and an empty (0,6) gradient array.
- Physics: full 8-6 LJ over all pairs (no cutoff) plus rdie electrostatics with a shifted 50 A cutoff; r^2 floored at 0.001. Pair buffers are chunked (128 ligand x 512 receptor atoms) per pose. Atom type 99 (dummy) beads are dropped.

## What changed, and commit status
- Original path: JAX, plus a precomputed or generated grid and an optional compiled `nb_kernel*.so`. The new commits add a numpy backend that imports neither JAX nor the kernel (a test blocks JAX imports in a subprocess), plus `--no-grid`.
- Moved JAX-free helpers into `util/pdb_io.py` and `util/grid_io.py`. In-house grid defaults changed from 5/7 A to 10/12 A.
- Validation: grid-free matches legacy `attract-infinite --score` to about 1e-11. Versus the grid it differs by up to about 0.5 (5/7 grid) or 0.025 (10/12 grid), because there is no interpolation and no LJ cutoff.
- Git state: both commits (`9453ee6`, `22b23e4`) are local in attract-jax `main`, 2 ahead of origin (`12bba50`), unpushed. Alaric's gitlink still records `12bba50` (`git status` shows `M attract-jax`). Three untracked files in `attract-jax/docs/` (`numpy-grid-free-*.csv|json`) are uncommitted.
- Alaric does not use the numpy path yet. `alaric/score.sh`, `util/score-single.sh` and `util/score-fragments.sh` still call `--oracle jax --nb-kernel compiled` (the latter two with a generated grid), which needs JAX and the built `.so`.

## Environment
- Runtime needs only numpy and Python. No scipy and no jax are needed on the grid-free path; pyproject still lists jax, scipy and pyyaml, and numpy is unpinned.
- Verified here, run read-only. Pytest (`tests/test_energy_only.py`, `tests/test_numpy_scoring.py`): `alaric` env (Py3.12, numpy 1.26.4, jax 0.7.1) 25 passed; `preparic` env (Py3.15, numpy 2.5.4, no jax) 15 passed, 7 JAX-parity tests skipped. CLI smoke test on `1B7F-bound` (`receptorr.pdb`, `frag5r.pdb`, `--no-grid --identity`) gives -27.994417 (matches the docs) in `preparic` and `alaric2` (Py3.14, numpy 2.4.6), in about 0.14 s. numpy 2 therefore works.
- Alaric's own stack (sortednp) wants numpy<2, so run the energetics transform in a numpy-2 env, or in the `alaric` env (numpy 1.26, Py3.12).

## Seamless
- No Seamless wrapping of scoring exists. The only Seamless code is for best-fit (`util/refe-best-fit-SEAMLESS.*`, `seamlessify_refe_best_fit.py`, `documents/refe-best-fit-seamless.md`).
- Easy to wrap. The grid-free score is a pure, deterministic function of arrays (receptor beads and types, ligand beads and types, `attract-par.npz`, epsilon), with no per-receptor setup such as grid building. Every (ligand, receptor) pair is independently cacheable.
- Two routes: (a) a Python transformer importing `pdb_io.py`, `jax_scorer.py` and `numpy_grid_score.py` as staged modules and calling `JaxScoreOracle`; (b) a bash transformer around `minfor.py`, following the refe-best-fit-SEAMLESS pattern. Cleaner still: pass the .npy and PDB checksums to a small function that reduces and scores.
- Hard-coded paths: `ATTRACT_PAR_NPZ` is relative to the script dir, and `reduce.dat` and the templates are relative to `util/`. These must be declared as inputs.

## Limitations and gotchas
- `reduce-npy.py` discards the charges in `reduce.dat`, and the scripts pass no `--*-charges`, so electrostatics are zero. This is harmless for RNA ligands: RNA beads in `reduce.dat` carry no charge. It matters if a charged ligand or a charge-aware comparison is ever needed.
- `reduce-npy.py` asserts on unknown residues or missing bead atoms (for example P at a dinucleotide's 5' end), so the PDB must be complete.
- The numpy oracle rejects `--nb-kernel`, `--no-conformer-grouping` and the `--no-*-grid-eval` flags, so the existing scripts need flag edits, not just `--oracle numpy`.
- `util/` is not an installable package; it must be on `sys.path` or run as a script. Scoring is a single-process Python loop of O(poses x Nrec x Nlig), so large pose sets are slower than the grid. That is acceptable for a few dinucleotides against a protein range.
- Treat grid-based and grid-free scores as non-comparable (differences above).
- A batch of conformers against one receptor should be possible via `--input-npy` with zero rotvec DOFs plus `--ligand-conformers`; I did not test it.
