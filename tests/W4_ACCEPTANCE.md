# W4 A5 acceptance set

Authority: oracle-contract G2–G4/G8 and A5; contract-boundaries T13 and §4;
implementation-plan §7 W4. A1 prepared these checks without running tests.

Run from the Alaric root, after A2 provides the scorer, exact DEPS list and
published prerequisite data:

```sh
conda run --live-stream -n alaric python -m pytest tests/test_energetics_deps.py -q
conda run --live-stream -n preparic python -m pytest tests/test_energetics_deps.py -q
```

Both environments need numpy, BioPython, pytest and the local Seamless CLI
packages. Missing packages are setup failures, not passing/skipped acceptance.
Fixtures are the committed `frag-1-GU-bestfit.pdb` and `1b7f_dom1-aa.pdb`.
The CLI is `python energetics_seamless_support.py score LIGANDS RECEPTOR`,
writing `energies.json` in its working directory. No optional factory API is
required by the test set.

The tests cover the exact eight dependency paths and sidecar prohibition,
mandatory `input_paths()`, a flat import-restricted execution, a real
`seamless-run -I` sandbox with eight spawn workers and persistent caching off,
full-precision identity CLI agreement, float64 JSON preservation, bead counts,
exact output fields, unknown-ligand errors, receptor failure, one reduction
per input, and committed/published prerequisite artifacts. The reduction-count
test adds observation code to the disposable copy of the declared reducer.
The import blocker permits installed numpy/BioPython and blocks JAX, scipy,
compiled kernels and hidden source-checkout modules.

Additional delivery evidence required before W4 closes:

- Compare the actual chosen transformation identity before/after editing each
  of the eight byte-pinned files in a disposable checkout; every edit changes
  the identity. Editing only a grid/JAX file leaves it unchanged. Test the real
  factory if provided, otherwise the `seamless-run -I` delivery graph. The
  manifest checks alone are not proof of this identity behavior.
- If a factory is supplied, show its ligand-folder and receptor pins are still
  unbound after construction and it does not initialize/execute Seamless.
- After the coordinator commits the Alaric pointer bump, verify that
  `git ls-tree HEAD attract-jax` names the published commit containing both
  NumPy commits and all three accuracy files. The pre-commit test checks the
  working submodule because the coordinator owns the Alaric phase commit.
- Record raw successful energy and bead counts from both environments and
  compare to the printed precision. Never use minfor's float32 `.npy` output
  as the threshold source.

These checks concern A5 only. Per-range reconciliation, clip independence in
the Preparic graph and stage-5 UI are W5 acceptance.

Focused uncovered A5 cases and W5 integration/editor acceptance are specified in
[W45_ACCEPTANCE.md](W45_ACCEPTANCE.md). Neither set was executed during authoring.
