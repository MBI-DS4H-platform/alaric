# Alaric preparation steps

This document describes the Alaric steps that preparic runs inside Seamless
transformations during project preparation (A1–A6 and A9), the data they rely
on (A7, A8, A10, A11), and the Alaric actions whose behaviour the preparation
fixes (A12). The step letters and T-numbers are those of
[`plans/final/oracle-contract.md`](../plans/final/oracle-contract.md) and
[`plans/final/contract-boundaries.md`](../plans/final/contract-boundaries.md).

Every step is either a module with a Python function, attached to the
transformation, or a CLI with a dependency list that Seamless stages next to
it. A dependency list (`X.DEPS` or `X.py.DEPS.txt`) is read from its own
directory, has one relative file path per line, and has no comments or blank
lines. The listed files have no `.CHECKSUM` sidecars. A step reads only its
declared files: it does not import an installed Alaric package. Editing a
listed file changes the transformation's identity; editing anything else does
not. Each step is sandbox-tested with preparic's setup: eight local spawn
workers, no persistent caching, hashserver or database.

Fragments are numbered ordinally everywhere: fragment N is the Nth and
(N+1)th residues present, whatever their residue numbers, and a window across
a numbering gap is not a fragment.

## A1 / T5: discrete best-fit poses

The command-line wrapper accepts an RNA PDB and fitter options:

```bash
util/refe-best-fit-SEAMLESS.sh \
  --output /tmp/1b7f-best-fits \
  --fraglib-config ~/.alaric/fraglib.yaml \
  --conda alaric \
  tests/1b7f/pdbs/rna.pdb --exclude-pdb-code 1b7f
```

The grid spacing is fixed to `sqrt(3)/3`; the fitter emits its normal header
and rejects all three ignore flags. The output folder contains `best-fits.tsv` and
`pdbs/frag-N-SEQ-bestfit.pdb`. The TSV columns are exactly
`fragment sequence conformer rotamer grid_x grid_y grid_z rmsd`; fragment,
conformer, and rotamer indices are one-based. Fits use the discrete library
pose, with no minimization or exact superposition. TSV RMSDs use six decimal
places.

Workflows can use the factory in `util/seamlessify_refe_best_fit.py`:

```python
import json
from pathlib import Path

import seamless_config
from util.seamlessify_refe_best_fit import seamlessify_refe_best_fit
from util.refe_best_fit_seamless_support import identities

seamless_config.init()
identities_json = json.dumps(
    identities(Path("/path/to/alaric/fraglib")),
    sort_keys=True,
    separators=(",", ":"),
)
fit = seamlessify_refe_best_fit(
    ("--grid-spacing", "0.5773502691896257"),
    fraglib_config="~/.alaric/fraglib.yaml",
    conda_environment="alaric",
)
fit.args.rna_pdb.value = open("rna.pdb", "rb").read()
fit.args.exclude_pdb_code.value = "1b7f"
fit.args.fraglib_identities.value = identities_json
result = fit.run()
```

The caller initializes Seamless configuration. Factory construction leaves
three project pins unbound and does not initialize configuration or run the
transformation: `rna_pdb` (`bytes`), `exclude_pdb_code` (`str`), and
`fraglib_identities` (`str`). The result pin is a folder. The library YAML is
a Seamless meta-file named `META__FILE__fraglib.yaml`; its machine-specific
locator does not become part of transformation identity. The committed
library identity JSON does become an input, so changing library identity
invalidates the fit while relocating unchanged files does not.

`util/refe-best-fit.DEPS` declares the staged fitter modules and all 20
mononucleotide/dinucleotide template arrays. The large conformation and
rotamer payloads are located through the meta-file. The identity pin
contributes to transformation identity, but the fit does not rehash or
compare the payloads against it; installed library files and their sidecars
must be kept aligned. This matches the existing fitter's
`config(verify_checksums=False)` behavior. The factory pins only these fixed
dependencies and the locator; it never binds the three project inputs.

## A2 / T6: reference fragment PDBs

```bash
python util/refe-fragments.py RNA.pdb --templates TEMPLATE_DIR --output OUT_DIR
```

The output contains `frag-N-SEQ-refe.pdb` files, numbered by one-based RNA
sequence position. Windows spanning a residue-number gap are omitted. For
each nucleotide, input atom names must exactly match the staged monomer
template order; the reference PDB is then written in the corresponding staged
dinucleotide template order. Fragment numbering is ordinal, and A2's keys equal
A1's, also for an RNA with a numbering gap or a first residue other than 1
(`tests/test_w8_numbering_contract.py`). Preparic itself refuses RNA ranges
with unobserved residues, so gaps arise only in direct Alaric use.
`util/refe-fragments.DEPS`
declares the script, its three parser/writer modules, and the 20 templates.

## A3 / T7: pair and base tables

```bash
python util/refe-best-fit-tables.py BEST_FITS.tsv RNA.pdb \
  --exclude-pdb-code 1b7f \
  --fraglib-identities fraglib_identities.json \
  --fraglib-config ~/.alaric/fraglib.yaml \
  --templates-dir TEMPLATE_DIR \
  --output OUT_DIR
```

This single deliverable writes `pairs.tsv` and `bases.tsv`. Pair rows contain
`down`, `up`, `sequence`, `ovRMSD`, and `cRMSD`; base rows contain `fragment`,
`sequence`, `nucleotide1_ovRMSD`, and `nucleotide2_ovRMSD`. The existing
pair/base calculations read exact coordinates from the declared library, not
the rounded best-fit PDB files. Keep the `crmsds` key in `fraglib.yaml`; the
library configuration requires it even when the calculation does not load its
matrices. `util/refe-best-fit-tables.DEPS` declares the wrapper, both existing
table scripts, their Alaric modules, and all templates.

## A4 / T8: RMSD threshold conversion

```bash
python util/refe-thresholds.py pairs.tsv bases.tsv --output thresholds.json
```

The output is schema 1 with `pairs` and `bases` arrays. Each rounded threshold
is a two-decimal string. Pair rows retain integer `down`/`up`, string
`sequence`, rounded `ovrmsd`/`crmsd` strings, and numeric `ovrmsd_raw`/
`crmsd_raw` values. Base rows retain integer `fragment`, string `sequence`,
rounded `nuc1_ovrmsd`/`nuc2_ovrmsd` strings, and numeric raw counterparts.
Raw values come from the table text and must not be reconstructed from the
rounded values. `util/refe-thresholds.DEPS` declares the CLI and the shared
rounding module.

## A6: energy threshold conversion

`alaric.energy_threshold.energy_threshold(raw: float) -> str` returns the
smallest two-decimal value strictly above the raw value. It uses Python's
shortest round-trip float representation before decimal rounding.

Both A4 and A6 call `alaric.threshold_rounding.strictly_above_cent(text) ->
str`. The rounding rule is floor-to-cents plus one cent, so exact cent values
advance to the next cent, including negative values. For example, `0.12` maps
to `0.13`, `-16.59` maps to `-16.58`, and `-0.001` maps to `0.00`.

## A5 / T13: gridless identity-pose energetics

```bash
python util/energetics_seamless_support.py score LIGAND_DIR RECEPTOR.pdb
```

The CLI writes `energies.json` into its working directory; workflows may call
`score(ligand_directory, receptor_pdb)` instead. It reduces the receptor once
and every ligand PDB once, and scores each ligand in its identity pose against
the one receptor with attract-jax's float64 numpy gridless kernel, epsilon 15,
without minimization, cutoff, grids, JAX, scipy or compiled kernels. The
output has `schema: 1`, bead counts, per-ligand energies and per-ligand errors
(for example an unknown ligand residue). A receptor that cannot be reduced fails
the whole invocation. The output carries no range index: preparic runs one
instance per protein range and merges them itself.

`util/energetics_seamless_support.py.DEPS.txt` lists exactly eight files: the
script, `alaric/parse_pdb.py`, `util/reduce-npy.py`, `alaric/reduce.dat`, and
from the pinned attract-jax submodule `util/jax_scorer.py`,
`util/numpy_grid_score.py`, `util/pdb_io.py` and
`attract-original/attract-par.npz`. Grid and JAX code is deliberately absent,
so changing it does not invalidate preparic's energetics. The results agree with
attract-jax's numpy identity CLI to `1e-6`. The CLI's float32 `.npy` output is
not used for thresholds.

## A7 / A8: Gaussian stacking model and precomputed filters

The model, the twelve filters (`rrm50` … `rrm95`, and the `tilt30`/`disp45`
variants of `rrm90` and `rrm95`), their applied mask, five-decimal cutoffs,
cache layout and checksum rule are documented in
[gaussian-filter.md](gaussian-filter.md). The authoritative JSON is
`alaric/gaussian_stacking_filter.json`. Install the caches once per checkout
with `python util/gaussian-precalc-all.py` (§6 there). Preparic and the
`anchor` action only verify and read them; neither ever generates them.

Consumers read the applied `mask` (d < 5 Å and φ < 60°, both strict) and each
filter's `cutoff` from the JSON, with acceptance `score >= cutoff`. A variant
uses its parent's cutoff. The wider d < 5.7 Å envelope and the `raw_cutoff`
values are provenance only. Cache data is identified by `npy_sha256` (the
decompressed `.npy` bytes, header included), so recompressing a cache changes
no identity.

## A9 / T21: all-filter Gaussian check

```bash
python util/gaussian-check.py \
  --jobs jobs.json --best-fit BEST_FIT_DIR --proteins PROTEIN_DIR \
  --exclude-code 1b7f --fraglib-identities fraglib_identities.json \
  --semantic-identity identity.json [--repo-root ALARIC_ROOT] \
  --output result.json
```

Given a list of jobs, one per kept best-fit candidate of a selected stacking
(fragment, nucleotide, conformer, rotamer, grid cell, protein range and
residue), it decides for every one of the twelve filters whether Alaric's
`anchor` action would emit that pose cell. It reports count, capacity, fill
(the action's minimum fraction is 1/3), emission, pose pass and score. Filters
are walked from broad to strict along the JSON's containment edges. A filter
that is not reached is reported as `{emitted: false, implied_by: <filter>}`,
never as a measured zero. A job that cannot be evaluated (for example an
incomplete ring or a reconstruction error above 0.002 Å) gets a per-job
`error`.

The caches are opened only through `gaussian_cache.verify_filter_cache`. A
missing or corrupt cache raises `GaussianCacheInstallationError`, naming the
generator command. A corrupt `.npy` is never replaced by a different file, and
nothing is regenerated. `--semantic-identity` carries only `spec_digest` and the
`npy_sha256` values, so compression-only metadata does not enter the
transformation's identity. About twenty jobs take a few seconds in a cold
spawned worker, excluding decompression.
`util/gaussian-check.py.DEPS.txt` lists the CLI, its Alaric modules, the six
`ring-refe/` references (A10) and the twenty templates.

## A10 / A11: ring references and anchor constants

`ring-refe/refe-{A,C,PHE,HIS,ARG,TRP}.{pdb,npy}` are the ring references.
Preparic fits ring normals onto them by Kabsch superposition, both for stage 3
and for the best-fit stackings. `alaric/anchor_constants.py` holds the shared
limits: `L_OV = sqrt(3)` (the largest ovRMSD the 1/3-cell grid search covers
without misses), `BOUNDARY_RECOMPUTE_TOLERANCE = 1e-5` and
`MIN_CELL_FRACTION = 1/3`. Preparic mirrors them and tests the mirror
(`tests/frontend/w6-anchor-constants.test.ts` there). No environment variable
changes them.

## A12: action behaviour fixed by preparation

The actions are described in [middle-level.txt](middle-level.txt). The
preparation fixes these points:

- `grow` and `anchor-refe` raise an error for an ovRMSD above `L_OV` instead
  of clamping it. 1.73 runs, 1.74 raises.
- The middle layer additionally accepts `ovrmsd: -1` for `anchor-refe`. That is
  the export's sentinel for a disabled positive control. Such an action deploys
  but fails when it runs, and it does not stop the project's other actions.
- `energy-filter` replaces `score`, `filter` and `score_add`. It scores one
  fragment against exactly one protein range and keeps poses with energy
  strictly below the threshold. Several ranges are chained as successive AND
  filters and never summed. Its thresholds come from
  `DATA/energy-thresholds.json` (`schema: 1` plus `frag-N-SEQ` → `range-K` →
  two-decimal string, nothing else), and its `cascade`/`min_cascade` defaults
  come from `DATA/score-filter.yaml`. The grid cascade only accelerates the
  decision: poses still undecided are scored gridless, and the output equals
  the all-gridless result.

## Export seam

The export is implemented in Preparic as two pure graph transformers. The
shared `export_admission` result validates the pinned decisions and computed
evidence, publishes addressed hard blocks and backend-computed warnings, and
provides the canonical snapshot digest. The unshared `export_tree` folder
rebuilds that snapshot, checks admission and digest agreement, and returns the
DATA files, manifest and nested generator catalogue. Seamless's existing share
server serves one consistent cut as `export.zip`; no Context capture callback
or persistent export job is involved. Preparic includes the transformer
sources as modules, so it does not import Alaric from the installation.

`DATA/reference.pdb` is byte-identical to the all-atom RNA file used by the
fit. The archive carries the prepared range bytes, schema-1 constraints and
energy thresholds, score-filter defaults, kept fragment PDBs and a manifest
that records the digest and generator inventory. Gaussian `anchor` recipes
retain per-stacking actions. The `anchor-refe` tree deduplicates candidate
actions; each threshold comes from A4's rounded base value, or uses the
deliberately failing `-1` sentinel plus a YAML comment when that rounded value
exceeds `L_OV`. Edited kept-pair thresholds populate `constraints.json`; an
ovRMSD above `L_OV` blocks admission. Fixed recipes score a pool against the
required protein ranges as a tandem AND chain. Each unfixed range-rooted
recipe scores only against its own range, and already-scored restrictions do
not trigger redundant scoring.

The Stage 6 browser reads admission through its share and state-graph status.
It displays backend-supplied warning reasons and binds the local Proceed
anyway acknowledgment to the digest. A GET has no consent field. The browser
checks the downloaded manifest and current request state before releasing the
ZIP, cancels a request when its decision or readiness changes, and marks an
older download stale after later edits. Hard blocks are also enforced by
`export_tree`, so direct GETs cannot serve a successful blocked tree.

The ZIP's nested gen folders are alternatives, not a runnable action graph in
place. A user chooses compatible gens and moves their action directories into
the project root beside `DATA/`; Alaric's project loader ignores nested action
directories. The exporter does not choose merge locations or perform
run-final dead-end removal. See [the export plan](../plans/export-gens-plan.md)
and [project layout](../plans/alaric-project-layout.md) for the inventory and
selection rules. The old synthetic `alaric/export_bundle.py` API was retired
after its useful assertions moved to tests of real Preparic snapshots and
export trees.

## Verification

Run from the Alaric root. The sandbox tests use eight local spawn workers with
persistent storage disabled.

```bash
# A1-A4, A6
conda run --live-stream -n preparic python -m pytest \
  tests/test_w2_contract.py tests/test_w2_contract_boundaries.py \
  tests/test_w2_sandbox.py tests/test_w2_acceptance_edges.py
# A5 (in both environments)
conda run --live-stream -n alaric python -m pytest tests/test_energetics_deps.py \
  tests/test_w45_energetics_acceptance.py
# A7/A8 installation, A9-A11: see tests/W6_ACCEPTANCE.md
# A12, numbering and export seam
conda run --live-stream -n alaric python -m pytest tests/test_w8_*.py \
  tests/middle/test_anchor_refe_action.py
```

The shared payload contracts and their fixtures are described in
[preparation-contracts.md](preparation-contracts.md).
