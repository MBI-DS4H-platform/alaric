# Best-fit reference poses through Seamless

`util/refe-best-fit-SEAMLESS.sh` runs `util/refe-best-fit.py` as one cached bash
transformation. It works from any working directory and does not require an
Alaric pip installation on the submitting or executing host. A checkout is
needed on the submitting host: the wrapper declares the fitter, seven imported
source modules, its support module, and the 20 RNA templates as file inputs.
Seamless stages those files on the executing host; no preliminary copies are made.

## Requirements

- Bash and Python 3.12 or later on the submitting and executing hosts.
- NumPy (`numpy<2`, as specified by this repository), SciPy, `ruamel.yaml` and
  BioPython in the executing Python environment. The shell wrapper's input
  preparation uses only Python's standard library. No ATTRACT/JAX installation
  is needed here.
- A current Seamless installation providing `seamless-run` with `--metafile`
  support (`seamless-core`, `seamless-transformer`, and `seamless-config` with
  their dependencies). The Python CLI additionally imports
  `seamless.transformer` and `seamless_config`.
- Persistent Seamless storage: either an explicitly configured `SEAMLESS_CACHE`
  directory, or an existing Seamless cluster configuration. Neither wrapper
  chooses a cache path when `SEAMLESS_CACHE` is unset. The shell wrapper keeps
  the caller's working directory for configuration discovery. `SEAMLESS_CACHE`
  selects local execution and Seamless manages its local storage services.
- The immutable 0.5 dinucleotide conformer and rotamer library on the
  **executing host**, matching the committed sidecars in `fraglib/conformers/`
  and `fraglib/rotamers/`. These large files are neither read on submission nor
  uploaded. The submitting host needs only the checkout and configuration file.
- `~/.alaric/fraglib.yaml`, or an explicit `--fraglib-config FILE`. See
  [INSTALLATION.md](../INSTALLATION.md#8-configure-the-fragment-library) for
  the complete example. It must contain `fraglen: 2` and the conformer,
  replacement, extension, origin, rotamer and index patterns. The `crmsds` key
  is required by the library parser, but this fitter never reads cRMSD data.
  `XX` is expanded using the four representative sequences AA, AC, CA and CC;
  G/U coordinates are produced by mutation. The YAML is passed unchanged;
  variables and `~` are expanded by `library.py` on the executing host.
  Prefer absolute library paths or YAML-local variables so they do not depend
  on the temporary transformation workspace.

The templates are this checkout's `fraglib/templates` files, supplied as ordinary
file inputs. The wrapper overrides the YAML's template-directory setting to
use those staged templates, so workers need no separate template installation.

## Invocation

From any directory, with the repository checked out at `/path/to/alaric`:

```bash
/path/to/alaric/util/refe-best-fit-SEAMLESS.sh \
  --output /tmp/1b7f-best-fits \
  /path/to/alaric/tests/1b7f/pdbs/rna.pdb --exclude-pdb-code 1b7f
```

The original Python CLI path remains available with the same options and output
layout. It initializes configuration, calls the importable factory described
below, runs its Transformer, and writes the results:

```bash
/path/to/alaric/util/refe-best-fit-SEAMLESS.py \
  --output /tmp/1b7f-best-fits \
  /path/to/alaric/tests/1b7f/pdbs/rna.pdb --exclude-pdb-code 1b7f
```

The output directory contains `best-fits.tsv` and `pdbs/frag-*-bestfit.pdb`.
All wrapper options go **before** the RNA PDB; fitter options go after it.
`--output` defaults to `best-fits` in the caller's directory. `--pdb-output`
is managed internally; use `--output` instead. Seamless may also write its
normal checksum sidecars next to downloaded outputs.

Use `--conda ENV` to run the transformation in an existing conda environment
on the executing host. Without this option, the command uses the executing
host's `python` on `PATH`.

## Importable factory for workflows

The construction code is available as
`util/seamlessify_refe_best_fit.py`. With the checkout root on Python's import
path, import it as follows:

```python
from util.seamlessify_refe_best_fit import seamlessify_refe_best_fit
from seamless_workflow import Context

# The application has already initialized seamless_config and selected storage.
with Context() as ctx:
    ctx.best_fit = seamlessify_refe_best_fit(
        "/path/to/alaric/tests/1b7f/pdbs/rna.pdb",
        ("--exclude-pdb-code", "1b7f"),
        fraglib_config="/path/to/fraglib.yaml",
    )
    ctx.compute(timeout=300)
    result_index = ctx.best_fit.result.value
    # Consume the folder result in the workflow, or materialize it here.
```

Alternatively, with the checkout's `util/` directory on the import path, use
`from seamlessify_refe_best_fit import seamlessify_refe_best_fit`. Neither form
requires an Alaric pip installation.

The signature is
`seamlessify_refe_best_fit(rna_pdb, fitter_arguments=(), *, fraglib_config=None, conda_environment=None)`.
It returns a delayed bash `Transformer`, ready for assignment to a workflow
Context. Construction registers the input files through Seamless, but does not
run the fit or write an output directory. Outside a workflow, callers may also
explicitly call the returned Transformer's `.run()` and materialize its folder
result with Seamless's `write_to_directory` helper.

**The caller must initialize `seamless_config` before calling the factory.**
The factory never calls `seamless_config.init()` or changes the caller's
configuration. The existing Python CLI remains responsible for its own
initialization. `fraglib_config=None` selects `~/.alaric/fraglib.yaml`;
`conda_environment` selects the worker environment. Pass fitter options as a
sequence of strings; `--pdb-output` is managed internally.

## Identity and materialization

The RNA PDB, source modules, and templates are ordinary checksum-bearing inputs.
Fitter arguments are part of the bash command. The `fraglib_identities` string
pin records `fraglen: 2` and the precomputed SHA-256 values read from the committed
sidecars, keyed by library-relative artifact name. The 36 sidecars cover the
conformer arrays, replacements, extensions, origins, rotamers and indices for
AA, AC, CA and CC. They use `library.py`'s expected layout and contain a SHA-256
hex digest followed by a newline. Neither wrapper hashes library payloads,
copies inputs, or writes intermediate metadata files. The Python factory registers
files with Seamless's `files_to_checksums` and materializes the result through
its `write_to_directory` helper.

`fraglib.yaml` is injected with `--metafile`: its machine-specific locations
do not affect the transformation identity. It is a materialization hint for the
fixed artifacts identified by the sidecars. Installation must ensure those
files match their recorded identities; the wrapper trusts this, consistently
with the fitter's `config(verify_checksums=False)`. To change library content,
regenerate the sidecars as a deliberate library update; the changed identity pin
then invalidates cached results. Moving the same artifacts changes only the
meta-file. Templates are always read from staged file inputs, independently of worker
HOME configuration. No HOME rewriting or installed Alaric package is needed.

For workers with a different filesystem layout, supply a `--fraglib-config`
whose hints are appropriate there. The wrapper does not rewrite library paths
or download missing library data. A cache hit needs no worker execution.

Conda selection and other Seamless environment settings are execution
metadata, excluded from checksum identity. Keep numerical dependencies
compatible between runs sharing a cache. If comparing toolchains whose
differences affect results, use separate cache directories; selecting another
conda environment alone does not invalidate a cached result.

## Golden comparison

The 1b7f run above should reproduce
`tests/1b7f/data/refe-best-fit.tsv` and the 11 PDB fixtures in
`tests/1b7f/best-fits` (the `.score` files are for a different operation):

```bash
diff -u /path/to/alaric/tests/1b7f/data/refe-best-fit.tsv \
  /tmp/1b7f-best-fits/best-fits.tsv
for expected in /path/to/alaric/tests/1b7f/best-fits/*.pdb; do
  cmp "$expected" "/tmp/1b7f-best-fits/pdbs/$(basename "$expected")"
done
```

Focused tests guard both import styles and factory construction against
configuration initialization, verify the returned Transformer and its pins,
and check Context binding without running the fit. The opt-in integration tests
retain both original CLIs and execute the factory through a real Context,
compare the TSV and all 11 PDBs byte for byte, and check persistent-cache replay
with changed meta-file/environment settings:

```bash
RUN_SEAMLESS_BEST_FIT_TESTS=1 python -m pytest -q \
  tests/test_seamlessify_refe_best_fit.py \
  tests/test_refe_best_fit_seamless_inputs.py \
  tests/1b7f/test_refe_best_fit_seamless.py
```

Validation passed six focused tests and three real-library integration cases
in a Python environment with no Alaric distribution installed. The Context
integration initializes configuration in the caller, then replaces
`seamless_config.init` with a failing guard during factory construction and
workflow execution. The integration tests explicitly
select temporary caches for testing; neither CLI assigns `SEAMLESS_CACHE`.

Persistent caching and local execution are implemented. Configured Seamless
remote execution can receive the small payload, but automatic library
distribution, HPC deployment, scratch/fingertip, and parallel fitting are
outside this single-step wrapper's scope.
