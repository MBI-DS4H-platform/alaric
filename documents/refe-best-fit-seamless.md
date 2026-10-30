# Alaric preparation steps: best fit, reference tables, and thresholds

This document describes W2's isolated Alaric deliverables. Each `.DEPS` file
is read from its own directory, has one relative file path per line, and has no
comments or blank lines. The listed files have no `.CHECKSUM` sidecars. The
steps use staged code and templates; they do not import an installed Alaric
package.

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
dinucleotide template order. Fragment numbering is ordinal. Gapped A2/A1 key
parity is deferred to W8; the W2 1b7f fixture is contiguous.
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

## Verification

The W2 acceptance set exercises the 1b7f best-fit and table goldens, reference
atom/key agreement, threshold schema and raw values, G7 examples, existing
YAML thresholds, declared dependency lists, flat sandbox imports, and
post-construction project pin connections. CLI sandbox runs use eight local
spawn workers, with persistent storage disabled:

```bash
conda run --live-stream -n preparic python -m pytest \
  tests/test_w2_contract.py tests/test_w2_contract_boundaries.py \
  tests/test_w2_sandbox.py
```
