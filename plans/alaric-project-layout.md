# Alaric project layout (as a frontend sees it)

> **Updated 2026-11-07 (E7).** Corrected for the delivered preparation work and export catalogue: retired score/filter actions, the twelve Gaussian filters, the √3 Å ovRMSD limit, threshold sources, and how to select and move exported gens. The actions are specified in `documents/middle-level.txt`.

An Alaric project is a plain directory. Its root is whichever directory holds `DATA/` (CLIs search upward from the cwd). It is a DAG of **action dirs** over shared inputs, with a content-addressed cache.

```
myproject/
  DATA/                 hand-authored inputs
  frag4-anchor-fwd/     action dir: any subdir with alaric.yaml
    alaric.yaml           the only file the user edits
    sigil.txt             from alaric-sigil
    run.sh check.sh       from alaric-deploy (chunk mode: chunkN.sh, organize.sh)
    result.txt            result checksum; present = done
    results -> ../CACHE/results/<SIGIL>
  CACHE/                generated
    parameters/<SIGIL>    resolved parameters + dependency sigils (JSON)
    checksum/<SIGIL>      result checksum
    results/<SIGIL>/      bulk data, plus <SIGIL>.INDEX / .CHECKSUM sidecars
  templates/<deployer>/<action>.sh   optional script-template overrides
```

## DATA
`<protein>.pdb` (original numbering, for anchoring) and `<protein>-aa.pdb` (all-atom, for scoring); `reference.pdb` (bound RNA); `rna.pdb` (best-fit input); `pdbcode.txt` (PDB code excluded from the fragment library); `anchor.yaml` (`ovrmsd`); schema-1 `constraints.json` (`fragN.sequence` and adjacent `pairs[{down,up,sequence,crmsd,ovrmsd}]`); `energy-thresholds.json` (`schema: 1` plus `frag-N-SEQ` → `range-K` → two-decimal threshold) and `score-filter.yaml` (`cascade`, `min_cascade`) for `energy-filter`. These supply the defaults behind `auto`. The constraints consumer rejects legacy/unversioned or malformed files. Nothing fills `constraints.json` from the best-fit tables: a Preparic export writes explicit thresholds from the preparation's edited values (see below).

## Preparic export catalogue

Preparic downloads a ZIP with one top-level `alaric_project/` directory. It
contains `DATA/`, the kept reference and best-fit fragment PDBs, a schema-1
manifest, and nested generator folders. A generator folder is a catalogue
choice, not an action directory in the project root. Alaric's loader therefore
ignores it until its action directories are moved beside `DATA/`.

Choose a compatible set of anchor, grow and merge generators for one run, then
move the selected `gen-*/*` action directories into the project root. Fixed and
unfixed layouts are alternatives. Gaussian-anchor and reference-anchor trees
are alternatives for each anchor run; merge folders identify the source tree
when their inputs differ. Pool names may collide between alternative gens, so
do not move conflicting choices into the same run. Common-conformer generators
read plain pools and write distinct filtered pools; choosing them is optional.
The manifest records each generator's final pools and external inputs to help
with selection. The exporter does not assemble the run, choose a merge point,
or perform run-final dead-end removal. There is no bundled README.

## Actions
`alaric.yaml` is strict (`action:` key; legacy `type:`). Dir names follow `frag<N>-<role>`: N is the dinucleotide at residues N and N+1, and a `-fwd`/`-bwd` suffix sets the grow direction. Dependency fields accept only the matching output kind.
- **Pose pools**: `anchor` (stack on protein `resid`/`nucleotide` through one of the twelve precomputed Gaussian filters `rrm50` … `rrm95-tilt30-disp45`; see `documents/gaussian-filter.md`), `anchor-refe` (poses within `ovrmsd` of the reference nucleobase), `grow` (`input`, `restrict_input`, `crmsd`, `ovrmsd`), `energy-filter` (one protein range, keep energy < `threshold`; ranges chain as AND), `identity` (merge two pools or deduplicate `input1` when `input2` is omitted), and `apply-mask` (filter a pose pool with the corresponding side of a `mask-common-conformer` result). `anchor-test`/`grow-test` are single-conformer variants. `grow` and `anchor-refe` raise on an `ovrmsd` above √3 Å; `anchor-refe` admits `-1` as an export's disabled-control sentinel, which fails when run.
- **Scores** (`score.npy`): `rmsd`. The `score`, `filter` and `score_add` actions are retired (`tests/legacy-middle-actions/`).
- **Masks**: `mask`, `mask-common-conformer` (paired selectors plus source-order metadata), `apply-mask` (matching pose-pool filter).

Pool dirs hold `poses-N.arc.zst`, plus `provenance.npy.zst` (grow, energy-filter, or apply-mask) or `map-1/2.npy.zst` (identity; one-input identity writes only `map-1.npy.zst`).

## Lifecycle and caching
1. `alaric-sigil` writes `sigil.txt` = `<10hex>-<10hex|null>`: hash of resolved parameters (DATA files by SHA-256), then hash of dependency sigils (`null` for roots). Editing `alaric.yaml` makes it stale and aborts unless `--force`/`--delete`.
2. `alaric-deploy <local|local-chunk|remote|remote-chunk> [nchunks] [dir]`: a hit in `CACHE/checksum/<SIGIL>` just writes `result.txt`. Otherwise it writes the scripts; remote mode ships them to `ALARIC_REMOTE_DEPLOYMENT_DIR`, with results in `ALARIC_REMOTE_RESULT_DIR/<SIGIL>`.
3. The user or SLURM runs `run.sh`; `check.sh` verifies the upstream sigils, results and data.
4. `alaric-result-check|download|upload|clean` move or free bulk data. The checksum is the durable identity, so data is re-fetchable.

## Downstream
`alaric-pool-graph` (graph JSON, `--select` for twice-visited fragments), `alaric-propagate-provenance`, then `alaric-chain --count | -o DIR` (`chains.txt`, `chains.json`, per-fragment pose dirs), `alaric-chain-focus`, `alaric-chain-coordinates`. `alaric-copy` and `alaric-count-poses` are helpers.

## Seamless and best-fit
- The project cache is Alaric's own sigil/checksum scheme. Seamless only computes checksums and canonical JSON; there is no `seamless-run` in project actions.
- `refe-best-fit` (`util/refe-best-fit-SEAMLESS.sh`, or the `seamlessify_refe_best_fit` Transformer factory) is a standalone step, not an action dir. It reads `rna.pdb` and writes `--output` (default `./best-fits`): `best-fits.tsv` and `pdbs/frag-*-bestfit.pdb`. Seamless caches it via `SEAMLESS_CACHE` or a cluster config. Its identity is the inputs, the fitter arguments, and the `fraglib/**/*.CHECKSUM` sidecars.
- The best-fit pairs and bases tables are **not** copied into `DATA/constraints.json` or `DATA/anchor.yaml`, and no code populates them from those tables. In Preparic, the tables become the computed thresholds of stage 4 (`documents/refe-best-fit-seamless.md`, A3/A4). The export writes the user's **edited** kept-pair thresholds to schema-1 `constraints.json`, uses computed base thresholds for `anchor-refe`, and writes edited energy thresholds for true kept matrix cells to `energy-thresholds.json` (`plans/export-gens-plan.md` §§4–5; `plans/final/oracle-contract.md` X3/X4).

## Outside the project
`~/.alaric/fraglib.yaml` and the env vars `ALARIC_DIR` and `ALARIC_REMOTE_{HOST,DEPLOYMENT_DIR,ALARIC_DIR,RESULT_DIR}`, which must also be set up on the remote machine.

Examples: `tests/1b7f-single-conformer`, `tests/1b7f-anchor-refe`, `tests/1b7f-gaussian-anchor` (yaml only), `tests/1b7f-project-alaric2-local` (with `CACHE`).
