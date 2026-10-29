# Alaric project layout (as a frontend sees it)

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
`<protein>.pdb` (original numbering, for anchoring) and `<protein>-aa.pdb` (all-atom, for scoring); `reference.pdb` (bound RNA); `rna.pdb` (best-fit input); `pdbcode.txt` (PDB code excluded from the fragment library); `anchor.yaml` (`angle`, `dihedral`, `ovrmsd`); `constraints.json` (per-fragment `sequence` and `scores{protein}`, and `pairs[{down,up,crmsd,ovrmsd}]`). These supply the defaults behind `auto`.

## Actions
`alaric.yaml` is strict (`action:` key; legacy `type:`). Dir names follow `frag<N>-<role>`: N is the dinucleotide at residues N and N+1, and a `-fwd`/`-bwd` suffix sets the grow direction. Dependency fields accept only the matching output kind.
- **Pose pools**: `anchor` (stack on protein `resid`/`nucleotide`; `angle`+`dihedral` or a precomputed gaussian r90/r95/r99 filter), `anchor-refe` (poses within `ovrmsd` of the reference nucleobase), `grow` (`input`, `restrict_input`, `crmsd`, `ovrmsd`), `filter` (`score_input`+`threshold`, or `mask_input`), `identity` (merge/dedup). `anchor-test`/`grow-test` are single-conformer variants.
- **Scores** (`score.npy`): `score` (ATTRACT), `rmsd`, `score_add`.
- **Masks**: `mask`, `mask-common-conformer`.

Pool dirs hold `poses-N.arc.zst`, plus `provenance.npy.zst` (grow/filter) or `map-1/2.npy.zst` (identity).

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
- The `refe-best-fit*.tsv` pairs and bases tables look like the source of the `DATA` thresholds. That link is inferred; no code enforces it.

## Outside the project
`~/.alaric/fraglib.yaml` and the env vars `ALARIC_DIR` and `ALARIC_REMOTE_{HOST,DEPLOYMENT_DIR,ALARIC_DIR,RESULT_DIR}`, which must also be set up on the remote machine.

Examples: `tests/1b7f-single-conformer`, `tests/1b7f-anchor-refe`, `tests/1b7f-gaussian-anchor` (yaml only), `tests/1b7f-project-alaric2-local` (with `CACHE`).
