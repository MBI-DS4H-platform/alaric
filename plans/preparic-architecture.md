# Preparic architecture summary

Preparic (`/home/agent/preparic`, GPL, "Phase 1") is a browser app for preparing Alaric runs: pick a PDB entry and biological assembly, define residue ranges, review RNA-protein stacking interactions. It is a Svelte frontend over a Seamless `Context` that acts as the backend.

## Frontend

Svelte 5 (runes) + TypeScript + Vite 6, with Mol* 5.13.1 embedded without its own UI. `App.svelte` drives a three-stage wizard held in `src/lib/state.svelte.ts`:

1. `Stage1Structure`: PDB code and biounit. Mol* downloads the CIF from RCSB.
2. `Stage2Ranges` + `RangeTable`: an Excel-style table of (chain, start, end, always_visible) rows, filled by typing or by picking in the viewer.
3. `Stage3Stackings` + `StackingTable`: a stacking table with per-row select checkboxes. Selected pairs are drawn in Mol* as sticks plus dashed centroid connectors.

`src/lib/molstar/viewer.ts` is the only Mol*-aware module. Pure logic (chain naming, range validation, stacking merge and auto-selection) is in `src/lib/*.ts`, covered by Vitest.

## Backend (Seamless)

`backend.py` builds one dev `Context` in share namespace `preparic`, with a Seamless share server on 127.0.0.1:5813. The graph:

- **Inputs:** `pdbcode`, `biounit` and `ranges` are file-authoritative mounted cells under the gitignored `deployment-mounts/`, so state survives restarts. `cif` is a writable, unmounted share that the browser fills.
- **Steps:** `ranges_to_pdb` (Biopython, `range_pdbs.py`) turns CIF plus range specs into `range_pdbs`, a list of PDB texts. Two branches consume it.
  - `detect_stacking` (`detect_stacking.py`) writes `stackings`, a JSON cell mounted at `stackings.json`. `stacking_selection` is a mounted `list[bool]` cell that the frontend and backend can both write.
  - `pdbs_to_folder` writes `pdbs/` as a folder cell. The `range-pdbs-aareduce.sh` Bash transformer, run in conda env `preparic` with the vendored ATTRACT `aareduce/` folder, writes `pdbs-aa/` (all-atom, via PDB2PQR).
- **Contract:** range 1 is RNA and the rest are protein. Transformer functions are self-contained, so results are content-addressed and cached by Seamless. Local setup uses `~/.seamless/clusters.yaml` (hashserver, database, jobserver).

## Connection

The Seamless share protocol is used, not a custom API. `index.html` loads `/seamless-client.js`. `src/lib/seamless.svelte.ts` calls `connect_seamless(null, null, "preparic")`, which opens REST and websocket connections to the share server. It exposes the shared cells as reactive `$state`, with revision counters, pending-write flags and loop-avoidance logic. The Vite dev server proxies `/preparic` (with `ws`) and `/seamless-client.js` to port 5813, so the frontend code is the same in dev and prod. Transformer failures reach the UI through the `state-graph` stream and appear as error banners. The viewer is the one part that does not use shared cells: it fetches the CIF directly from RCSB.

## Launch and deployment

Development only. `preparic-start` runs `backend.py` and `npm run dev` as supervised subprocesses. Options are `--log`, `--shell` (backend in IPython with `ctx`) and `--force`. `preparic-stop`, `preparic-restart` and `preparic-install-hook` put the scripts on the conda PATH. `environment.yml` provides Node 22, Biopython, PDB2PQR and `seamless-suite`, but the `seamless_workflow` package is not installed by it.

## Maturity

Stages 1-3 are working and tested. There are Vitest tests for the frontend, and pytest unit, isolated-Seamless and workflow tests with a 1B7F fixture. There are two known upstream xfails for empty-list results.

Not implemented:

- **Alaric:** nothing invokes Alaric. The `aareduce` output and stacking selection are produced and mounted but not yet consumed by any later stage, and Stage 3 has no completion action.
- **Deployment:** `preparic-on-seamless.md` is design only. It proposes packing `dist/` into a bytes-cell graph on a hashserver, a generic graph server, per-session Contexts, and a reverse proxy. None of this exists, and `dist/` is just an old build.
- **Other gaps:** no authentication, no multi-user state (one shared Context), no CI, and no remote or HPC execution (the Stage 3 plan scoped it out).
