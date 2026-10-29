# Preparic ↔ Alaric contract boundaries during project preparation

Final, 2026-10-29. This document fixes where preparic stops and Alaric starts while an Alaric project is being prepared (stages 1–6; the export stage is deferred). It lists every existing and planned preparic Seamless transformer and says, for each, what it takes from Alaric. §8 covers the second boundary, between preparic's browser frontend and its Seamless backend, through shared cells. The stage model built on those cells is `final/state-transitions.md`, whose proposed changes to §8 were applied here when both documents were finalized.

It applies the author's decisions, which override the mini-plans where they disagree (§6 lists those overrides). They come from:
- `open-questions.md`;
- the author's answers in `final/state-transitions.md` §11;
- `final/gaussian-precomputation-decisions.md`.

Questions that are still open are marked **Unresolved**, with their ID in `contract-open-questions.md`. Until they are answered, the text gives the provisional behaviour. Other plan documents are named by their path under `plans/`.

Short version: **Alaric delivers deterministic, sandbox-safe code (a module with a Python function, or a CLI tool with its file-dependency list) and data files. Preparic runs that code only inside Seamless transformations and pins the data by checksum.** Preparic never runs Alaric code outside Seamless and never imports the `alaric` package into its own process. Whether a given piece also gets an Alaric factory function is decided case by case (§1).

## 1. Rules that hold for every crossing

**Location.** Same machine, no remote workers. `ALARIC_DIR` keeps Alaric's meaning (the `alaric/` package folder, default `~/alaric/alaric`). The repo root is `Path(ALARIC_DIR).parent`, Alaric code for preparic lives in `<root>/util/` or `<root>/alaric/`, and preparic resolves both through one helper, `preparic/alaric_dir.py` (`preparic-start --alaric-dir`).

**What Alaric delivers.** For each step preparic runs, Alaric provides deterministic code in one of two forms:
- **(a) A module containing a Python function (preferred).** Inputs are function arguments, the output is the return value, and every file it reads is an argument or a module.
- **(b) A CLI tool together with a list of its file dependencies** (a DEPS file, §4). Inputs are files and arguments; the output is a file or directory.

Either form must **work in an isolated sandbox**: given only its declared inputs and dependencies, it runs and gives the same result. The only exception is data that is identified but not staged (next paragraph). Seamless itself is the sandbox test, and that test is part of the deliverable:
- for (a), run the function as a Python transformer, with its sibling modules attached as Seamless modules;
- for (b), run the tool through `seamless-run` with `-I <DEPS file>`.

A step passes when it succeeds there on the 1b7f fixtures and matches its golden output.

**Factories are a case-by-case decision for the implementer, not part of the contract.** A factory is an Alaric function that returns a ready Seamless transformer for preparic to assign into its `Context`. The existing example is a chain of three files:
- `util/refe-best-fit-SEAMLESS.sh` runs the fitter through `seamless-run`, with the dependencies from `input_paths()`, a meta-file and an identity variable.
- `util/refe-best-fit-SEAMLESS.py` ports that into Python.
- One component of that port is `util/seamlessify_refe_best_fit.py`, a factory that returns the equivalent bash `Transformer`.

How hard it is to turn a `seamless-run` invocation into a bash transformer depends on the dependency files.
- **Trivial:** a few flat files. Preparic can build the bash transformer itself from the CLI command and the DEPS list, and no factory is needed.
- **Complex:** many or nested dependencies, staged templates, meta-files, identity pins or name-collision rules, as for the best-fit fitter. Then a factory keeps that knowledge in Alaric.
- **Module case:** usually needs no factory. Preparic wraps the function in its own Python transformer. One constraint to weigh: a Python transformer with attached modules runs in-process by Seamless's default (the `mem` backend; for the spawn workers, see Execution below), so in the `preparic` env (Py3.15, numpy 2.5). That env is open-ended within limits. A module's dependencies can be added to it if they are modest and compatible with Py3.15/numpy 2. Heavy or conflicting ones are excluded: no JAX (the author kept preparic free of it on purpose), and nothing that pins numpy < 2. In Alaric, `sortednp` is the only such dependency, and only `identity_filter.py` uses it, so it is not needed by any step here. A module whose dependencies cannot meet those limits goes to form (b).

This document suggests a factory for T5 (exists) and T13; for every other Alaric step the form is left open (§7).

**If a factory is written**, it follows `seamlessify_refe_best_fit.py`:
- it imports only the standard library and Seamless, because it runs in preparic's backend process;
- it pins code, templates and parameter files at construction, with `files_to_checksums`;
- **every project-dependent input is an unbound pin** with a documented name and celltype, so preparic connects its cells and never rebuilds the transformer. The current factory bakes the RNA PDB path into `transformer.code`; that is the first thing to fix (§3, T5);
- it never calls `seamless_config.init()`, never runs the transformation and never materialises the result. Preparic's backend owns configuration, execution and mounts.

CLI-form and factory-built bash transformers can name a worker env, usually conda `alaric` (Py3.12, numpy 1.26). Module-form code runs in `preparic`. Either way, results must not depend on the env, because the env is not part of the cache key (§4).

**Execution (author decision, 2026-10-29).** Preparic's backend runs transformations on eight local spawn workers, with no persistent result cache, hashserver or database. Decisions persist through their mounts (§8.1); results are recomputed after a restart. Only mounted cells survive a restart, and the contract holds across one only if the browser restarts right afterwards (author, O6). Where this document says that a Python transformer with attached modules runs in-process (`mem`), that is Seamless's default. W0's probe found that a transformer with attached modules runs on the spawn workers (V1, `w0-checks/`). Each actual module-form step is still tested on them in its delivery package; a step that cannot run there moves to staged code or CLI + DEPS.

**Large immutable data is identified, not staged.** The conformer/rotamer library (`/opt/fraglib`) and the precomputed Gaussian caches (`<root>/precomputed-filters/gaussian/`) stay where they are. Their identity enters each transformation as a string pin made from committed checksums (`fraglib_identities`; a filter's `precalc` checksums from `gaussian_stacking_filter.json`). Their location enters as a meta-file (`META__FILE__fraglib.yaml`; `META__*` pins are excluded from the checksum) or a fixed path under the repo root, which does not affect the cache key. In this design the worker materialises the data itself, opening the files at the locator paths the meta pin (or fixed path) supplies. Seamless therefore never sees the content and does not check it against the identity pin. That is a consequence of the design, not a Seamless limitation: data passed as real checksum inputs and materialised by Seamless would be resolved, and so verified, by checksum. Here the fitter trusts the installation (`verify_checksums=False`), and T21 verifies the cache files itself (§5.1).

**Wiring.** Bash transformers have no optional pins. Every pin a bash transformer declares must be connected before the node runs; until then Seamless reports it as `unwired`. Folder pins arrive in the job directory as a directory named after the pin. Preparic's deep cells are cast to `folder` (via `deepfolder`) before they are connected.

**Shared pin cells.** These preparic cells feed every library-dependent Alaric step, so all of them see the same values:

| Cell | Value | Consumers |
|---|---|---|
| `rna_pdb` | `pdbs-aa/range-1-aa.pdb` (aareduce `--rna --heavy --nalib --no-renumbering`, never `--manual`) | T5, T6, T7 |
| `exclude_pdb_code` | stage-1 code, lower-cased, always set | T5, T7, T21 |
| `fraglib_identities` | from `refe_best_fit_seamless_support.identities()` | T5, T7, T21 |
| `protein_pdbs` | `pdbs-aa/range-{2..}-aa.pdb` (heavy, `--pdb2pqr`, original numbering) | T13 (one file per instance), T21 |

**Data rules.**
- **No lenient flags.** `--ignore-unknown/missing/reordered` are never passed.
- **No renumbering.** Best-fit PDBs keep template numbering (chain B, residues 1–2). Everything downstream keys on `frag-N-SEQ` and fragment numbers, never on resid or row index.
- **Thresholds are rounded in Alaric, once.** Preparic either receives rounded values from an Alaric transformer or pins Alaric's stdlib-only module. It never re-implements a rounding rule. Alaric accepts with a strict `<`, so the rounded value must be strictly greater than the raw value: the smallest multiple of 0.01 above raw, for RMSDs and energies alike. Under that rule the best-fit pose always passes (−16.5913 → −16.59; an exact −16.59 → −16.58; 0.123 → 0.13). Confirmed by the author. Both `util/refe-thresholds.py` and `alaric/energy_threshold.py` implement it, with tests on exact multiples, negative values and values just below a multiple.
- **Schemas.** Every JSON crossing the boundary carries `schema: 1`. Preparic's consumers raise on an unknown schema, and the error shows as a banner. Alaric bumps the schema on any breaking change. There are no exceptions (author).

## 2. Inventory

"Alaric side" names what preparic consumes from Alaric. Owner = whose code runs. "Planned" means a mini-plan describes it; nothing planned is implemented.

| # | Preparic cell | Stage | Owner | Alaric side | Status |
|---|---|---|---|---|---|
| T0 | `parse_ranges` | 2 | preparic (py) | none | planned (§8.6) |
| T1 | `ranges_to_pdb` | 1–2 | preparic (py) | none | exists; must apply assembly operators (D7) |
| T2 | `pdbs_to_folder` | 2 | preparic (py) | none | exists |
| T3 | `aareduce_ranges` | 2 | preparic (bash, `preparic` env) | its output is Alaric's input format | exists |
| T4 | `detect_stacking` | 3 (and 6) | preparic (py) | `ring-refe/` (D8) | exists; Kabsch normals and reuse planned |
| T5 | `best_fit` | 4 | **Alaric** code | fitter; factory `seamlessify_refe_best_fit` | exists, needs unbound pins |
| T6 | `refe_fragments` | 4 | **Alaric** code | reference-fragment cutter (form open) | planned |
| T7 | `best_fit_tables` | 4 | **Alaric** code | pairs/bases table tools (form open) | planned |
| T8 | `best_fit_thresholds` | 4 | **Alaric** code | `refe-thresholds.py` (form open) | planned |
| T9 | `parse_best_fits` | 4 | preparic (py) | `best-fits.tsv` format | planned |
| T10 | `validate_threshold_edits` | 4 | preparic (py) | none | planned |
| T11 | `clip_fragments` | 4 | preparic (py) | none | planned |
| T12 | `energetics_ligands` | 5 | preparic (py) | none | planned, reshaped |
| T13 | `energetics_range_<k>` (one per protein range) | 5 | **Alaric** code | gridless scorer + reduced DEPS; factory suggested | planned |
| T14 | `merge_energetics` | 5 | preparic (py) | `energies.json` format | planned |
| T15 | `interaction_handoff` | 5 | preparic (py) | none | planned |
| T16 | `energy_thresholds` | 5 | preparic (py) | pinned module `alaric/energy_threshold.py` | planned |
| T17 | `client_model` | 6 | none | none | removed: the browser computes no Gaussian (author, O5) |
| T18 | `detect_stacking_bestfit` | 6 | preparic (py), T4's code | as T4 | new (author decision) |
| T19 | `validate_gaussian_classes` | 6 | preparic (py) | filter names from the JSON | planned |
| T20 | `gaussian_check_jobs` | 6 | preparic (py) | job format of T21 | planned, reshaped |
| T21 | `gaussian_check` | 6 | **Alaric** code | all-filter check tool (form open) + precomputed files | new (author decision) |
| T22 | `merge_gaussian_validation` | 6 | preparic (py) | T21 output format | planned |
| T23 | `validate_refe_anchors` | 6 | preparic (py) | `*_raw` fields of T8, the ovRMSD coverage limit √3 Å | planned, reshaped |
| T24 | `list_fragments` | 2 (used from 4) | preparic (py) | T6's file names (`frag-N-SEQ-refe.pdb`) | new (state-transitions) |
| T25 | `share_range_pdbs_aa` | 3 (shown 4–6) | preparic (py) | none (T3's output, unchanged) | new (author decision) |
| T26 | `structure_id` (name proposed) | 1 | preparic (py) | none | new (state-transitions) |
| T27 | `validate_energy_threshold_edits` (name proposed) | 5 | preparic (py) | none | new (author, D13) |
| T28 | `gaussian_scores` (name proposed) | 3 (shown in 6) | preparic (py) | delivered Gaussian scoring module, `gaussian_stacking_filter.json` | new (author, O5) |

"Form open" means module or CLI + DEPS, with or without a factory, as decided by the implementer (§1, §7). Not preparic transformers, and outside this boundary: Alaric's energy-grid Seamless wrapper (grids serve only the `energy-filter` action; preparic's thresholds are gridless) and the Gaussian cache generator (§5.1).

## 3. Transformer by transformer

### Stages 1–3 (exist)
- **T0 `parse_ranges` (planned).** Preparic only. It parses the `ranges` YAML into `range_specs`, so the backend derives `range_specs` instead of the frontend writing it (§8.6). It also takes S2 `biounit` and records it in `range_specs`, because the extracted atoms depend on it (D7). Every echo that carries the T0-normalised range rows therefore covers the biounit too.
- **T1 `ranges_to_pdb`, T2 `pdbs_to_folder`.** Preparic only. Range 1 is RNA, the rest protein. An RNA range in which any residue is unobserved is refused, by the browser's row validation and by T1, with a diagnostic naming the missing residues (author, D15). T1 applies the selected assembly's operators from the CIF: a range picked on a copy such as `A-2` is extracted with that copy's operators, where it was picked, and validated against the same assembly as in the browser (author, D7). Today it strips the suffix and extracts asymmetric-unit coordinates (`range_pdbs.py`).
- **T3 `aareduce_ranges`.** Preparic only, but its output **is** Alaric's input contract. `range-1-aa.pdb` must have no unknown residues, no missing heavy atoms and topology atom order, matching `fraglib/templates`. At export it is copied byte-identically to `DATA/reference.pdb` (anchor-refe compares against that file). From stage 4 on, its output is also exactly what the frontend displays (T25). Known limitation: modified nucleotides fail here.
- **T25 `share_range_pdbs_aa` (new; author decision).** Preparic only. Copies every file of B3 (`range-{1..K}-aa.pdb`) into the plain cell S23, so the browser can display them; folder cells cannot be shared (§8.1). The texts are copied byte for byte: no parsing, renumbering, filtering or reformatting. It also takes S5 `range_specs` and B5 `exclude_pdb_code` and echoes them, so the frontend can tell whether the PDBs belong to its current ranges and PDB code (§8.4). Output `{schema:1, pdb_code, range_specs, pdbs:[{range, name, pdb}]}`, in range order. It needs no library and is not gated on stage 3, so it is ready as soon as T3 is.
- **T26 `structure_id` (new; name proposed).** Preparic only. It reads the entry id from S3 `cif` into S21 `structure_id`, the consistency check between the stored structure and S1 (§8.6.1).
- **T4 `detect_stacking`.** Preparic only, with SVD ring normals today. It switches to Kabsch ring normals, fitted onto Alaric's `ring-refe/refe-{A,C,PHE,HIS,ARG,TRP}.pdb`, so that stage 3 and stage 6 descriptors agree (author, D8; `open-questions.md` stage 6 Q5). Those files become checksum-pinned inputs of T4, the same files T17 and T21 pin. That is a data dependency only; no Alaric code runs. Stage 3's numbers change by up to 0.7°.

### Stage 4: best fit and RMSD thresholds
- **T5 `best_fit` = `seamlessify_refe_best_fit` (exists).** Changes: `rna_pdb=None` leaves an unbound `rna.pdb` pin (bytes); `exclude_pdb_code` becomes an unbound str pin instead of a baked argument. Keep the meta-file and `fraglib_identities` pins as they are.
  - Output folder: `best-fits.tsv` (`fragment sequence conformer rotamer grid_x grid_y grid_z rmsd`) + `pdbs/frag-N-SEQ-bestfit.pdb`. The TSV columns and the PDB naming are contract; T9, T13 and T21 read them.
- **T6 `refe_fragments` (new Alaric code; form open).** Cuts the reference into one PDB per dinucleotide with the templates' atom order. Input `rna.pdb` plus the 20 templates. Output folder `frag-N-SEQ-refe.pdb`. The `templates=` argument for `Reference` is deferred by the author. As a CLI it can reuse the best-fit wrapper's approach (meta-file plus `templates_dir = "."` over staged templates). As a module the templates are function arguments.
- **T24 `list_fragments` (new).** Preparic only. It lists the fragment keys of range 1, and so the pair and base rows, from T6's output into S22 `fragments`. Neither T6 nor T24 waits for the fit or for stage 3, so S22 is ready in stage 2 and stage 4 opens before the fit (author, `final/state-transitions.md` §11.7). It takes B5 and the range-1 row and echoes them (§8.4).
- **T7 `best_fit_tables` (new Alaric code; form open).** Runs the logic of `refe-best-fit-pairs.py` and `refe-best-fit-bases.py` in one step. Inputs: `best_fits` (T5's folder, or just its TSV), `rna.pdb`, `exclude_pdb_code`, `fraglib_identities`, plus the `fraglib.yaml` meta-file. Output folder `{pairs.tsv, bases.tsv}`. It needs the installed library, so it is closest to T5. The `crmsds` key must exist in `fraglib.yaml` even though cRMSD data is not read.
- **T8 `best_fit_thresholds` (new Alaric code; form open).** Stdlib-only `util/refe-thresholds.py`, which holds the RMSD rounding rule (§1). It is a natural module-form candidate. Input: T7's folder. Output: plain JSON `{schema, pairs:[{down,up,sequence,ovrmsd,crmsd,*_raw}], bases:[{fragment,sequence,nuc1_ovrmsd,nuc2_ovrmsd,*_raw}]}`. The `*_raw` fields are contract: T23 reads them.
- **T9–T11.** Preparic only. They parse T5/T8 output and apply edits and clips.
  - Before stage 5 the threshold table must be complete for its kept rows: every pair and base value of a kept row filled, by Copy and/or typing. Rows of clipped fragments are not required (author; `open-questions.md` stage 4 Q2, `final/state-transitions.md` §11.5, D4).
  - T10 validates S11 against S22, not S9, so it does not wait for the fit. S9 still feeds Copy, the "differs from computed" marks, T23 and the export's raw-value comparison.

### Stage 5: energetics
- **T12 `energetics_ligands`.** Preparic only. It joins T6's folder and T5's `pdbs/` into one `ligands` folder; the names (`frag-N-SEQ-refe.pdb`, `frag-N-SEQ-bestfit.pdb`) are already distinct.
- **T13 `energetics_range_<k>`, one instance per protein range (new; author decision).** CLI form `python -B energetics_seamless_support.py score ligands receptor.pdb` with the reduced DEPS file (§4). A factory `seamlessify_energetics(*, epsilon=15.0, conda_environment=None)` returning the bash transformer fits well here, because every range uses an identical transformer. The implementer may still choose otherwise.
  - Inputs (unbound pins in the factory form): `ligands` (folder, T12) and `receptor.pdb` (bytes, `range-<k>-aa.pdb` from `protein_pdbs`). An edit to one range rescores only that range; a refit rescores every range.
  - Preparic adds or removes instances when `range_specs` changes. That reconciliation is preparic-internal: every range gets the same transformer. Unbound receptors (deferred) will later be swapped in at the `receptor.pdb` pin, as extra instances.
  - Code dependencies: the files listed in the **reduced DEPS file** (§4), which include `alaric/reduce.dat` and `attract-jax/attract-original/attract-par.npz`.
  - The worker reduces the receptor once and every ligand to ATTRACT beads (`parse_pdb` → `reduce-npy.py`). It then scores each ligand with `JaxScoreOracle(backend="numpy", grid_free=True, energy_only=True)`, identity pose, score only. The best-fit ligand is the discrete library pose T5 writes, not an exact superposition (author, D10).
  - Output `energies.json`: `{schema:1, epsilon, receptor:{beads}, ligands:{name:{beads}}, pairs:[{ligand, energy, error}]}`. The range index is not inside the result (the cache key then does not depend on it); preparic knows it from the instance. A failed reduction sets `error` on that ligand only; a receptor that fails to reduce fails the job.
  - Must run in the `preparic` env as well as in `alaric`: no JAX, no compiled kernel, numpy 1.x or 2.x.
- **T14 `merge_energetics`, T15 `interaction_handoff`.** Preparic only. T14 collects the per-range `energies.json` files into the `energetics` rows.
- **T16 `energy_thresholds` (reshaped: the computed values; author, D13).** Preparic transformer, Alaric code: it attaches `$ALARIC_DIR/energy_threshold.py` (stdlib only: `decimal`, `json`) as a Seamless module (`.modules`), so the module source is part of the checksum, and calls its rounding function. A Python transformer with modules runs in the `preparic` env (in-process by Seamless's default; on the spawn workers, see §1 Execution). Being stdlib-only, `energy_threshold.py` fits there without extending the env; keep it that way. Inputs: `energetics` and `best_fits`. The matrix and the clip are no longer inputs, so a matrix edit does not re-run it. Output S15: the **computed** threshold, R(best-fit energy), for every (fragment, protein range) cell that has a best-fit energy, as two-decimal strings, with S13's echo. Thresholds come from **best-fit** energies (author decision); refe energies are shown only. S15 is the Copy source of the editable energy-threshold table S24, and what its "differs from computed" marks compare against (T27).
  - **The exported file.** The export writes `energy-thresholds.json` from S24, not from S15: `{"schema":1, "frag-N-SEQ":{"range-K":"-16.59"}, …}`. It holds the schema key of §1 plus the threshold mapping at top level, true kept matrix cells only, with the user's values as two-decimal strings. There is no other key (author decision). Every true cell has an energy: the matrix editor forbids, and read validation refuses, a true cell on a row with a scoring error (author, D12). This file is the input to Alaric's planned `energy-filter` action (`threshold: auto`), so its shape is a two-way contract. Alaric raises unless `schema` is 1, every other key is `frag-N-SEQ`, every inner key is `range-K`, and every value is a two-decimal string.
- **T27 `validate_energy_threshold_edits` (new; author, D13; name proposed).** Preparic only: the stage-5 counterpart of T10.
  - It validates S24 against S22 and S5. It accepts null, and leaves stale parts to Continue 4→5: a whole S24 whose basis is not S22's echo, and an entry whose `row` matches no current protein range. Otherwise every key must be a fragment key of S22, and every value null or a two-decimal string (any sign). It does not wait for scoring.
  - Before stage 6 the energy-threshold table must have a value for every true cell of the matrix, by Copy and/or typing. False and clipped cells need not be filled; they are greyed but stay editable.
  - Its result is not shared. Gate 5→6 needs it not failing.
  - **Energy-filter parameters added at export time.** The former wrapper fields (`method`, `epsilon`, `source`, `rounding`) no longer travel in the file. The export writes these parameters into each `energy-filter` action instead:
    - `protein: range-K`: one action per true cell (fragment, range). A fragment with several ranges gets one action per range, chained (AND).
    - `threshold: auto`: the action reads its cell from `DATA/energy-thresholds.json`. A missing cell is an error.
    - `epsilon: 15`: the ε that T13 scored with. The thresholds are valid only for that ε.
    - `cascade: auto` and `min_cascade: auto`: both read `DATA/score-filter.yaml`, which the **export stage** writes (it is not a preparic cell in stages 1–6). Its content must be `min_cascade: 1000` and `cascade: [{plateau_distance: 5, neighbour_distance: 7, max_error: 1.0}, {plateau_distance: 10, neighbour_distance: 12, max_error: 0.05}]`. Each cascade step scores the undecided poses with that grid: grid energy < T − `max_error` accepts a pose, ≥ T + `max_error` rejects it, and anything in between leaves it undecided. When fewer than `min_cascade` undecided poses remain in the action (counted over all chunks), the cascade is exited. Every pose still undecided is then decided gridless with T13's kernel, which replaces `method: "gridless"`.
    - `fragment`, `sequence` and `exclude`: `auto`, with `exclude` resolving to the project's `exclude_pdb_code`.
    - `source: "bestfit"` and `rounding` were provenance only, and no action needs them.

### Stage 6: Gaussian filter and refe anchors
- **T17 `client_model` (removed; author, O5).** It built a browser model (mean, precision, normalizer, cutoffs, golden vectors). The browser now computes no Gaussian, so it needs none; T19 and T28 read B15 themselves.
- **T18 `detect_stacking_bestfit` (new, replaces `ring-geometry.ts`).** The author accepted running T4's code on the best-fit conformers instead of a TypeScript descriptor module. Same code as T4, other inputs: the best-fit PDBs plus the protein ranges. T18 gets T4's Kabsch normals, so stages 3 and 6 agree (D8). T28 scores its descriptors.
- **T28 `gaussian_scores` (new; author, O5; name proposed).** Preparic transformer, Alaric code: it attaches the Gaussian scoring module delivered with A7/A8 and reads A7's JSON (B15). The browser computes no Gaussian; it reads S25.
  - Inputs: S17 (best-fit descriptors), S7 (reference descriptors), S8 (selection), S12 (clip) and B15.
  - It raises unless the JSON has `schema: 1` and all twelve names. On every run it checks the JSON's golden vectors at relative 1e-12 and its accept/reject goldens; a mismatch fails it, and stage 6 stays closed.
  - Output S25: the Gaussian score of every best-fit candidate and of every reference stacking. For each selected anchor, the computed class of `oracle-contract.md` P6.3 (class score of the worst unclipped candidate, RRM class, auto prefilters, filter name) and its reference score. Also `filter_set_version` and the twelve names, for the class editor.
  - The applied mask (d < 5 Å, φ < 60°) and the five-decimal cutoffs come from the JSON. On 1b7f its scores equal T21's `score` to 1e-6; this is a backend test, not a runtime comparison.
- **T19 `validate_gaussian_classes`.** Preparic only. Valid names are the JSON's `thresholds` keys, read from B15, never a hard-coded list.
- **T20 `gaussian_check_jobs` (reshaped).** Instead of 12 per-filter job lists, preparic emits **one** job list: one job per unclipped best-fit dinucleotide of each selected anchor (`{key, fragment, nucleotide, sequence, conformer, rotamer, grid:[x,y,z], range, protein_resid}`, taken from T9's rows). Because the job list does not depend on the class table, editing a class never re-runs T21.
- **T21 `gaussian_check` (new Alaric code; form open; §5).** One transformation checks all jobs against all 12 filters.
- **T22 `merge_gaussian_validation`.** Preparic only. It combines T21's per-filter results with the class table, so class edits are instant, and can suggest the strictest filter that still emits each best-fit cell. The class table has one row per selected stacking anchor, scored on the worst of its unclipped candidates (author, D9).
- **T23 `validate_refe_anchors` (reshaped).** Preparic only; no Alaric code and no library. The author ruled that both anchor modes are always exported in parallel, so the `anchor_modes` cell disappears and every unclipped candidate is checked against its edited bases threshold. Alaric constants it mirrors: the ovRMSD coverage limit of grow's translation-offset tables, √3 ≈ 1.73 Å (above it the export is **blocked**, by author decision; `MAX_OVERLAP_RMSD = 1.3` in `grow.py` is only the bound of the loop that builds those tables, `oracle-contract.md` A11), and the boundary-recompute tolerance (`grow.py:35`). The block also covers `grow`'s pair ovRMSD values (author, D11); T23 does not check those, the export does. The remedy for a value above the limit is to lower it, which may fail the best-fit pose and then needs an explicit "Proceed anyway". Alaric raises on an `ovrmsd` above √3 instead of clamping (author, D11).

## 4. The reduced DEPS file for gridless scoring

`attract-jax/util/minfor.py.DEPS.txt` lists everything `minfor.py` may load, including the JAX kernel, grid generation and the whole `nb_kernel/` tree. Gridless numpy scoring needs a fraction of it, verified from the imports: `JaxScoreOracle(backend="numpy", grid_free=True)` imports `pdb_io` at module level and `numpy_grid_score` lazily. `grid_io` is imported only for grid files, `reproduce_grid_score`/`jax` only for `backend="jax"`, and `forcefields`/`nb_kernel` only for the compiled kernel.

**File:** `util/energetics_seamless_support.py.DEPS.txt` in Alaric. The `<script>.DEPS.txt` name is Alaric's own convention (`alaric/anchor.py.DEPS.txt`); Seamless does not pick such files up automatically. The format is what `seamless-run -I FILE` reads: one path per line, relative paths resolved against the list file's own directory. There is no comment syntax, and a blank line is not skipped but becomes the list's directory, so the file must contain neither.

```
energetics_seamless_support.py
../alaric/parse_pdb.py
reduce-npy.py
../alaric/reduce.dat
../attract-jax/util/jax_scorer.py
../attract-jax/util/numpy_grid_score.py
../attract-jax/util/pdb_io.py
../attract-jax/attract-original/attract-par.npz
```

**Rules.**
- Alaric maintains it, not attract-jax: the subset belongs to Alaric's use of attract-jax, and the attract-jax submodule pointer is Alaric's to bump.
- `input_paths()` in `energetics_seamless_support.py` reads this file. It does not keep a second Python list, so `seamless-run -I` and any factory pin exactly the same files.
- **Sandbox test** (`tests/test_energetics_deps.py`), the §1 test for form (b): run `seamless-run -I util/energetics_seamless_support.py.DEPS.txt` on the scorer, with `jax` and `scipy` imports blocked (the attract-jax test already does this for JAX), and score one 1b7f fragment against `1b7f_dom1-aa.pdb`. It must import nothing outside the list and reproduce `minfor.py --oracle numpy --no-grid --identity` to 1e-6. A second check asserts that every attract-jax entry also appears in `minfor.py.DEPS.txt`, so a refactor in attract-jax that adds an import breaks the test instead of the cache. W0 found this check would fail because `attract-par.npz` was absent; I8 was fixed by adding it to attract-jax's full DEPS list.
- The listed attract-jax files must come from Alaric's pinned submodule. Its committed pointer is still `12bba50`, without the numpy path. The author gave the go-ahead (D16): publishing the numpy path and its accuracy data, and bumping the pointer, are a prerequisite task of `draft-implementation-plan.md` (W4).
- Changing any listed file changes the energetics cache key, because each becomes a `bytes` pin. That is intended, and it is why the list is kept minimal: edits to the grid or JAX paths do not invalidate preparic's energies. One trap: when a `<file>.CHECKSUM` sidecar exists, Seamless reads the checksum from it instead of hashing the file, so a stale sidecar hides an edit. No sidecars for the listed files.
- The basenames are unique, so they stage flat (the best-fit factory's collision check applies), and `jax_scorer.py` still finds `pdb_io.py` and `numpy_grid_score.py` beside it.
- The conda environment is not part of the cache key. A result computed in `alaric` (numpy 1.26) is served to a request that names `preparic` (numpy 2.5) and the other way round. The gridless kernel is float64 and agreed across both envs to the printed precision, so this is acceptable. If that ever changes, a result-affecting dependency must become a pin, not an environment choice.
- Hard-coded paths must go: `reduce-npy.py` finds `reduce.dat` next to `alaric/`, so the worker loads it through `importlib` with the staged path. `attract-par.npz` is passed explicitly, never resolved relative to the script.

## 5. Precomputed Gaussian filters and the check tool

### 5.1 Files (Alaric side, not Seamless)
- **Generator.** By author decision, the 12 caches are neither committed nor produced on demand by Seamless. An Alaric generator script (`util/gaussian-precalc.py NAME --output DIR`, and a wrapper that builds all 12) writes them to `<root>/precomputed-filters/gaussian/<name>/{rotations,translations,filtered-sizes,filtered-concat}.npy.zst`. That folder is in `.gitignore`. The applied mask, d < 5 Å and φ < 60°, both strict, is applied at this stage (author). The report's d < 5.7 Å envelope is provenance only. The cutoffs are the report's values rounded down to five decimals, never re-derived (author; `final/gaussian-precomputation-decisions.md` §3). Every prefilter variant keeps its parent's Gaussian-only cutoff (`rrm90-tilt30` uses `rrm90`'s 0.27592), with the prefilter added on top, so its recall falls below the parent's by definition. The variant-specific cutoffs in `alaric-gaussian-miniplan.md` §2 (0.229, 0.270, 0.164, 0.029) are a misreading of the report and must not be used.
- **Compressed or not.** Each cache file may be present as `.npy.zst` or as decompressed `.npy`. The JSON's `precalc` carries both `sha256` (the `.zst` bytes) and `npy_sha256` (the decompressed array), so either form is verifiable. Decompressing is an installation-time choice, not part of the contract.
- **The JSON is the contract.** `alaric/gaussian_stacking_filter.json` (committed) holds the model, the envelope, the 12 names with cutoffs, masks, `accept: ">="` and per-file `sha256`/`npy_sha256`, plus golden vectors. Preparic reads only the JSON and the `ring-refe/` PDBs. It never opens the caches itself; only T21's worker does, and it verifies the checksums before use.
- **Installation check.** A missing or mismatching cache is an installation error that T21 raises, with the generator command in the message. Preparic shows it as a banner. Nothing is regenerated silently.
- **Runtime parity.** `MIN_CELL_FRACTION` in `anchor.py` can be overridden by `ALARIC_ANCHOR_MIN_CELL_FRACTION`. T21 takes it as a pinned argument (default ⅓), and the export must not set the variable, or validation and the anchor run disagree.

### 5.2 The check tool (T21)
Requirement (`open-questions.md`, stage 6 Q4): check a small list of anchor poses against all filters within a few seconds, using the filter hierarchy for early cutoff. The delivery form (module or CLI + DEPS, factory or not) is open (§1). It needs the installed library and the Gaussian caches (read via locator paths), and numpy/zstd for the arrays.

- **Project inputs:** `jobs` (plain, from T20), `best_fit` (T5's folder, for the reconstruction sanity check), `proteins` (folder, `protein_pdbs`), `exclude_pdb_code`, `fraglib_identities`. **Fixed inputs:** code, the JSON (which carries all 12 cache identities), `ring-refe/*`, `min_cell_fraction`.
- **Per job and filter,** the quantities from `preparic-gaussian-miniplan.md` §7.1: `count`, `capacity`, `fill`, `emitted = fill ≥ min_cell_fraction`, `pose_pass` (score ≥ the JSON's five-decimal cutoff, inside the applied mask d < 5 Å and φ < 60°, and passing the strict prefilters), `score`. Plus a per-job `error` (incomplete ring, reconstruction RMSD > 0.002 Å).
- **Hierarchy, derived from the JSON, not hard-coded.** Filter A ⊆ B as a set of (rotation cell, translation) pairs iff A's masks include B's masks and `cutoff_A ≥ cutoff_B`. Both `count` and `pose_pass` are monotone under ⊆ (capacity does not depend on the filter). So:
  - if B does not emit a job's cell, no A ⊆ B emits it either, and those entries are returned as `{emitted:false, implied_by:B}` without loading A;
  - if A emits it, every B ⊇ A does too.

  Because every variant keeps its parent's cutoff, the hierarchy is a clean lattice:
  - `rrm50 ⊂ rrm60 ⊂ rrm70 ⊂ rrm80 ⊂ rrm90 ⊂ rrm95`;
  - for N ∈ {90, 95}: `rrmN-tilt30-disp45 ⊂ rrmN-tilt30 ⊂ rrmN` and `rrmN-tilt30-disp45 ⊂ rrmN-disp45 ⊂ rrmN`;
  - each 90 filter lies inside its 95 counterpart (`rrm90-disp45 ⊂ rrm95-disp45`, and so on).

  The tool walks from `rrm95` downwards. On each branch it stops at the first non-emitting filter and marks everything below it as implied. A cell that `rrm95` does not emit therefore costs one check.
- **Speed.** The few-seconds target excludes decompression. Time spent decompressing `.npy.zst` files (about 1–3 s per large filter) does not count, and is zero when the decompressed `.npy` files are installed. What must fit the budget is the check itself: loading the arrays, the capacity table, the library slice, and milliseconds per (job, filter).
- **Output:** `{schema:1, filter_set_version, min_cell_fraction, jobs:[{key, fragment, nucleotide, error?, filters:{<name>:{count, capacity, fill, emitted, pose_pass, score, implied_by?}}}]}`.
- **Code sharing.** The logic comes from `alaric/anchor_precalc.py` (`candidate_cells`, `cell_fill`), the pure-move refactor of `anchor.py`, so the check and the anchor action use the same code. Guard tests: `tests/1b7f-gaussian-anchor` stays bit-identical, and `cell_fill/capacity ≥ ⅓` holds exactly for every pose an anchor run emits.

## 6. Where this overrides the mini-plans

| Mini-plan says | Now (author decision) |
|---|---|
| Stage 6: browser descriptors in `ring-geometry.ts` | backend T18 reuses `detect_stacking.py` |
| Stage 6 (and `open-questions.md` stage 6 Q6): Gaussian scores and computed classes in the browser, from a browser model (T17) | computed in the backend by T28 from T18's and T4's descriptors; the browser only reads them, and T17 goes (author, O5) |
| Stage 6 §7.3: 12 static validator instances, re-run per edit | one T21 over all filters, which never re-runs on class edits |
| All mini-plans: a `seamlessify_*` factory for every Alaric step | Alaric delivers a sandbox-tested module (preferred) or CLI + DEPS; a factory is a case-by-case choice (suggested for T5, T13) |
| Test-anchor: `anchor_modes` cell, mode toggle | no mode choice; both modes are exported, and T23 checks every anchor |
| Test-anchor: thresholds > 1.73 Å warn | block the export; `grow` pair ovRMSD values too (D11) |
| AlaGaussian §7: Seamless factory and materialiser for caches | generator script, gitignored folder; no Seamless precalc factory is needed by preparic |
| AlaGaussian §2–3: a separate cutoff per prefilter variant, RRM-95 fallback | every variant uses its parent's Gaussian-only cutoff with the prefilter on top, so variants nest inside their parents |
| Energetics: two instances (refe, best-fit), each over all ranges | one instance per protein range, scoring both sets |
| Energy-filter §4: `energy-thresholds.json` is `{schema, method, epsilon, source, rounding, thresholds}` | `schema` plus the `thresholds` mapping, at top level; `epsilon`, `cascade` and the rest become `energy-filter` parameters written at export (T16) |
| Energy-filter §3–4: `cascade: [5-7, 10-12]` with margins 1.0/0.1 fixed in code; grid steps accept, reject or refine | `cascade` is a list of `{plateau_distance, neighbour_distance, max_error}` plus `min_cascade`; `auto` reads `DATA/score-filter.yaml`, written at export (defaults 5-7/1.0, 10-12/0.05, 1000); grid steps accept below T − `max_error` and reject at or above T + `max_error` (T16) |
| Energy-filter §4: `score`/`filter`/`score_add` kept | removed. `score-single.sh`/`score-fragments.sh` move to the gridless path or go, and they must then use the §4 DEPS list |
| Stage 4: `templates=` for `Reference` | deferred; T6 uses the staged-templates override |
| Energetics and energy-filter plans: energy thresholds computed only | user-editable, in a table that works like stage 4's: empty on entry, filled by explicit Copy from S15 and/or typing, a value for every true kept matrix cell before stage 6; one range's values reset only when that range changes (D13, S24, T27) |
| Stage 4 Q2: copy required only at export | the table's kept rows must be complete before stage 5, by Copy and/or typing; rows of clipped fragments are not required (`final/state-transitions.md` §11.5; D4) |
| Energetics EN:33, EN:88: `ClipControls` shared by stages 4–6; the matrix rewritten on every clip edit | the clip is edited in stage 4 only, and the kept matrix is adapted once, on Continue 4→5 |
| AlaGaussian §1, §5: envelope d < 5.7 Å as the mask | applied mask d < 5 Å and φ < 60°; d < 5.7 Å is provenance only |
| AlaGaussian §2: full-precision cutoffs derived from the CSVs | the report's values, rounded down to five decimals; no re-derivation |
| Stage 4 mini-plan, architecture: the viewer shows the RCSB structure throughout, with fragment overlays on top | from stage 4 on, the viewer shows only the backend's aareduce PDBs (S23), with fragment overlays on top |
| Stage 4: plain `ceil` to 0.01; energy-filter §4: round half away from zero | smallest multiple of 0.01 strictly above raw, for RMSDs and energies, with Alaric's strict `<` |

Stage 4 lines 3 and 42 (energetics consumes clipped fragments) were already flagged as wrong by the energetics plan. `alaric-project-layout.md` still claims that best-fit tables feed DATA thresholds; the author asked for that to be corrected.

## 7. Remaining points at the boundary
1. **Delivery form per Alaric step (implementer's decision, case by case).** For T6, T7, T8 and T21 the implementer chooses between a module with a Python function (preferred) and a CLI tool with a DEPS file. For the CLI form, they also choose between preparic building the bash transformer from the `seamless-run` command and Alaric providing a factory. Guidance from §1:
   - prefer a module whenever its dependencies fit the `preparic` env, which may be extended within limits (no JAX, nothing that pins numpy < 2);
   - choose a CLI + DEPS when they don't, or when the code is naturally a script with many file dependencies;
   - add a factory only when turning the `seamless-run` invocation into a bash transformer is not trivial (staged templates, meta-files, identity pins, nested dependencies), as it was for the best-fit fitter.

   T5 has its factory already. T13 is suggested as a factory but is not bound to it. Every form must pass the Seamless sandbox test.
2. **Stage 3 Kabsch** (author, D8; stage 6 Q5): T4/T18 use Kabsch normals and pin `ring-refe/`, and the stage 3 numbers change.
3. **Export stage** (deferred): it will consume T3's PDB, T8 edits via `clipped_fragments`, `interactions.json`, the S24 energy-threshold edits (written as `energy-thresholds.json`), the class table and T21/T23 results, and write `DATA/` and the actions, including `DATA/score-filter.yaml` and the `energy-filter` parameters listed under T16. It must use the same `exclude_pdb_code` and `min_cell_fraction` as the transformers above. Its red-text "proceed anyway" checkbox is computed in the frontend.

## 8. Frontend ↔ backend boundary: shared cells

§§1–7 cover where preparic's backend calls Alaric. This section covers the other boundary in preparic: the one between its Svelte frontend and its Seamless backend. It lists every shared cell that exists or is planned for stages 1–6, and what each one is to the transformers of §2. Existing facts come from preparic HEAD `5c85f67` (`backend.py`, `src/lib/seamless.svelte.ts`). Seamless behaviour comes from `seamless/docs/agent/contracts/shares.md` and `guides/seamless-web-server.md`. How the stage is derived from these cells, and every transition, is `final/state-transitions.md`.

### 8.1 Rules
- **Mechanism.** Preparic uses the Seamless share server only, with no custom routes.
  - The namespace is `preparic`, and the server listens on 127.0.0.1:5813. Vite proxies `/preparic` (including the websocket) and `/seamless-client.js` to it.
  - The browser reads values with GET. The websocket announces changes as checksums and markers, never as values.
  - Writes are conditional PUTs (`?marker=`), sent by `seamless.svelte.ts`, which also handles pending writes, revisions and loop avoidance. A `409` means the browser drops its value and takes the server's.
  - The share key is the cell's path, so cell names in the table below are part of the contract.
- **Edit cells and computed cells are disjoint.**
  - An *edit cell* is a writable share with no incoming edge. Seamless enforces this: `readonly=False` on a connected cell raises `AuthorityError`.
  - A *computed cell* is a read-only share of a transformer's result.
  - So the backend never writes an edit cell once it has started. Every reset below is a frontend write. The only other writers are the startup loader and out-of-band edits of the mounted file.
- **Persistence.**
  - Every edit cell that holds a user decision is mounted `authority="file"` under `deployment-mounts/`. That includes S3 `cif` (author decision), so no edit cell needs a browser after a restart.
  - Its startup loader rewrites malformed content to the cell's undecided value (null for a marker) before the cell is shared. A writable share on an empty cell would otherwise install null. The loader keeps the malformed content's diagnostics before it resets the cell (author, I7).
  - A stored marker without a basis, such as one written before bases existed, is malformed: it loads as undecided, the user is told why, and it is never migrated (author, I7).
- **Undecided is null, and stage decisions are markers.**
  - S4 `ranges`, S8 `stacking_selection`, S11 `threshold_edits`, S14 `interaction_matrix` and S18 `gaussian_classes` are the stage markers of stages 2–6.
  - Each is either null (its stage was never entered) or `{basis, active, …decision}`. Their loaders write null.
  - `[]` is no longer an empty value, so a genuinely empty result is never mistaken for "undecided".
  - `basis` records the upstream values the decision was made against. A decision whose basis no longer matches is invalid, and the stage falls back.
  - `active: false` means the stage was left with Back. The decision is kept, so that a later Continue can restore it (author). Consumers treat an inactive marker exactly like null.
  - S12 `best_fit_clip` is a stage-4 parameter, not a marker. It always holds `{clip1, clip2, basis}`, with loader default `{0,0}` and basis `pdbcode` plus the range-1 row. It never clips a candidate fragment of a selected anchor (author, D2a); Continue 3→4 clamps a kept clip that does, automatically and possibly silently (author, D2b; the only exception to G9 of `oracle-contract.md`).
  - S24 `energy_threshold_edits` is a stage-5 parameter, not a marker (author, D13). It holds `{schema:1, basis:{pdbcode, range-1 row}, ranges:[{row, values}]}`, with loader default null.
    - Each entry belongs to the protein range whose T0-normalised row equals its `row`, whatever that range's index.
    - Continue 4→5 starts S24 as an empty skeleton when it is null or its basis is stale. Otherwise it drops the entries of ranges that no longer exist, and adds empty ones for new ranges.
    - It is filled by explicit Copy from S15 and/or typing, and stays editable in stages 5–6.
  - Continue writes the next stage's marker: kept, merged by key, or by default. The defaults are:
    - the stacking merge;
    - a threshold skeleton with every value empty, filled by explicit Copy and/or typing (author);
    - the kept matrix adapted to the clip, or Recompute;
    - the computed classes.
  - A computed cell with no value means "pending". It is never a reason to write an edit cell.
- **Celltypes.**
  - Every planned shared cell is `plain` JSON.
  - Folder and deep cells cannot be shared (`TypeError`). Anything the browser needs from a folder result must therefore be copied into a plain cell by a preparic transformer. For example, T9 copies the fragment PDB texts into the `best_fits` rows, which `clip_fragments` also reads.
  - Each value travels whole, with no range requests. All planned computed cells are small, at most tens of kB, except S23, which holds the whole all-atom PDBs of every range (typically a few hundred kB). It changes only when the ranges or the structure change.
- **Validation happens twice.**
  - Seamless checks a PUT only against the celltype.
  - The frontend validates inline and again when it reads a value. It refuses bad values, shows a banner, and never repairs them silently.
  - Backend validators (T10, T19, T27) check again and raise. Every validator accepts null and an inactive marker. A failed T10 keeps gate 4→5 closed, and a failed T27 keeps gate 5→6 closed (author; kept after I4 rejected validation receipts).
- **The set of shares is fixed at backend start.** The per-range T13 instances come and go with `range_specs`, but none of them is shared, because T14 merges them. The frontend's `onsharelist` attachments therefore stay static.

### 8.2 Shared cells
"rw" means the frontend writes the cell; "ro" means it only reads it. Mounts are under `deployment-mounts/`:
- "file" means the mount is file-authoritative, with a startup loader where one is given.
- "cell→file" means the file is written by the cell and never read back.

T-numbers refer to §2.

| # | Cell (celltype) | Share | Mount | Output of | Input to | Frontend | Status |
|---|---|---|---|---|---|---|---|
| S1 | `pdbcode` (text) | rw | `pdbcode.txt`, file | none | none today; planned: source of `exclude_pdb_code` (B5) | stage 1 writes it only after the CIF download succeeds, together with S3 (§8.6.1), and reads it back on load | exists; gains a consumer; write is delayed |
| S2 | `biounit` (plain int) | rw | `biounit.json`, file | none | T0 (D7) | stage 1 | exists; gains a consumer |
| S3 | `cif` (text), key `cif.cif` | rw, not auto-read | file (author decision) | none | T1, T26 | stage 1 writes it after Mol*'s RCSB download, together with S1 (§8.6.1) | exists; becomes mounted |
| S4 | `ranges` (yaml; marker `{basis:{pdbcode,biounit}, active, rows}`) | rw | `ranges.yaml`, file, loader → null | none | planned T0 | stage 2 reads and writes it; this is the only range cell the frontend writes | exists; gains a consumer; becomes a marker |
| S5 | `range_specs` (plain) | ro | none | planned T0 | T1; planned T9, T14, T15, T20, T23 | not read | exists; becomes computed (§8.6) |
| S6 | `range_pdbs` (plain `list[str]`) | ro today; to be unshared | none | T1 | T2, T4; planned T18 | not read | exists; share removed (§8.6) |
| S7 | `stackings` (plain) | ro | `stackings.json`, cell→file | T4 | planned T15¹, T19, T20, T23, T28 | stage 3 table; stages 5–6 derive anchors from it | exists |
| S8 | `stacking_selection` (plain; marker `{basis: range rows, active, selected:{key: bool}}`, keyed by stacking key) | rw | `stacking-selection.json`, file, loader → null | none | T5 (group B starts only while it is active, author); planned T15¹, T19, T20, T23, T28 | stage 3 writes it; Back 3→2 deactivates it; stages 5–6 read it | exists; becomes a keyed marker (was `list[bool]`) |
| S9 | `best_fit_thresholds` (plain) | ro | `best-fit-thresholds.json`, cell→file | T8 | T23 | stage 4: the computed tables, clearly visible beside the editable one so that it is clear what Copy does (author), the copy source and the "differs from computed" marks; stage 6 refe panel | planned |
| S10 | `best_fits` (plain rows, PDB texts included) | ro | none | T9 | T11, T14, T15, T16, T18, T20 | stages 4–6: row order, Mol* overlays | planned |
| S11 | `threshold_edits` (plain; marker `{schema:1, basis:{pdbcode, range-1 row}, active, pairs, bases}`) | rw | `best-fit-threshold-edits.json`, file, loader → null | none | T10, T11, T23 | stage 4 creates it as a skeleton and fills it by Copy and/or typing; its kept rows must be complete before stage 5 (D4); editable in stages 4–6; the stage 6 refe panel edits `bases` | planned |
| S12 | `best_fit_clip` (plain `{clip1, clip2, basis}`) | rw | `best-fit-clip.json`, file, loader → `{0,0}` | none | T11, T15, T20, T23, T28 | edited in stage 4 only, where the control stops at the outermost selected anchors (author); read-only in stages 5 and 6; Continue 3→4 resets it when its basis is stale, and clamps it to the outermost selected anchors when it cuts one (author, D2b) | planned |
| S13 | `energetics` (plain rows) | ro | none | T14 | T16 | stage 5 display and Recompute | planned |
| S14 | `interaction_matrix` (plain; marker `{basis: selected keys, active, rows, ranges, grid}`: `rows` are the ordered keys of the fragments kept when it was built, so they record its clip, and `ranges` the ordered range ids of its columns (author, I2)) | rw | `interaction-matrix.json`, file, loader → null | none | T15; export | stage 5 editor; written by Continue 4→5, kept and adapted or by Recompute | planned |
| S15 | `energy_thresholds` (plain; the computed values) | ro | `computed-energy-thresholds.json` (name proposed), cell→file | T16 | none | stages 5–6: the computed values, clearly visible beside the editable table, the Copy source and the "differs from computed" marks (D13) | planned, reshaped |
| S16 | `gaussian_model` | | | | | | removed with T17 (author, O5) |
| S17 | `bestfit_stackings` (plain; name proposed) | ro | none | T18 | T28 | stage 6 computed table: descriptors | planned |
| S18 | `gaussian_classes` (plain; marker `{basis: selected keys, active, rows}`) | rw | `gaussian-classes.json`, file, loader → null | none | T19, T22 (not T20, so class edits never re-run T21) | stage 6 editor, with writes debounced by 500 ms | planned |
| S19 | `gaussian_validation` (plain) | ro | `gaussian-validation.json`, cell→file | T22 | none | stage 6 validation column; export gate | planned |
| S20 | `refe_validation` (plain) | ro | `refe-validation.json` (name proposed), cell→file | T23 | none | stage 6 refe panel; export gate | planned |
| S21 | `structure_id` (plain) | ro | none | T26 | none | gate 1→2: must equal S1 (§8.6.1) | new (state-transitions) |
| S22 | `fragments` (plain) | ro | none | T24 | T10 | stage 4: the skeleton's rows and the clip's range; gate 3→4 | new (state-transitions) |
| S23 | `range_pdbs_aa` (plain; name proposed) | ro | none | T25 | none | stages 4–6: the only structure the viewer shows (§8.6.5); gate 3→4 | new (author decision) |
| S25 | `gaussian_scores` (plain; name proposed) | ro | none | T28 | none | stage 6: candidate and reference scores, the computed classes (the class table's default and its "differs from computed" marks), `filter_set_version` and the filter names; the browser computes no Gaussian (author, O5) | new (author, O5) |
| S24 | `energy_threshold_edits` (plain `{schema:1, basis:{pdbcode, range-1 row}, ranges:[{row, values:{"frag-N-SEQ": value or null}}]}`; name proposed) | rw | `energy-threshold-edits.json`, file, loader → null | none | T27; export | Continue 4→5 creates it as an empty skeleton or keeps the entries of unchanged ranges; filled by Copy and/or typing; every true matrix cell must have a value before stage 6; editable in stages 5–6 (author, D13) | new (author, D13) |

¹ `interactions.json` lists `runs[].anchors`, which T15 can only derive from S7 and S8. The energetics mini-plan's signature for T15 omits them.

`anchor_modes` (test-anchor mini-plan) is dropped: by author decision, both anchor modes are always exported.

### 8.3 Backend-only cells
These cells are not shared. They are listed so that every pin of every §2 transformer is accounted for.

| # | Cell (celltype) | From | To | Status |
|---|---|---|---|---|
| B1 | `range_pdb_files` (folder; mount `pdbs/`) | T2 | T3 | exists |
| B2 | `aareduce_files` (folder; read-only mount) | vendored `aareduce/` | T3 | exists |
| B3 | `range_pdb_files_aa` (folder; mount `pdbs-aa/`) | T3 | B4, B6, T25 | exists |
| B4 | `rna_pdb` (bytes) | `range-1-aa.pdb` of B3 | T5, T6, T7 | planned |
| B5 | `exclude_pdb_code` (str) | S1, lower-cased | T5, T7, T9, T21, T24, T25 | planned |
| B6 | `protein_pdbs` (folder) | B3 without range 1 | T13 (one file per instance), T21 | planned |
| B7 | `fraglib_identities` (str) | `identities()`, at backend start | T5, T7, T21 | planned |
| B8 | `best_fit` (folder) | T5 | T7, T9, T12, T21 | planned |
| B9 | `refe_fragments` (folder) | T6 | T9, T12, T24 | planned |
| B10 | `best_fit_tables` (folder) | T7 | T8 | planned |
| B11 | `clipped_fragments` (folder; mount `clipped-fragments/`) | T11 | export | planned |
| B12 | `ligands` (folder) | T12 | T13 | planned |
| B13 | `energies_range_<k>` | T13 instance k | T14 | planned |
| B14 | `interactions` (plain; mount `interactions.json`) | T15 | export | planned |
| B15 | `gaussian_model_json` (plain; read-only mount from `$ALARIC_DIR`) | Alaric | T19, T21, T28 | planned |
| B16 | `ring_refe` (folder; read-only mount) | Alaric | T4, T18 (D8), T21 | planned |
| B17 | `gaussian_check_jobs` (plain) | T20 | T21 | planned |
| B18 | `gaussian_check` (plain) | T21 | T22 | planned |

The results of the validators T10, T19 and T27 are not shared. What matters to the frontend is only their node state in the state graph (§8.4); a failed T10 or T27 closes gate 4→5 or 5→6.

### 8.4 Errors, status and staleness
- **The state graph is the only error channel.** It is served at `/preparic/state-graph` and reaches the frontend through `onstategraph`.
  - The frontend finds a transformer's node by its path. The transformer names in §2 are therefore contract on this boundary too, and renaming one is a breaking change.
  - Today only T1 (stage 2 banner) and T4 (stage 3 banner) are surfaced. Planned:
    - T0 in stage 2, beside T1;
    - T26 in stage 1;
    - T3, T6 and T24 in stage 3, because an aareduce failure (for example, a modified nucleotide) must be shown before stage 4 can start;
    - T5–T11 in stage 4, T12–T16 and T27 in stage 5, and T18–T23 and T28 in stage 6;
    - T21's installation error (missing or mismatching caches) in stage 6.
  - When a pin's conversion fails, the transformer is `blocked` and its `exception` is empty. The frontend should show "blocked" for a blocked node whose upstream nodes have not failed.
- **Shares do not show staleness.**
  - While a transformer fails or recomputes, its share keeps serving the last value under the same marker.
  - The state graph is rebuilt at most every 0.5 s, and it can miss a short recompute or still read `complete` just after a PUT.
  - The reliable test is therefore the content. Each computed cell the frontend gates on echoes **every edit-cell value on its path**, restricted to the parts that matter. The echo carries values rather than checksums, so the frontend can compare them without hashing. A key set derived from one upstream is not enough (`final/state-transitions.md` §6).
    - S7 `stackings`: the structure id and the T0-normalised range rows.
    - S22 `fragments`: B5 and the range-1 row.
    - S9 `best_fit_thresholds`, S10 `best_fits`: B5, the range-1 row and its resids. Echoing B5 catches a change of PDB code with the same ranges, including the short window between the `cif` and `pdbcode` writes (§8.5).
    - S13 `energetics`: S10's echo plus every protein range row. Its fragment keys alone are not enough, because a protein-range edit leaves them unchanged.
    - S15 `energy_thresholds`: S13's echo.
    - S25 `gaussian_scores`: S17's echo, the selected keys and S12. It also carries `filter_set_version`: B15 and B16 are polled while the backend runs (author), so the model can change without a restart.
    - S17 `bestfit_stackings`: S10's echo plus the protein range rows.
    - S19 `gaussian_validation`: the S18 rows it judged, S12, the selected keys and S10's echo. T20's jobs echo the `best_fits` keys. A mismatch shows "validating…".
    - S20 `refe_validation`: the S11 `bases` values, S12, the selected keys and S10's echo.
    - S23 `range_pdbs_aa`: S5 `range_specs` and B5. If either differs from the tab's current values, the viewer shows "preparing all-atom structure…" instead of the old PDBs.
  - A cell is fresh when its echo equals this tab's current values, pending writes included. A tab that has just written sees its dependents go stale at once.
  - The export is blocked while any gate is stale, running or failed (§3, stage 6).

### 8.5 Writes that span several cells
Each key is written separately, and the writes are not atomic. With markers and bases (§8.1), only one user action writes two edit cells: a new PDB code writes S3 and S1, only once the CIF download has succeeded, under the arbiter rule (§8.6.1). Every other action writes one cell:
- a stage 3 change writes S8 only. S14 and S18 are inactive at stage 3, so they need no reset;
- a change of PDB code or range 1 makes S11 and S12 stale through their bases instead of resetting them. Continue 3→4 resets a stale S12 to `{0,0}` and starts a stale S11 as a skeleton. It clamps an S12 that cuts a selected anchor (D2b);
- Back writes one marker, with `active: false`;
- a clip edit writes S12 only. It happens in stage 4, while S14 is inactive. The kept matrix is adapted once, on Continue 4→5 (author decision);
- an energy-threshold edit, or Copy, writes S24 only. A change of PDB code or range 1 makes all of S24 stale through its basis, and a change of a protein range makes only that range's entry stale through its `row`.

One exception: Continue into stage k first deactivates any active marker beyond k, and, when entering stage 4, resets a stale S12 or clamps one that cuts a selected anchor. When entering stage 5, it resets a stale S24 or reconciles its entries with the current protein ranges. It waits for those writes to be accepted before writing M_k (`final/state-transitions.md` §3.5). In normal flow there is nothing to deactivate.

Between PUTs, transformers may run on a mix of old and new values. The rules:
1. **Bases, not write order.** A decision whose basis no longer matches is invalid, so the stage falls back instead of contradicting its data. Transformers that run on the mix are out of the stage's scope, and their failures are not shown.
2. **Suppress banners while writes are pending.** The frontend hides banners from validators while it has pending writes on their inputs.

Conflicting edits of one cell resolve through the conditional PUT: the last accepted write wins, and the losing tab adopts the server value. There are no sessions; every visitor sees and edits the same project.

### 8.6 Decisions on the shared cells
1. **`pdbcode` is written only after its CIF has been downloaded (author decision).**
   - The problem: B5 comes from S1, but the structure comes from S3. The browser writes them in separate PUTs. Today S1 is written as soon as the code is entered, so T5 could fit the old structure while excluding the new code.
   - The frontend now holds back its write to S1 until Mol* has downloaded the CIF for that code. It then writes S3 and S1 simultaneously:
     - both PUTs are issued in the same tick, and neither waits for the other's response;
     - the S1 write joins the deferred CIF flush (`flushCifWrite`) instead of going out on its own.
   - A failed download, such as for a PDB code that does not exist, writes neither cell. Stage 1 shows the error, and the backend keeps the previous project.
   - The share protocol has no multi-key PUT, so the server still applies the two writes one after the other (§8.5). The B5 echo in `best_fits` (§8.4) marks as stale any result computed in between.
   - B5 stays derived from S1.
   - **Arbiter rule.** If two tabs commit different codes at once, S1 decides.
     - The tab whose S1 write was refused gives up and loads the winner's structure.
     - The tab whose S1 was accepted but whose S3 got a 409 resends S3.
     - Until S21 (the entry id read from the stored CIF by T26) equals S1, Continue 1→2 stays closed.
   - **Dropped when the stage changed.** The S3 + S1 write is deferred until the download completes. It is dropped, and the user told, if meanwhile the derived stage has changed, the tab is no longer live, or either cell has received a remote update (`final/state-transitions.md` §4.1, §7.2c).
2. **`range_specs` is derived from `ranges` (author decision).**
   - The problem today: the frontend writes both cells, and the backend links neither to the other. An out-of-band edit of `ranges.yaml` updates S4, but S5 only changes when stage 2 next writes it. With five planned consumers of S5, that drift would matter more.
   - A new preparic transformer, T0 `parse_ranges`, parses S4 into S5. S5 becomes a read-only computed cell.
     - T0 reproduces the frontend's `parseRangeDefinition` and gives the shape `{ranges:[{chain,start,end,always_visible}]}`.
     - S4's scalars are strings, so integers are converted here. A parse error shows as a stage 2 banner.
   - The frontend stops writing S5: `setSharedRangeSpecs` and its calls in `App.svelte` go. The backend no longer initialises S5 from the file.
3. **`range_pdbs` (S6) is unshared (author decision).** No frontend code reads it. The stage 6 mini-plan planned a reader for `ring-geometry.ts`, but T18 replaced that module. The `share()` call at `backend.py:182` goes; the cell itself stays, because T2, T4 and T18 read it. The frontend displays the aareduce output (S23), never these raw extractions (§8.6.5).
4. **Export (deferred).**
   - It reads B11, B14, S24 and the mounted files.
   - Its "proceed anyway" checkbox is computed in the frontend, from the sources below. Amber states trigger it like red ones (author, D13).
     - S9/S11 against the raw values;
     - S13/S24: a true kept cell whose best-fit energy is not below its edited threshold (author, D13);
     - S19 and S20.
   - It needs no new shared cell, apart from whatever triggers the export.
5. **From stage 4 on, the viewer shows the backend's aareduce PDBs (author decision, 2026-10-29).**
   - In stages 1–3 the viewer shows the structure downloaded from RCSB, as today, because ranges are picked on it.
   - In stages 4, 5 and 6 it shows the files of S23, one per range, exactly as T3 wrote them, and nothing from the RCSB structure. Fragment overlays (reference and best-fit PDBs from S10) are drawn on top. All of them share the CIF frame.
   - The user therefore sees exactly the atoms every computation from stage 4 on uses: heavy atoms only, aareduce's atom names and order, the RNA unrenumbered and the proteins in original numbering.
   - The viewer maps a file to its range by S23's `range` field, never by the chain ids inside the file.
   - S23 is read like any computed cell: it is pending until fresh (§8.4), and it is read again after a reconnect, or by the reloaded browser after a backend restart (O6).
   - Gate 3→4 additionally requires that this tab holds a fresh S23, so stage 4 never opens on an empty or stale viewer.
   - Assembly copies (author, D7): the backend applies the assembly operators, so a range picked on an assembly copy is shown from stage 4 on where it was picked.
