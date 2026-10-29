# Preparic ↔ Alaric: oracle contract for project preparation

This is a contract, not a design. It says what preparic must do and what Alaric must provide during project preparation (stages 1–6), at the level a test oracle needs. Each clause states an observable property, and its **Oracle** line says how to decide whether an implementation meets it. How a clause is met is left open. Where a plan chose a mechanism (a factory, a shared cell, an echo, a marker), this document keeps only the property the mechanism was meant to secure. Appendix A points back to the plans that describe one way of meeting each clause.

**Sources and precedence.** The author's decisions win. They are recorded in these places:
- `open-questions.md`;
- the "author decisions" list of `final/state-transitions.md` and the answers in its §11;
- the author's clarification of 2026-10-29 on clip and interaction matrix (P4.4, P5.6);
- the author's decision of 2026-10-29 on the displayed structure (P11);
- the author's decision of 2026-10-29 to reduce the energy-thresholds file to its threshold mapping (P5.7, A12);
- the author's decision of 2026-10-29 on the energy-filter `cascade` and `min_cascade` parameters and `DATA/score-filter.yaml` (A12, X4);
- the author's decision of 2026-10-29 that stage 3's default selection uses displacement < 60° instead of tilt < 30° (P3.1);
- the author's decision of 2026-10-29 that slip angle and displacement angle are one and the same, φ, measured from the protein ring normal (A7, P3.1, P3.2);
- the author's answers of 2026-10-29 in `contract-open-questions.md` on the clip, the threshold table and the interaction matrix: D2, D3, D4, D12 and I2 (§1 clip, G9, P4.3, P4.4, P5.3–P5.7); on the scored pose, the go-ahead items and ovRMSD above L_ov: D10, D16 and D11 (P5.1, A11, A12, P6.6, X6); and on "Proceed anyway" and editable energy thresholds: D13 (P5.7, X5); on assembly copies, ring normals, class granularity, gaps and fragment numbering: D7, D8, D9, D15, O2, O3 (§1, P2.1, P3.2, A2, A12, P6.3); and on where Gaussians are computed: O5 (P6.1–P6.3); on this document's status, restarts, racing tabs and stored decisions: O1, O6, O7, D17 and I7 (this paragraph, G9, P2, P3, P5, P11); and on the fit gate: I6 (P10).

After the author's decisions come `final/contract-boundaries.md` and `final/state-transitions.md`, both finalized on 2026-10-29, and then the mini-plans listed in `INDEX.txt`. The author has adopted this document as the acceptance contract: its clauses are the requirements, and their oracles define the tests, at the precedence above (O1). Its own additions that the author has not yet ruled on are gathered in `contract-open-questions.md`.

**Scope.**
- **In:** stages 1–6, every crossing between preparic and Alaric during them, and the obligations of the export stage that are already fixed (§6).
- **Out** (deferred or excluded by the author): the design of the export stage, unbound protein structures, modified nucleotides, remote workers, deployment, authentication, and more than one project per backend.

**Conventions.**
- MUST, MUST NOT and SHOULD are normative.
- "1b7f" means the fixtures under `alaric/tests/1b7f*`. They are the acceptance fixtures for every Alaric step and for every preparic stage that runs one.

## 1. Terms

- **Preparic** is the browser app plus its Seamless backend, treated as one party. How work is split between browser and backend is preparic's own concern, except where a clause says otherwise.
- **Alaric** is the repository whose `alaric/` package folder is `ALARIC_DIR` (default `~/alaric/alaric`), so the repo root is its parent. It includes the `attract-jax` submodule, whose pointer Alaric owns.
- A **step** is a deterministic computation that Alaric provides and preparic runs (§3).
- A **decision** is a value the user sets: PDB code, biounit, structure, ranges, stacking selection, the threshold table, the clip, the interaction matrix, the energy-threshold table, and the Gaussian class table.
- **Stage decisions.** Five decisions are each the decision of one stage: ranges (2), stacking selection (3), threshold table (4), interaction matrix (5) and class table (6). The clip is a parameter of stage 4, not its stage decision; likewise the energy-threshold table is a parameter of stage 5 (author, D13).
- A **result** is anything computed from decisions, code and data.
- **Range 1** is the RNA. Ranges 2…K are protein ranges.
- **Residue numbers.** Range 1 is extracted with the structure's label numbering and never renumbered (G6). Fragment numbers do not depend on residue numbers.
- **Fragment N** (1-based) is the dinucleotide of the Nth and (N+1)th residues present in the prepared RNA file, whatever their residue numbers (author, O3). Every Alaric component uses this numbering (A12).
  - A fragment exists only if its two residues are consecutive in the chain. A window that spans an unobserved residue is not a fragment; its number is a hole, and the later numbers stay ordinal (author, O3). For residues 1, 2, 4, 5: fragment 1 is (1, 2), there is no fragment 2, and fragment 3 is (4, 5).
  - Preparic refuses an RNA range with any unobserved residue (author, D15; P2.1), so preparic's fragments are 1 … n with no hole.
  - n is the highest fragment number.
  - The key of fragment N is `frag-N-SEQ`, where SEQ is its two-letter sequence.
  - The pair (N, N+1) exists iff both fragments exist, and it shares the (N+1)th residue.
- The **reference fragment** is fragment N cut from the RNA structure.
- The **best-fit pose** is the library pose (conformer, rotamer, grid cell) closest to the reference fragment, fitted with the project's PDB code excluded from the library. The **best-fit PDB** is that pose written out.
- A **stacking anchor** is a selected stacking: (RNA residue r, protein range, protein residue). Its **candidates** are the fragments that contain r, where they exist: with r the kth residue present, fragment k−1 with r as `second`, and fragment k with r as `first`.
- The **clip** is (clip1, clip2). It is valid iff all three hold:
  - both are non-negative integers;
  - clip1 + clip2 ≤ n − 1;
  - it does not reach beyond any selected stacking anchor: no candidate fragment of a selected anchor is clipped (P4.4; the strict reading, author, D2).

  The kept fragments are clip1+1 … n−clip2. A pair is kept iff both of its fragments are kept. A candidate is unclipped iff its fragment is kept.
- **R(x)** is the threshold rounding: the smallest multiple of 0.01 that is strictly greater than x. Here x is the value as delivered: the table's decimal text for RMSDs, and the shortest round-trip decimal of the float64 for energies.
- The twelve **filter names** are `rrm50`, `rrm60`, `rrm70` and `rrm80`, plus `rrmN`, `rrmN-tilt30`, `rrmN-disp45` and `rrmN-tilt30-disp45` for N ∈ {90, 95}.
- **L_ov** is the largest ovRMSD threshold for which Alaric guarantees that it misses no pose. Today it is √3 Å ≈ 1.7320508 (§3.11).

## 2. Global clauses

**G1. One machine.** Preparic and Alaric run on the same machine, with no remote workers. Preparic finds every Alaric file it uses through one setting (`ALARIC_DIR`, settable at start).
*Oracle:* pointing that setting at another checkout is the only change needed to use that checkout.

**G2. Alaric code runs only inside Seamless transformations.** Preparic never imports Alaric from the installation into its backend process. Alaric code, rounding included, enters each transformation as part of its identity.
*Oracle:* a byte change in any file a step declares changes that step's result identity. Preparic's backend process holds no `alaric` import.

**G3. Determinism.** Every result is a function of its declared inputs only: decisions, declared code and parameter files, and data identities.
- It does not depend on the worker environment (`alaric`: Py3.12/numpy 1.26; `preparic`: Py3.15/numpy 2.5), on chunking, on process count, or on cache state.
- If a result would ever depend on the environment, that dependency becomes a declared input instead.

*Oracle:* each step's sandbox test (§3.0) gives the golden output, to the precision of its output format, in every environment the step is declared to run in.

**G4. Large data is identified, not staged.** The fragment library and the Gaussian caches stay where they are installed. Their identity enters results through committed checksums. Their location never does.
*Oracle:* moving the library and updating its locator gives cache hits. Changing a committed library checksum gives cache misses.

**G5. One excluded PDB code.** Every step that consults the library receives the same exclude code: the stage-1 PDB code, lower-cased. It is always set, and there is no way to turn it off. The export uses the same code.
*Oracle:* every library-dependent result of a project is computed with the same code. Exclusion cannot be disabled from the UI.

**G6. Input fidelity.**
- **No lenient flags.** `--ignore-unknown`, `--ignore-missing` and `--ignore-reordered` are never passed.
- **No renumbering.** Best-fit PDBs keep template numbering (chain B, residues 1–2). Everything downstream keys on `frag-N-SEQ` and fragment numbers, never on residue numbers or row positions.
- **RNA preparation.** Range 1 is prepared as all-atom RNA (`--rna --heavy --nalib --no-renumbering`, never `--manual`). The resulting file has no unknown residues, no missing heavy atoms, and topology atom order, which is the order of Alaric's templates. It is the input of every library step, and it becomes `DATA/reference.pdb` at export (X2).
- **Protein preparation.** Protein ranges are prepared as heavy-atom PDB2PQR output with their original numbering.
- **Known limitation (deferred).** An RNA range that contains modified nucleotides fails at preparation.

*Oracle:* on 1b7f, the prepared RNA file is accepted by the fitter with no lenient flag. No invocation carries one.

**G7. One rounding rule, applied once, in Alaric.**
- Every threshold derived from a raw value (ovRMSD, cRMSD, energy) is R(raw). No margin is added.
- Alaric accepts a pose iff its value is strictly less than the threshold.
- Preparic never re-implements R. It either receives rounded values from an Alaric step or runs Alaric's rounding code.

*Oracle:* for every derived threshold T and its raw value x, the IEEE-double comparison x < T holds, so the best-fit pose always passes. Required cases:

| raw | threshold |
|---|---|
| 0.123 | 0.13 |
| 0.12 | 0.13 |
| −16.5913 | −16.59 |
| −16.59 | −16.58 |
| −3.141 | −3.14 |
| 0 | 0.01 |
| −0.001 | 0.00 |

The 18 hand-copied 1b7f YAML thresholds are reproduced unchanged.

**G8. Versioned formats.** Every JSON that crosses the boundary, in either direction, carries `schema: 1`. Every consumer raises on any other schema. Alaric bumps the schema on any breaking change. There are no exceptions (author).

*Oracle:* feeding `schema: 2`, or a file without `schema`, to each consumer gives a visible error and no partial use.

**G9. Nothing is repaired or regenerated silently.**
- An invalid decision is refused or shown as invalid. It is never corrected.
- A stored decision that cannot be parsed counts as undecided.
- A stored decision without a basis, such as one written before bases existed, also counts as undecided, and the user is told why. It is never migrated (author, I7).
- When the startup loader resets a malformed stored file, the file's diagnostics are kept first (author, I7).
- Missing or mismatching installed data is an installation error, and the message names the command that produces the data.
- Every failure reaches the user (P6).
- **The only exception (author, D2):** Continue 3→4 clamps a clip that cuts a newly selected anchor (P4.4). This correction is automatic and may be silent.

## 3. What Alaric must provide

### 3.0 The form of a step (applies to A1–A9)
A step is deterministic code with declared inputs and one output. The implementer chooses the form for each step:
- a module with a Python function (preferred); or
- a CLI tool with its file-dependency list.

A factory that returns a ready transformer is optional. One exists for A1, and one is suggested for A5. Whatever the form, the step MUST meet the following.
- **Sandbox-complete.** Given only its declared inputs and dependencies, plus identified-not-staged data (G4), it runs and gives the same result. The sandbox test is part of the deliverable: the step runs as a Seamless transformation on 1b7f and matches its golden output.
  - Module form: a Python transformer with its sibling modules attached.
  - CLI form: `seamless-run -I <dependency list>`.
- **Project values are inputs.** Every project-dependent value is a named input with a documented name and type: RNA PDB, protein PDB, exclude code, library identity, job list, and so on. Nothing project-dependent is baked in at construction. Preparic connects its values to the inputs and never rebuilds a step because a decision changed.
- **Environment limits.**
  - Module-form code runs in preparic's environment (Py3.15, numpy 2). Its dependencies must be modest and compatible with that: no JAX, and nothing that pins numpy < 2.
  - Code that cannot meet those limits uses the CLI form and may name a worker environment, usually `alaric`. Its result must not depend on that choice (G3).
- **Exact, minimal dependency lists.**
  - The list names every file the step may load, and nothing else.
  - It is readable by `seamless-run -I`: one path per line, relative to the list's own directory, with no comments and no blank lines.
  - No listed file has a `.CHECKSUM` sidecar, because a stale sidecar would hide an edit.
  - The step reads its list from that one file, so the sandbox test and any factory pin the same files.
- **Factories, if any:**
  - import only the standard library and Seamless;
  - pin code and fixed data at construction;
  - leave every project input unbound;
  - never configure Seamless, run the transformation or materialise its result.

*Oracle:* the sandbox test passes. Every project input can be connected after construction. With imports outside the declared set blocked, the step still passes.

### 3.1 A1. Best fit (exists; one change required)
- **In:** the RNA PDB (bytes), the exclude code (str) and the library identity (str).
- **Fixed:** code, the 20 templates, and the library locator.
- **Out:** a folder holding:
  - `best-fits.tsv` with exactly the columns `fragment sequence conformer rotamer grid_x grid_y grid_z rmsd`. `fragment`, `conformer` and `rotamer` are 1-based; the grid is in units of √3/3 Å.
  - `pdbs/frag-N-SEQ-bestfit.pdb` for every fragment, in the RNA's frame, chain B, residues 1–2, template atom order.
- **The pose is one that Alaric's pose pools can contain:** coordinates = conformer · R(rotamer) + grid · √3/3, with the same convention and spacing as the anchor action.
- **How the pose is chosen.** The conformer is the one with the lowest superposed RMSD to the reference fragment. The rotamer and grid cell then minimise the in-place RMSD, with no superposition, over all heavy atoms of both nucleotides in template order. `rmsd` is that minimum, printed with 6 decimals.
- **Change required:** the RNA PDB and the exclude code are fixed at construction today. They must become inputs (§3.0).

*Oracle:* the 1b7f golden TSV is reproduced. Both inputs can be connected after construction. The column set and the PDB names are exactly as above, because preparic, A5 and A9 read them.

### 3.2 A2. Reference fragments (new)
- **In:** the RNA PDB.
- **Fixed:** code and templates. No library is needed.
- **Out:** a folder with `frag-N-SEQ-refe.pdb` for every fragment. Each has the same atoms, in the same order, as A1's PDB for that fragment. The file names are distinct from A1's.
- **Same fragments as the fit** (agreed by the author, O2). For a given RNA PDB, A2's set of fragment keys equals A1's. Preparic may therefore list the fragments, and open the threshold table in stage 4, before the fit has finished (P4.3; author, D6).

*Oracle:* the fragment key sets are equal, including on a copy of the RNA with a numbering gap. The atom lists match A1's PDBs atom for atom. On 1b7f, for every fragment, the in-place RMSD between the reference PDB and the best-fit PDB, compared atom by atom in template order with no superposition, equals the TSV `rmsd` to within 1e-4 Å. The tolerance covers the 3-decimal PDB coordinates.

### 3.3 A3. Best-fit tables (new)
- **In:** A1's output (or just its TSV), the RNA PDB, the exclude code and the library identity.
- **Fixed:** code, templates and the library locator. The locator's configuration must contain the cRMSD key, even though no cRMSD data is read.
- **Out:** two tables.
  - **Pairs:** one row per adjacent fragment pair (down N, up N+1, trinucleotide).
    - ovRMSD: the unaligned RMSD of the shared nucleotide as placed by the two best-fit poses.
    - cRMSD: the superposed (Kabsch) RMSD between the two conformers' versions of that nucleotide.
  - **Bases:** one row per fragment. For nucleotide 1 and for nucleotide 2, the unaligned RMSD of its base heavy atoms (two-character atom names) in the best-fit pose against the reference.
- Values come from library coordinates, never from written PDBs. PDB precision flips one 1b7f threshold, so the library-free shortcut is not acceptable.

*Oracle:*
- The output is byte-identical to `tests/1b7f/data/refe-best-fit-{pairs,bases}.tsv`.
- **Subset property:** for any clip, a run on the kept sub-table gives exactly the kept rows of the full run.

### 3.4 A4. RMSD thresholds (new; standard library only)
- **In:** A3's output.
- **Out:** `{schema:1, pairs:[{down, up, sequence, ovrmsd, crmsd, ovrmsd_raw, crmsd_raw}], bases:[{fragment, sequence, nuc1_ovrmsd, nuc2_ovrmsd, nuc1_ovrmsd_raw, nuc2_ovrmsd_raw}]}`.
  - The rounded fields are R(raw) (G7).
  - The `*_raw` fields carry the table values unchanged. They are contract: preparic validates edits against them (P6.6, X5).

*Oracle:* G7's cases pass, together with an exact multiple, a negative value and a value just below a multiple. The 18 1b7f YAML values are reproduced.

### 3.5 A5. Gridless energetics (new)
- **Granularity:** one run per protein range, scoring a whole folder of ligands against that range.
- **In:** a ligand folder (every reference and every best-fit fragment PDB) and one receptor PDB (one protein range).
- **Fixed:** code, the bead-reduction rules, the force-field parameters, and ε = 15.
- **Semantics:**
  - The receptor and each ligand are reduced to ATTRACT beads.
  - Each ligand is scored where it sits: identity pose, no minimisation.
  - The kernel is gridless: full 8-6 LJ with no cutoff, plus distance-dependent-dielectric electrostatics, in float64. Charges are dropped, so electrostatics are zero for RNA.
- **Out:** `{schema:1, epsilon, receptor:{beads}, ligands:{name:{beads}}, pairs:[{ligand, energy, error}]}`.
  - The range index is not part of the result.
  - A ligand that fails reduction gets an `error` and no energy.
  - A receptor that fails reduction fails the run.
- **Dependencies:** the reduced list of `final/contract-boundaries.md` §4 (8 files), which Alaric maintains.
  - Every attract-jax entry in it also appears in attract-jax's own full list.
  - The step needs no JAX, no scipy and no compiled kernel, and runs on numpy 1.x and 2.x.

*Oracle:*
- With JAX and scipy imports blocked, one 1b7f fragment scored against `1b7f_dom1-aa.pdb` reproduces `minfor.py --oracle numpy --no-grid --identity` to 1e-6.
- A ligand with an unknown residue gets a per-ligand error.
- Editing a grid or JAX file of attract-jax leaves the result identity unchanged. Editing a listed file changes it.
- Both environments give the same result to the printed precision.

### 3.6 A6. Energy-threshold rounding (new; standard library only)
A function, in a module that imports only the standard library, maps a raw energy to its threshold: R(raw), as a two-decimal string. Preparic's computation of the computed energy thresholds (P5.7) uses it for rounding and nothing else. P5.7's file is a two-way contract, because Alaric's energy filter reads it (A12).

*Oracle:* G7's cases pass.

### 3.7 A7. Gaussian filter specification (committed data)
One committed JSON at a stable path (`$ALARIC_DIR/gaussian_stacking_filter.json`), with `schema: 1` and a `filter_set_version`. It contains:
- **model:** the descriptor order (distance in Å, displacement in rad, tilt in rad) and their definitions; mean, covariance, precision and normalizer; and the formula score = exp(−½ ΔᵀPΔ) / normalizer.
  - v is the RNA ring centroid minus the protein ring centroid, in the protein ring's frame.
  - d = |v|.
  - φ is the angle of v from the protein ring normal.
  - θ is the angle between the two ring normals.
  - Normals come from a Kabsch fit of each ring's atoms onto the reference rings (G→A, U→C, TYR→PHE).
  - Stage 3's slip angle is φ: both are measured from the protein ring normal (author).
- **mask:** d < 5 Å and φ < 60°, both strict (the applied mask). The cutoff-derivation envelope (d < 5.7 Å) is provenance only and is not a top-level key.
- **filters:** `tilt30` means θ < 30° and `disp45` means φ < 45°, both strict.
- **thresholds:** exactly the twelve names. Each entry has its cutoff (the report value rounded down to five decimals) and `raw_cutoff` (the report's full-precision value), `accept: ">="`, its filters, cutoff rule, target, achieved recall and counts. For each cache file it also has `sha256` (compressed bytes) and `npy_sha256` (decompressed array).
  - The cutoff of `rrmN` is the Gaussian-only RRM-N cutoff (rounded down to 1e-5) from `envelope-analysis-5-report.md`.
  - Every variant `rrmN-…` has exactly its parent's cutoff, with its prefilters added on top. Its recall is therefore below N, by definition.
  - The re-derived per-variant cutoffs (0.229, 0.270, 0.164, 0.029) MUST NOT appear.
- **golden:** vectors mapping (d, φ°, θ°) to a score, plus accept/reject pairs on either side of every cutoff and at the 30° and 45° boundaries.
- **Legacy compatibility is a non-goal** (author, 2026-10-29, legacy r90/r95). No legacy model/metadata entry or support for existing legacy-filter projects/fixtures is required.

*Oracle:*
- The golden vectors reproduce at relative tolerance 1e-12, in Python. The browser computes no Gaussian (author, O5).
- Each variant's cutoff equals its parent's.
- The six Gaussian-only `raw_cutoff` values equal the report's values bit for bit, and the applied cutoffs are those values rounded down to five decimals:

| name | raw_cutoff | cutoff |
|---|---|---|
| rrm50 | 3.303844018095503 | 3.30384 |
| rrm60 | 2.904318239783102 | 2.90431 |
| rrm70 | 2.04198041493334 | 2.04198 |
| rrm80 | 1.1974880057674258 | 1.19748 |
| rrm90 | 0.27592087772968793 | 0.27592 |
| rrm95 | 0.06141634850710175 | 0.06141 |

**Containment.** Filter A ⊆ filter B iff A's masks include B's masks and cutoff_A ≥ cutoff_B. With the cutoffs above, this gives:
- `rrm50 ⊂ rrm60 ⊂ rrm70 ⊂ rrm80 ⊂ rrm90 ⊂ rrm95`;
- for N ∈ {90, 95}: `rrmN-tilt30-disp45 ⊂ rrmN-tilt30 ⊂ rrmN`, and `rrmN-tilt30-disp45 ⊂ rrmN-disp45 ⊂ rrmN`;
- each 90 filter lies inside its 95 counterpart.

### 3.8 A8. Gaussian caches and their generator
- **Generator:** an Alaric script, neither Seamless nor committed data, writes the four CSR arrays for each name to `<root>/precomputed-filters/gaussian/<name>/`.
  - That folder is gitignored.
  - The arrays may be installed compressed (`.npy.zst`) or decompressed (`.npy`).
- **Content:** the Hopf rotation grid (98,304 cells) crossed with the 0.25 Å translation lattice in ±5 Å. A (rotation, translation) pair is kept iff:
  - its score is ≥ the cutoff;
  - it lies inside the mask (d < 5, φ < 60°);
  - it passes the name's prefilters.

  All masks are strict and evaluated exactly on the lattice. Ties do occur there, so the boundary tests must not go through `arccos`.

*Oracle:*
- The generated files' checksums equal the JSON's.
- Random membership samples agree with the model.
- Legacy r90 reproduction is not an acceptance requirement (author, 2026-10-29).
- The CSR invariants hold.

### 3.9 A9. All-filter check (new)
The author's requirement: check a small list of anchor poses against all twelve filters within a few seconds, using the filter hierarchy to stop early.
- **In:**
  - a job list: one job per unclipped candidate of each selected anchor, carrying key, fragment, nucleotide, sequence, conformer, rotamer, grid cell, protein range and protein residue;
  - A1's output;
  - the protein range PDBs;
  - the exclude code and the library identity.
- **Fixed:** code, A7's JSON (which carries the cache identities), the reference rings, and the minimum cell fraction (default ⅓).
- **Per job and filter:**
  - `count`: the accepted translations of the rotamer's rotation cell that land on the best-fit grid cell, computed exactly as the anchor action computes them;
  - `capacity`: the cell capacity the anchor action uses;
  - `fill` = count / capacity;
  - `emitted` ⇔ fill ≥ the minimum cell fraction;
  - `pose_pass` ⇔ the actual, unsnapped pose scores ≥ the cutoff and the filter's prefilters hold;
  - `score`.
- **Per job:** an `error` for an incomplete ring, or when the pose rebuilt from (conformer, rotamer, grid) differs from the best-fit PDB by more than 0.002 Å RMSD.
- **Early cutoff:** when a containing filter does not emit a job's cell, the filters inside it are not evaluated. They are returned as `{emitted:false, implied_by:<that filter>}`. The hierarchy is derived from the JSON, never hard-coded.
- **Out:** `{schema:1, filter_set_version, min_cell_fraction, jobs:[{key, fragment, nucleotide, error?, filters:{<name>:{count, capacity, fill, emitted, pose_pass, score, implied_by?}}}]}`.
- **Installation check:** the step verifies the caches' checksums before use. A missing or mismatching cache is an installation error that names the generator command.

*Oracle:*
- **Parity with the anchor action:**
  - on 1b7f, a job's `emitted` equals whether an anchor run with that filter emits that cell;
  - every pose an anchor run emits has fill ≥ the minimum cell fraction;
  - the existing Gaussian anchor fixture output stays bit-identical.
- **Monotonicity:** `emitted` and `pose_pass` never go from false to true when moving from a filter to one it contains.
- **Speed:** with about 20 jobs, the check finishes within a few seconds, not counting decompression.

### 3.10 A10. Reference rings
`ring-refe/refe-{A,C,PHE,HIS,ARG,TRP}.pdb` stay at a stable path under the repo root. Preparic pins them by checksum for its stage 3 and stage 6 detection, which uses Kabsch normals fitted onto them (author, D8), and for the check.

### 3.11 A11. Constants preparic mirrors
Alaric defines each of these in one place. Preparic mirrors it, and a test checks that the two are equal.
- **L_ov**, the ovRMSD coverage limit: √3 Å ≈ 1.7320508, three grid steps.
  - `MAX_OVERLAP_RMSD = 1.3` (`grow.py:32`) is only the limit of the loop that builds the translation-offset tables. That loop adds one step past it, so the largest table covers √3 (`grow.py:268-310`).
  - Today, for a larger threshold, the table lookup clamps to the √3 table (`grow.py:318-320`). There is no error and no warning, so poses whose true ovRMSD is below the threshold can be missed.
  - This applies equally to `grow`'s `ovrmsd` and to `anchor-refe`, which reuses grow's kernel.
  - A simulation suggests misses begin at about 2.03 Å, but that is not a guarantee.
  - `crmsd` has no such limit.
  - Alaric MUST raise on an `ovrmsd` above L_ov, in `grow` and `anchor-refe` alike, instead of clamping. The tables are not extended, so L_ov stays √3 Å (author, D11).
- **The boundary-recompute tolerance:** 1e-5 Å. Values within it of a threshold are recomputed exactly.
- **The minimum cell fraction:** ⅓. Alaric lets `ALARIC_ANCHOR_MIN_CELL_FRACTION` override it, and an exported project MUST NOT set that variable.

### 3.12 A12. What an exported project will run
Export is deferred, but these obligations on Alaric's actions are already fixed.
- **`energy-filter`** is new. It replaces `score`, `filter` and `score_add`, which are removed. Alaric's score scripts either move to the gridless path, using A5's dependency list, or go.
  - One action handles one fragment against exactly one protein range.
  - The threshold is a number, or comes from the energy-thresholds file (P5.7). A missing cell is an error.
  - **Parameters the export must write.** The file carries thresholds only, so everything else is an action parameter:
    - `protein: range-K`: exactly one range per action; one action per true matrix cell.
    - `threshold: auto` (or a number).
    - `epsilon: 15`: the ε that A5 scored with. A threshold is valid only for the ε its energy was computed with.
    - `cascade: auto` and `min_cascade: auto`, which read `DATA/score-filter.yaml` (below).
    - `fragment`, `sequence` and `exclude`: `auto`, with `exclude` resolving to the project's exclude code (G5, X1).

    The provenance that the old file carried (`source: "bestfit"`, the rounding rule) is fixed by P5.7 and G7, and no action needs it.
  - A pose is kept iff its gridless energy E < T.
  - A fragment that interacts with several ranges is filtered against each in turn (AND). Energies are never summed.
  - **Cascade (author).** `cascade` is a list of grid steps `{plateau_distance, neighbour_distance, max_error}` (Å, Å, energy units), or `[]` for all-gridless. `min_cascade` is an integer. `auto` for either reads `DATA/score-filter.yaml`, which the export writes (X4) and which must be:
    ```yaml
    min_cascade: 1000
    cascade:
      - {plateau_distance: 5, neighbour_distance: 7, max_error: 1.0}
      - {plateau_distance: 10, neighbour_distance: 12, max_error: 0.05}
    ```
    Semantics, with P the undecided poses (initially the whole input):
    - before each step, if |P| < `min_cascade`, the cascade is exited;
    - otherwise the step scores P with its grid, giving g. A pose with g < T − `max_error` is accepted, one with g ≥ T + `max_error` is rejected, and both leave P. The rest stay undecided;
    - after the last step, or on exit, every pose in P is scored gridless and kept iff E < T.

    The output keeps the input order. |P| counts the whole action's undecided poses, never one chunk's. The output therefore equals all-gridless filtering whenever every grid error is ≤ its step's `max_error`, whatever the chunking or process count.
  - Prerequisite: the fix for the 10-12 bug, where a bead exactly on a grid node mis-scores the identity pose by 0.23. The author gave the go-ahead for it (D16).

  *Oracle:*
  - On 1b7f, output is byte-identical to the `cascade: []` run, chunked and unchunked.
  - Injected errors: a pose whose grid energy lies within `max_error` of T is decided by gridless scoring; a shift to g < T − `max_error` accepts it and a shift to g ≥ T + `max_error` rejects it, without gridless scoring.
  - A pool smaller than `min_cascade`, and a step whose undecided poses fall below it, both exit the cascade (visible in the action's statistics) and give the all-gridless output.
  - `auto` with a missing or malformed `DATA/score-filter.yaml` is an error.
- **`anchor`** with `precomputed_filter_name: gaussian-rrm` accepts exactly the twelve names, validated against A7. The cache identity enters the action's sigil.
- **`anchor-refe`** takes an explicit `ovrmsd` per action. It accepts a pose iff the ovRMSD of the selected nucleotide's base atoms against `DATA/reference.pdb` is < `ovrmsd`. It raises on an `ovrmsd` above L_ov (A11; author, D11).
- **`grow`** takes per-pair `crmsd` and `ovrmsd` and accepts with strict `<`. It raises on an `ovrmsd` above L_ov (A11; author, D11); `crmsd` has no such limit.

  *Oracle:* `grow` and `anchor-refe` raise with `ovrmsd: 1.74` and run with `ovrmsd: 1.73`.
- **One fragment numbering.** Every Alaric component that maps a fragment number to residues MUST use the ordinal numbering of §1: fragment N is the Nth and (N+1)th residues present, whatever their residue numbers (author, O3). Today:
  - `Reference` (`reference.py:101-144`), and with it the fitter, the tables and `anchor-refe`, numbers by residue offset from the first residue present. That agrees on gap-free input, but after a gap the numbers jump: residues 1, 2, 4, 5 give fragment 4 instead of 3. It MUST number ordinally (author, O3);
  - `rmsd.py --fragment N` uses absolute residue numbers N and N+1, which is wrong whenever the first residue is not 1. Tools that assume fragment N is residues N and N+1 MUST be fixed (author, D15);
  - `sequence: auto` (`middle/resolve.py:39-65`) and `sequence.txt` count positions among the residues present, so they already agree.

  *Oracle:* on copies of the 1b7f reference renumbered to start at a residue other than 1, and to open a gap, every component maps fragment N to the Nth and (N+1)th residues present, and none treats the window across the gap as a fragment.

## 4. What preparic must do: the shared project

**P1. One project.** Every visitor sees and edits the same project; there are no sessions. Up to propagation delay, every open tab shows the same stage and the same decisions.

**P2. Durable decisions.** Every decision, the downloaded structure included, survives a backend restart, and after a restart the project is at the same stage. Results are recomputed.
- Only mounted cells survive a restart, so every decision is mounted (author, O6).
- This clause, and every other, holds across a backend restart only if the browser is restarted right afterwards: every open tab reloads. A tab kept open across a backend restart has undefined behaviour (author, O6).

**P3. The stage.** The stages are linear: 1 structure, 2 ranges, 3 stackings, 4 best fit and thresholds, 5 energetics, 6 Gaussian filter, then the export.
- The stage is a function of the stored decisions alone. It does not depend on computation state, on results, or on which tab looks.
- Each stage decision is valid only against the upstream decisions it was made with. For the interaction matrix, that includes the clip it was built for (P5.6).
- The stage is the highest k such that the stage decisions of stages 2..k are all present, in force (not set aside by Back), well-formed and valid against the current upstream decisions.

*Oracle (scenario tests):*
1. The stage does not change while results compute, fail, lag, or are recomputed after a restart.
2. An upstream change from any source lowers the stage in every tab, without anyone writing. Sources include another tab, a stale tab's delayed write, and a hand-edited stored file. No stage ever shows decisions made for other upstream data.
3. Reverting that change restores the later stage and its cached results.
4. If tabs do not race, stage k is never reached unless its entry gate held when its decision was last written or reactivated by a Continue, and a stale later decision never comes back to life on its own (author, O7, D17). Interleavings of Continue, Back and edits across tabs are outside this guarantee.
5. With several tabs open, when tab A presses Continue while tab B is not editing, tab B proceeds to the next stage as if it had pressed Continue too (author, D17).

**P4. Transitions.**
- **Continue** from stage k is enabled only while its gate (§5) holds in this tab's view. It is never armed to fire later. It puts stage k+1's decision in force:
  - **restored:** a decision set aside by an earlier Back, if it is still valid against its upstream. For the matrix, "its upstream" means its selected anchors; it is then adapted to the current clip (P5.6);
  - **merged:** a stale stacking selection, merged by key into the default;
  - **default:** otherwise.
- **Back** from stage k takes every tab to stage k−1.
  - Back never destroys a decision. It sets aside stage k's decision so that a later Continue can restore it, and it therefore needs no confirmation.
  - This holds for every stage decision (author, D1). The matrix reuse on 5→4→5 requires it.
  - A decision made for other upstream data is never restored. A table made for another PDB code or range 1 is one example.
- **Deferred actions** (a debounced edit, a download completion, a delayed upload) belong to the stage they were issued in. One is dropped if, by the time it would be applied:
  - the stage has changed;
  - the tab is not connected; or
  - the same decision was changed remotely meanwhile.
- **Concurrent edits** of one decision: the last accepted write wins. The losing tab adopts it and tells its user.
- **Write granularity:** ranges are written per completed row, never per keystroke. The clip and the class table are debounced. Everything else is written per action.
- **One decision per user action.** No user action changes two decisions, except committing a new PDB code with its structure (P1.1). A decision stays editable in a later stage only if no later stage decision would have to change with it. That is why the clip is editable in stage 4 only (P4.4).

**P5. Freshness.** A result is shown or used as current only if it was computed from the decisions this tab now holds, its own unconfirmed edits included. Otherwise the tab shows the result's pending wording or its error. Gates use only fresh results.
- Freshness covers every decision on the result's path. For example, energetics depend on the protein ranges, not only on the fragment keys.
- A backend restart, which a code update implies, is covered only together with a browser restart (P2, O6), so no result read before it is ever reused.

*Oracle:* hold a result back while an upstream decision changes. The gate closes and the pending wording shows. Revert the change, and the result is fresh again.

**P6. Errors and status.**
- A computation's failure is shown in the stage it belongs to and in every later stage, never in an earlier one. Computations of later stages that run on lingering decisions may fail without being shown.
- A failure is hidden while this tab has an unconfirmed write on one of that computation's inputs.
- No failure is shown before the first status report after loading.
- A computation that cannot run because an input failed to convert is shown as "blocked".
- Installation errors (library, Gaussian caches) are shown in the stage that needs the data.

Pending wording:

| Stage | Pending wording |
|---|---|
| 1 | "uploading structure…" |
| 2 | none |
| 3 | "detecting stackings…", "preparing all-atom structure…" |
| 4 | "computing reference thresholds…" (Copy disabled), "fitting…", "preparing all-atom structure…" (viewer, P11) |
| 5 | "scoring…", "preparing all-atom structure…" (viewer, P11) |
| 6 | "computing best-fit stacking geometry…", "validating…", "preparing all-atom structure…" (viewer, P11) |

**P7. Validation.**
- Every editable decision is validated three times: in the browser inline, in the browser on every read, and in the backend, which raises (shown as an error).
- An invalid value is refused or shown as invalid, never silently corrected. The only exception is the clip clamp on Continue 3→4 (G9, P4.4).
- Every validator accepts "undecided".

**P8. Undecided is not empty.**
- "Undecided" is distinguishable from every legitimate decision value.
- A pending or absent result is never a reason to write a decision.

**P9. Load and connectivity.**
- Until the stored decisions have been read, the UI is read-only.
- While disconnected, the UI is read-only over the last state. On reconnect, the tab reads everything again.
- URL parameters act only at stage 1, and a PDB code from the URL only when no code is set. Opening a link therefore never resets a shared project.

**P10. Cost of edits.**
- A clip edit never re-runs a library step or scoring.
- A class edit never re-runs the all-filter check (A9).
- A matrix edit re-runs only the hand-off (P5.8). An energy-threshold edit re-runs only its validation (P5.7, P7).
- The fit and everything downstream of it start when stage 3 is entered, never earlier (author). A committed range edit in stage 2 starts no library work while the stacking selection is not in force. An upstream change that lowers the stage without a Back (a stale tab's delayed write, a hand-edited file) leaves the selection in force, and the fit then restarts at stage 2 (author, I6).

**P11. What the viewer shows (author).**
- In stages 1–3 the viewer shows the downloaded structure, on which the ranges are picked.
- In stages 4, 5 and 6 it shows the all-atom range PDBs exactly as the backend's preparation (G6) wrote them, one per range, and no atom of the downloaded structure. Fragment overlays (P4.2) are drawn on top, in the same frame.
  - The files reach the browser unchanged: not parsed and rewritten, renumbered, filtered or merged on the way.
  - A file is matched to its range by range index, never by the chain ids inside it.
- The displayed PDBs are a result like any other (P5): they are shown only while fresh for this tab's PDB code and ranges. Otherwise the viewer shows "preparing all-atom structure…", never the previous PDBs and never the downloaded structure instead.

*Oracle:*
- In stages 4–6 the set of displayed atoms (names, residues, coordinates) equals the union of the backend's all-atom range files, and the bytes the browser received equal those files.
- Held-back preparation after an upstream change shows the pending wording, not the old PDBs.
- After a backend restart in stage 4 or later, followed by a browser restart (P2, O6), the viewer shows the pending wording until the recomputed files have been read.

## 5. Preparic, stage by stage

### Stage 1: structure
- **P1.1** The user picks a PDB code and a biological assembly, and the browser downloads the structure.
  - The code and the structure are committed together, and only after the download succeeds.
  - A failed download changes nothing, and the error is shown.
- **P1.2** The project never proceeds with a code and a structure that disagree. The entry id read from the stored structure must equal the code.
  - If two tabs commit different codes at once, the code decides: the losing tab loads the winner's structure.
- **P1.3** The exclude code is derived from the code (G5).
- **Gate 1→2:**
  - a code is set;
  - this tab shows its structure;
  - the stored structure's entry id equals the code.

### Stage 2: ranges
- **P2.1** Ranges are rows (chain, start, end, always visible). Range 1 is RNA; every other range is protein.
  - Only valid, completed rows are committed.
  - The structured form that computations use is derived from the stored rows, by a parser that agrees with the browser's (integers converted, errors shown).
  - **No unobserved RNA residue (author, D15).** An RNA range in which any residue is unobserved is refused: the browser shows the row as invalid, naming the missing residues, and the backend refuses it too. Preparic's fragment numbers therefore have no holes.
  - **A row means the same atoms everywhere.** The atoms the backend extracts for a row are the atoms the user saw and picked in the viewer. The same holds for row validity: a row is valid in the browser iff the backend can extract it.
    - **Assembly copies (author, D7).** The backend applies the selected assembly's operators from the CIF. A range picked on a copy such as `A-2` is extracted with that copy's operators, so it is prepared, fitted and shown where it was picked. Row validity is checked against the selected assembly in the browser and in the backend alike. With D7 this property holds by design; the author considers the separate question O4 moot.
    - Today neither holds: the backend strips the suffix and extracts asymmetric-unit coordinates, which may sit elsewhere, and it checks validity against the asymmetric unit.
- **P2.2** From the ranges and the biounit (D7), preparic computes:
  - the per-range PDBs;
  - the all-atom preparation (G6);
  - stacking detection;
  - the reference fragments (A2).
- **Gate 2→3:**
  - the rows are valid and confirmed;
  - the stacking table is fresh and has at least one row (author).

### Stage 3: stackings
- **P3.1** Preparic shows the detected stackings.
  - The selection is keyed by stacking key, never by row.
  - The default selection is distance ≤ 5 Å and slip (= displacement φ, A7) < 60° (author). The distance test is inclusive and the angle test is strict. There is no tilt condition.
  - Continue into stage 3 merges the default with any earlier selection, by key.
- **P3.2** Stage 3's slip angle IS the displacement angle φ of A7: the angle of the centroid displacement from the protein ring normal, so 0° is the zenith and 90° the horizon. Stage 3 uses Kabsch ring normals fitted onto A10's reference rings, like stage 6 (author, D8), so their descriptors agree. Its numbers differ by up to 0.7° from those of the former SVD normals.
- **Gate 3→4:**
  - the selection's keys equal those of a fresh stacking table;
  - at least one stacking is selected (author);
  - the fragment list is fresh: the all-atom preparation and the reference fragments (A2) are complete. Their failures, for example a modified nucleotide, are shown in this stage;
  - this tab holds the fresh all-atom range PDBs, so stage 4 opens on the structure it must show (P11).

  The fit is not needed (author: the fit waits for stage 3, but stage 4 does not wait for the fit).

### Stage 4: best fit and RMSD thresholds
- **P4.1** Preparic runs A1–A4 on the full RNA range, never on a clipped one. All four use the same RNA PDB, exclude code and library identity. Stage 4 opens as soon as the fragments are known (A2), without waiting for these results.
- **P4.2** The stage's main content is the cRMSD/ovRMSD threshold table (author). Shown alongside it:
  - the computed reference values, rounded;
  - per fragment: residues, sequence, conformer and `rmsd`;
  - per pair: down→up and trinucleotide;
  - raw values are never displayed;
  - hovering a row overlays the reference and best-fit PDBs in the viewer, on top of the all-atom range PDBs (P11).
- **P4.3** The threshold table, which is the stage 4 decision:
  - **Entering stage 4.** Continue 3→4 puts the table in force:
    - a table set aside by an earlier Back is restored if it is still valid for the PDB code and range 1;
    - otherwise the table starts as a skeleton: one row per pair and one per base of the current fragments, with every value empty. It is never prefilled: Copy is an explicit action (author, D5). The computed values are clearly visible beside the table, so that it is clear what Copy does.
  - **Filling.** The user fills it by Copy, by typing, or both (author).
    - Copy is enabled only while the computed thresholds are fresh.
    - Copy asks before it overwrites typed values.
  - **Editability.** It stays visible and editable in stages 4–6 (author). Its bases part can also be edited from stage 6 (P6.6).
  - **Revert.** Each cell has its own revert to the computed value.
  - **Marks.** Cells that differ from the computed value are marked, with a count.
  - **Values.**
    - Each value is finite, > 0, with at most two decimals. More decimals are rejected, not rounded.
    - A value above 10 Å gives a warning.
  - **Keys** are exactly the pairs and bases of the fragments, by fragment number.
  - **Basis.**
    - The table is tied to the PDB code and the range-1 row it was made for. A table made for others is never restored: Continue into stage 4 replaces it with a skeleton. If the PDB code and range 1 revert before that, the table comes back with its values.
    - If the computed values change for the same rows (after a code or library update), the values are kept and the differing cells are marked.
  - **Clipped rows** are greyed but stay editable (author). Only kept rows go downstream, and only kept rows must be filled before stage 5 (author, D4).
- **P4.4** The clip (author):
  - **A parameter, always set.** The clip is a parameter of stage 4, never undecided; it starts at (0, 0). Like the table, it is tied to the PDB code and range 1. Continue into stage 4 resets a clip made for others to (0, 0).
  - **Where it can be edited.** It is editable in stage 4 only, and shown read-only in stages 5 and 6.
  - **Never beyond a selected stacking anchor.** No candidate fragment of a selected anchor is ever clipped (the strict reading, author, D2). The clip control stops at the outermost anchors' candidates. To clip further, the user first deselects that anchor in stage 3.
    - A stage 3 change can put a newly selected anchor inside the clipped region. Continue 3→4 then clamps the clip, before it puts the table in force: clip1 and clip2 are each reduced to the largest value that clips no candidate of a selected anchor (author, D2). The clamp is automatic and may be silent; it is the only exception to G9. Stage 4 therefore never opens with a clip that cuts an anchor.
    - A clamp can uncover empty threshold rows; the gate 4→5 then waits for them to be filled.
  - **Greying.** Clipped fragments, and any pair that touches one, are greyed in every table.
  - **Cost.** A clip edit re-runs no library step and no scoring. It never touches the interaction matrix, which is set aside whenever the clip can change and is adapted once, on Continue 4→5 (P5.6).
- **Gate 4→5:**
  - every value of the threshold table's kept rows is filled and valid, and its backend validator is not failing. Rows of clipped fragments, and pairs that touch one, need not be filled (author, D4); a clip reduction can therefore uncover empty rows, and the gate then stays closed until they are filled. The computed thresholds are not needed;
  - the clip is valid: it fits the fragment count and cuts no selected anchor;
  - the energetics are fresh. In practice this is the only place the user waits for the fit.

### Stage 5: energetics and interactions
- **P5.1 Scoring.** Preparic runs A5 once per protein range, over all reference and all best-fit fragments. The clip, the anchors and the matrix are not inputs.
  - The best-fit fragments are the discrete library poses that A1 writes, not exact superpositions onto the reference fragments (author, D10).
  - Editing one protein range rescores only that range.
  - A refit rescores every range.
  - Per-ligand errors do not block Continue.
- **P5.2 Display.** For each fragment and range, preparic shows the reference and best-fit energies, colour-capped at about ±20.
  - Clipped rows are greyed, but their energies stay visible.
  - Per-ligand errors are shown on their row.
  - The threshold table and the energy-threshold table (P5.7) are editable here; the clip is read-only.
- **P5.3 Anchors and columns.**
  - Each selected stacking gives an anchor (RNA residue r, protein range). Anchors with the same (r, range) collapse into one.
  - An anchor locks its candidates. Under the strict reading (author, D2), none of them is ever clipped, so locks never change with the clip.
  - There is one column per anchor, sorted along the RNA.
  - The **invariants:**
    1. Each column is one contiguous run that contains its lock.
    2. Columns on the same range are disjoint, unless joined by "until next anchor".
    3. No run contains a clipped row.
  - Columns on different ranges may overlap, so one fragment can interact with several ranges.
  - **"Until next anchor"** joins a column to the next anchor downstream.
    - It is allowed only when that anchor is on the same range.
    - It is forced on, and cannot be turned off, when the two locks overlap or touch.
  - A range with no selected anchor never gets interactions (author).
- **P5.4 Recompute.** The button is always enabled. It rebuilds every column from the energies, range by range, independently:
  1. Start from the lock and grow towards 5' and towards 3', one row at a time.
  2. A row is added while |E_refe| ≥ 1 or |E_bestfit| ≥ 1 for that range. Growth stops before the first row where both are < 1.
  3. Growth also stops before a clipped row, before a row with a scoring error, and at another same-range anchor's lock. If two same-range runs then touch or overlap, they are joined.

  Clashes (large positive energies) count as interacting (author).
- **P5.5 Editor.**
  - The user moves the caps of each column.
  - A cap cannot pass its lock, enter a clipped row, enter a row with a scoring error (author, D12), or overlap a neighbouring column on the same range.
- **P5.6 Stored matrix, its reuse, and read validation.**
  - **Stored form.** The matrix is a boolean grid with addressed rows and columns (author, I2). It stores the ordered keys of its rows, which are exactly the fragments that were kept when it was built, and the ordered ids of its columns, the protein ranges.
    - Its row keys record the clip it was built for, so that clip can be read from the matrix alone (author). A bare row count could not tell a 5′ clip from a 3′ clip.
    - A run that holds k anchors reads back as k joined columns. Two separate runs on one range are not joined.
  - **Tied to its clip.** A matrix built for another clip than the current one is not valid (P3), so the stage is at most 4. This happens, for example, when Continue 4→5 in one tab races a clip edit in another. The next Continue 4→5 adapts the matrix; nothing needs repairing.
  - **Reuse on 5→4→5 (author).** Continue 4→5 restores a set-aside matrix whose selected anchors are unchanged, adapted to the current clip:
    - the matrix's own clip is read from its row keys;
    - only the outer ends change. For each protein range, those are the 5' end of its 5'-most column and the 3' end of its 3'-most column. Every other cell, every lock and every "until next anchor" join stays as it was. This holds because the clip never reaches beyond an anchor;
    - rows that the new clip covers are dropped (the matrix is reduced);
    - rows that the new clip uncovers are filled in by Recompute's growth rule (P5.4), applied to the new rows only, and only at an outer end that touched the old clip boundary. From there the run grows outwards while |E_refe| ≥ 1 or |E_bestfit| ≥ 1 for that range, and stops before the first row where both are < 1 and before a row with a scoring error. An outer end that stopped short of the old boundary stays as the user left it (author, D3).

    Without such a matrix (none was set aside, or the anchors changed), Continue 4→5 runs Recompute.
  - **Read validation.** A read is refused unless:
    - the shape is kept fragments × ranges, with booleans only;
    - the matrix's extent equals the current clip;
    - every true run contains a lock of its range;
    - every anchor cell is true;
    - no true cell lies on a row with a scoring error (author, D12). If an anchor's own lock is such a row, no matrix can pass: the error names that anchor, and the remedy is to deselect it in stage 3. Nothing is deselected automatically (G9). This consequence is confirmed by the author.
  - **On a refused read:**
    - the editor is disabled;
    - an error names the failed check;
    - Recompute is offered, and nothing is written until the user presses it.
- **P5.7 Energy thresholds.** The user edits them in a table that works like the stage-4 threshold table (author, D13).
  - **Computed values.** For every (fragment, range) cell with a best-fit energy, the computed threshold is R(best-fit energy) via A6. Reference energies are information only (author). Neither the matrix nor the clip enters this computation.
  - **The energy-threshold table**, a parameter of stage 5:
    - **Entering stage 5.** Continue 4→5 puts the table in force, before the matrix:
      - values set aside earlier are restored for each protein range whose row is unchanged, as long as the PDB code and range 1 are unchanged too;
      - otherwise that range starts empty, and so does the whole table for another PDB code or range 1. It is never prefilled: Copy is an explicit action. The computed values are clearly visible beside the table, so that it is clear what Copy does.
    - **Filling.** By Copy, by typing, or both. Copy is enabled only while the computed values are fresh, fills every cell that has one, and asks before it overwrites typed values.
    - **Editability.** It stays visible and editable in stages 5–6.
    - **Revert.** Each cell has its own revert to the computed value.
    - **Marks.** Cells that differ from the computed value are marked, with a count.
    - **Values.** Each value is finite, of either sign, with at most two decimals. More decimals are rejected, not rounded.
    - **Keys** are every fragment × every protein range. Cells that are false or clipped in the matrix are greyed but stay editable. Only true cells go downstream, and only they must be filled before stage 6.
    - **Basis.** The table is tied to the PDB code and range 1. Each range's values are also tied to that range's row, not to its index. A protein-range edit therefore resets only that range's values, when stage 5 is next entered (author, D13). If the computed values change for the same cells (after a refit or a code update), the values are kept and the differing cells are marked.
  - **The file** is `{"schema":1, "frag-N-SEQ":{"range-K":"-16.59"}, …}`: `schema` (G8) plus the threshold mapping at top level, with true cells only and no other key (author). Its values are the table's. The energy-filter parameters that the file no longer carries are written by the export (A12).
  - Every true cell has a best-fit energy, because read validation refuses a true cell on a row with a scoring error (P5.6, D12). Copy can therefore fill every true cell, and X5 can compare each with its best-fit energy.
- **P5.8 Interaction hand-off.** It covers kept rows only:
  `{clip1, clip2, kept_residues, ranges:[{index, chain, start, end}], runs:[{range, first_fragment, last_fragment, anchors:[{fragment, nucleotide}]}], interactions:{"frag-N-SEQ":[ranges]}}`.
  The backend checks the shape. Checking the anchor cells is deferred to the export (author).
- **Gate 5→6:**
  - the matrix passes read validation against fresh energetics, a fresh fit and the current clip;
  - the backend's Gaussian scores and computed classes are fresh, and its golden check passes (O5);
  - the best-fit stacking geometry is fresh;
  - every true cell of the matrix has a valid value in the energy-threshold table, and its backend validator is not failing (author, D13). The computed values are not needed. A cap move, or a matrix adapted to a smaller clip, can add true cells without a value; the gate then stays closed until they are filled.

*Oracle for P5.3–P5.6:* table-driven tests covering:
- Recompute's stop-row exclusion, clip stop, same-range joining, and independent overlapping ranges;
- grid ↔ columns round trips;
- each refused-read case, including a true cell on a row with a scoring error;
- a Continue 3→4 that clamps a clip cutting a newly selected anchor;
- reuse on 5→4→5:
  - with the clip unchanged, the identical matrix comes back;
  - with the clip widened or narrowed, only the outer-end cells of P5.6 differ, and the result passes read validation;
  - with a smaller clip, the uncovered rows are grown by Recompute's rule at an end that touched the old boundary, and stay false at an end that did not;
  - a Continue 4→5 that races a clip edit leaves the stage at 4.

*Oracle for P5.7:*
- the computed values on 1b7f reproduce R(best-fit energy) for every cell, and a matrix edit does not recompute them;
- the table opens empty, Copy is disabled while the computed values are stale, and fills every cell with a value once they are fresh;
- 5→6 stays closed while a true cell is empty and opens once it is filled, whether by Copy or by typing;
- a protein-range edit followed by Continue 4→5 resets that range's values only; the values of the other ranges, and of a range that only changed its index, come back;
- a typed value at or below the best-fit energy is kept as typed, marked as differing, and makes the export show "Proceed anyway" (X5).

### Stage 6: Gaussian filter and reference anchors
- **P6.1 Model.**
  - Preparic's backend loads the model from A7. It requires `schema: 1` and all twelve names, and takes the names from the JSON, never from a hard-coded list.
  - Every Gaussian score and computed class is computed in the backend; the browser only reads them and computes no Gaussian (author, O5).
  - The backend checks the golden vectors at 1e-12 whenever it scores. A failure disables stage 6.
  - A changed `filter_set_version` is shown as "model changed".
- **P6.2 Descriptors.**
  - The best-fit candidates' descriptors, against the anchor's protein residue, are computed by the same code as stage 3's detection (author). The backend scores them with A7's model (author, O5).
  - There is no browser score, so nothing is compared at run time. *Oracle:* on 1b7f, these scores equal A9's `score` to 1e-6.
- **P6.3 Computed class, one per stacking anchor (author, D9).** The backend derives it, and the browser shows it (author, O5).
  - The class score is the lowest best-fit score among the anchor's unclipped candidates.
  - The class is the smallest N ∈ {50, 60, 70, 80, 90, 95} whose cutoff the score reaches. If even RRM-95 fails, the class is 95, shown in red as "below RRM-95".
  - **Auto prefilters,** for classes 90 and 95 only:
    - `tilt30` is ticked iff every unclipped candidate has θ < 30°;
    - `disp45` is ticked iff every unclipped candidate has φ < 45°.
  - **Filter name:** `rrmN` for N ≤ 80. For N ∈ {90, 95}, `rrmN` followed by `-tilt30` and/or `-disp45` for the ticked prefilters.
  - The reference score is shown but never sets the class.
- **P6.4 Editable class table.**
  - **Rows:** one per selected stacking.
  - **Validity:**
    - `rrm` is one of the six values;
    - the prefilter flags are present iff `rrm` is 90 or 95;
    - keys are unique, and the key set equals the selected stackings.
  - **Default:** a table set aside by an earlier Back 6→5 is restored by Continue 5→6. Otherwise Continue 5→6 records the computed classes.
  - **Copy, revert, diff:**
    - "Copy computed → editable" asks before overwriting edits;
    - each row has its own revert;
    - rows that differ from the computed values are marked.
  - **Switching class:** to N ≤ 80 drops the flags; back to 90 or 95 re-seeds them from the auto values.
  - **Kept** across refits (differing rows are marked) and model-version changes. Back 6→5 sets it aside, and the next Continue restores it (D1).
- **P6.5 Gaussian validation.**
  - A9 runs on its job list (§3.9), which does not depend on the class table, so it never re-runs on a class edit (P10). Merging its result with the classes is instant.
  - Each row shows the worst case over its candidates:

    | State | Display |
    |---|---|
    | cell emitted and pose passes | green, with the fill |
    | 0 < fill < ⅓ | amber |
    | fill = 0 | red |
    | fill > 0 but pose fails | amber |
    | running or stale | grey "validating…" |

  - The UI may suggest the strictest filter that still emits each candidate's cell.
- **P6.6 Reference-anchor validation.**
  - The author ruled that both anchor modes are always exported, so there is no mode choice. Every unclipped candidate of every selected anchor is checked.
  - thr is the editable bases value for that candidate; raw is the computed raw value.

    | Condition | Status |
    |---|---|
    | thr − raw ≥ 1e-5 | green |
    | \|thr − raw\| < 1e-5 | amber: inside Alaric's boundary zone |
    | thr ≤ raw − 1e-5 | red: the best-fit pose is rejected |
    | thr > L_ov (that is, thr ≥ 1.74) | the export is blocked (X6) |

  - **Above L_ov (author, D11).** The remedy is to lower thr to at most L_ov (1.73 or less). If the best-fit pose then fails, the export needs an explicit "Proceed anyway" (X5).
  - The panel shows, per candidate: sequence, computed threshold, editable value and status.
  - The editable value is the threshold table's own cell (P4.3), not a second copy, so stages 4 and 6 always show the same number.
- **Gate 6→export:** deferred. At least, every stage 6 result is fresh.

## 6. Export: obligations already fixed

The export stage is deferred (author). These obligations are already fixed.
- **X1. Same parameters.** The export uses the same exclude code and minimum cell fraction as the computations above. It does not set `ALARIC_ANCHOR_MIN_CELL_FRACTION`.
- **X2. Same RNA file.** `DATA/reference.pdb` is byte-identical to the RNA all-atom PDB that was fitted and displayed (P11), and its checksum is recorded.
- **X3. Both anchor modes.** Every selected anchor is exported in both modes, in parallel (author):
  - Gaussian `anchor` actions with `precomputed_filter_name: gaussian-rrm` and the row's filter name;
  - one `anchor-refe` action per unclipped candidate, with an explicit `ovrmsd` from the edited bases value (never `auto`), deduplicated.
- **X4. Other inputs.**
  - `grow` thresholds come from the edited pairs values of kept pairs.
  - The energy filter reads P5.7's file, which holds the energy-threshold table's values for the true cells. Its other parameters, `epsilon: 15`, `cascade: auto` and `min_cascade: auto` among them, are written into each action as listed in A12.
  - The export writes `DATA/score-filter.yaml` with exactly A12's default content.
  - The hand-off (P5.8) and the kept fragments travel with the export.
- **X5. "Proceed anyway".** When the nearest-to-native structure fails anything the export would apply, one checkbox appears next to the red bold text "The nearest-to-native structure doesn't pass the filters. Proceed anyway". This covers:
  - a Gaussian validation that is not green, or a class below RRM-95;
  - an edited ovRMSD or cRMSD threshold that does not exceed its raw value by at least 1e-5 (thr − raw < 1e-5): red, or amber inside Alaric's boundary zone (P6.6);
  - an energy threshold of a true cell that the best-fit pose does not meet: E_bestfit ≥ T, because the filter keeps a pose iff E < T (author, D13). Energies have no boundary zone, so there is no amber state here.

  Amber states trigger it like red ones (author, D13): 0 < fill < ⅓, a pose that fails, and a threshold inside the 1e-5 boundary zone.

  The browser computes this condition (author). When the checkbox is shown, the export proceeds only once it is ticked.
- **X6. Blocking, with no override.** The export is blocked while:
  - any result it relies on is stale, running or failed;
  - a decision it needs is undecided or invalid;
  - any ovRMSD threshold it would emit exceeds L_ov (author), whether an `anchor-refe` bases value or a `grow` pairs value of a kept pair (author, D11). The remedy is to lower the value; if the best-fit pose then fails it, the export needs "Proceed anyway" (X5).
- **X7. One consistent snapshot.** The export is computed from one consistent set of decisions. An edit made during the export leaves it visibly stale, never mixed.

## 7. Decisions needed from the author

Each item names what is open, the options, and the clause it changes. Where a plan made a recommendation, it is given. Every open item is now gathered, with this document's own proposals, in `contract-open-questions.md`; this list is kept for its IDs. Items the author has answered are marked **Answered**. Other decisions the author has already made are not repeated here; they appear as clauses above, marked "(author)". Those recorded in `final/state-transitions.md` and in the clarification on clip and matrix are now:
- the fit waits for stage 3 (P10);
- Continue is closed on an empty stacking table and on an empty selection (gates 2→3 and 3→4);
- stage 4 is about the threshold table, filled by Copy and/or typing (P4.2, P4.3);
- the clip is edited in stage 4 only and never beyond a selected anchor (P4.4);
- on 5→4→5 the matrix is reused, with only its outer ends adapted to the clip (P5.6);
- from stage 4 on, the viewer shows the backend's all-atom range PDBs, not the downloaded structure (P11).

1. **D1. Back keeps every stage decision** (P4; `final/state-transitions.md` §10.3, §11.3). **Answered:** yes, for every stage (author, §11.3). The matrix reuse requires this for the matrix. The draft applies it to every stage decision, so Back 4→3 keeps the threshold table, Back 6→5 keeps the class table, and Back 3→2 keeps the ticks. Confirm it for all stages, or limit it to the matrix. Limiting it would throw away the threshold table on every Back 4→3.
2. **D2. The clip and the anchors** (§1 clip, P4.4, P5.3; `final/state-transitions.md` §11.4b). **Answered:** the strict reading, and Continue 3→4 clamps the clip (author, `contract-open-questions.md` D2a, D2b).
   - **What "beyond a stacking anchor" means.** Options:
     - **Strict (recommended):** no candidate fragment of any selected anchor may be clipped. Locks then never change with the clip, so the matrix reuse touches only the ends of runs. This matches the draft's "the control stops at the outermost anchor fragments".
     - **By residue:** only the anchor's residue must stay among the kept residues. The anchor's outer candidate may then be clipped: the fragment with the anchor as `second` at the 5' end, or as `first` at the 3' end. That anchor's lock then shrinks to one fragment when the clip changes.
   - **An anchor newly selected inside the kept clip.** The author ruled that clipping beyond an anchor is not allowed, and that the user should first disable the anchor in stage 3. That settles clip edits, but not this reverse case. Options:
     - flag the clip in stage 4 and keep Continue 4→5 closed until the user reduces it (the draft); or
     - have Continue 3→4 clamp the clip to the outermost anchor automatically, at the cost of one more awaited write.
3. **D3. Filling uncovered rows when the matrix is reused** (P5.6; `final/state-transitions.md` §11.4a). **Answered:** (a) (author). When a smaller clip uncovers rows at an outer end, which of them become true? Options:
   - **(a), recommended:** apply Recompute's growth rule to the new rows only (|E| ≥ 1, stopping at error rows). Do so only if the column's end touched the old clip boundary, that is, if the clip had cut the run. A run that ended before the boundary stays as the user left it. This gives what Recompute would have given at the cut ends, and keeps the user's caps elsewhere.
   - **(b):** the end run simply extends to the new boundary, with every new row true.
   - **(c):** leave the new rows false; the user extends the run by hand.
4. **D4. Clipped rows and the 4→5 gate** (P4.3, gate 4→5; `final/state-transitions.md` §11.5). **Answered:** rows of clipped fragments are not required (author, confirming the draft). The draft requires only unclipped values to be filled. Should clipped rows be required too? If they are not, reducing the clip later in stage 4 uncovers empty rows, and the gate closes again until they are filled.
5. **D5. The empty table on entering stage 4** (P4.3; `final/state-transitions.md` §11.6). **Answered:** explicit Copy, with the computed values clearly visible (author). Should the table start empty, with Copy one click away, or be prefilled from the computed values when those are already fresh? Prefilling makes the stored default depend on timing: two tabs racing Continue 3→4 could write different defaults.
6. **D6. Opening stage 4 before the fit** (A2, P4.1; `final/state-transitions.md` §11.7). **Answered:** fragment keys from A2 are acceptable (author). The draft takes the fragment keys from the reference fragments (A2), which do not wait for the fit. A2 now must yield the same key set as the fit, so this is safe once A2 meets its clause. Confirm, or require the keys from the fit. With keys from the fit, stage 4 opens only after the fit, which takes minutes.
7. **D7. Assembly copies and range atoms** (P2.1). **Answered:** (b), the backend applies the assembly operators (author). Original question (P2.1; this also answers the fact behind `final/state-transitions.md` §11.9, where the author kept `biounit` in M2's basis because chains may be duplicated). Found in preparic's code:
   - The backend never receives the biounit. It extracts asymmetric-unit chains by label ids and strips copy suffixes, so `A`, `A-1` and `A-2` give the same atoms.
   - The browser names chains, validates rows and highlights picks against the selected assembly.

   So the biounit changes which rows are valid, but not which atoms are used, and a picked symmetry copy is computed at its asymmetric-unit position. Options:
   - (a) Ranges refer to asymmetric-unit chains only. Copy suffixes are refused, and rows are validated against the asymmetric unit. The biounit becomes display-only and no longer invalidates the ranges.
   - (b) The backend applies the assembly operators. The biounit becomes a real input of range extraction, and the ranges stay tied to it.

   Under P11, a range picked on a copy is shown from stage 4 on at the position the backend extracted. That makes the discrepancy visible, but does not settle D7.
8. **D8. Kabsch normals in stage 3** (P3.2; `open-questions.md` stage 6 Q5). **Answered:** use Kabsch normals (author). Should stage 3 switch to Kabsch ring normals, so that its slip (φ) and tilt agree with stage 6? They differ by up to 0.7° today. If yes, the stage 3 numbers change and the reference rings become a stage 3 input (A10).
9. **D9. Gaussian class granularity** (P6.3; stage 6 Q1). **Answered:** one class per stacking anchor, as recommended (author). Every plan assumes one class per stacking anchor, scored on the worse of its unclipped candidates. The alternative is one class per candidate, that is, per fragment × first/second.
10. **D10. Which pose stage 5 scores** (P5.1; stage 4 Q3, adopted without confirmation). **Answered:** the discrete best-fit library pose, as recommended (author). The plans score the discrete best-fit library pose, which is what docking can sample and what the existing `.score` fixtures use. The alternative is an exact superposition.
11. **D11. ovRMSD above L_ov = √3 Å** (A11, P6.6, X6). **Answered** (author): the block also covers `grow`'s pair ovRMSD; the remedy is to lower the threshold, with an explicit "Proceed anyway"; Alaric raises above √3. The author ruled that `anchor-refe` thresholds above the limit block the export. The three points were:
    - **Scope.** Does the block also apply to `grow`'s pair ovRMSD? It clamps in exactly the same way.
    - **Consequence.** Both anchor modes are always exported, so a single anchor candidate above √3 blocks the whole export. End fragments do reach such values (1b7f fragment 11, nucleotide 2: 2.246 Å). The clip can no longer remove an anchor's fragments, so the user's remedies are:
      - lower the threshold, which fails the best-fit pose and requires "Proceed anyway"; or
      - deselect the anchor in stage 3.

      Is that the intended workflow?
    - **Alaric's side.** Should Alaric raise on an `ovrmsd` above √3 instead of clamping silently, or extend its offset tables? Extending would move L_ov and lift the block.
12. **D12. Interacting cells without an energy** (P5.7). **Answered:** the editor forbids it and read validation refuses it (author; P5.5, P5.6). Can a run cover a fragment whose scoring failed? Recompute never grows into one, but the editor does not forbid it, and such a cell has no energy to derive a threshold from. Options:
    - the editor forbids it and read validation refuses it (recommended); or
    - the export is blocked on such a cell.
13. **D13. What triggers "Proceed anyway"** (X5). **Answered** (author): amber states trigger it like red ones. Energy thresholds are user-editable, in a table that works like stage 4's: it starts empty, is filled by explicit Copy and/or typing, and needs a value in every true cell before stage 6. One range's values reset only when that range changes (P5.7).
    - Energy thresholds are computed only (P5.7), so the best-fit pose always meets them. Is the energetics trigger meant for user-editable energy thresholds, which no plan has? If so, should they be editable like the RMSD thresholds?
    - Do amber states trigger the checkbox like red ones? These are fill < ⅓, a pose that fails, and a threshold inside the 1e-5 boundary zone.
14. **D14. A persistent result cache** (P2; `final/state-transitions.md` §11.10). **Answered:** preparic runs with no persistent result cache (author's execution decision, `draft-implementation-plan.md` §1), and a restart is out of scope for state transitions (§11.10). Only mounted cells survive it, and the browser must restart right afterwards (author, O6; P2). Without one, every backend restart from stage 3 onwards re-runs the fit and the tables: minutes, with an 8 GB library load. Stage 4 stays editable meanwhile, but Continue 4→5 waits. Should preparic's Seamless configuration keep results across restarts?
15. **D15. Gaps in the RNA range** (§1; A12 "one fragment numbering"). **Answered:** (a) for preparic. In Alaric, fragment N is the Nth and (N+1)th residue, whatever the residue numbers: tools that assume residues N and N+1 are fixed, and `Reference` renumbers ordinally (author, D15, O3). Original question: Alaric numbers fragments from the first residue present and leaves gaps where residues are unobserved. Preparic's plans number from the range's `start`. The two disagree when the range's first residue is unobserved, and the clip, pair and skeleton rules must cope with missing fragment numbers. 1b7f has no gap. Options:
    - (a) Refuse, at the stage 2 gate, an RNA range with any unobserved residue (the simplest fit with "no lenient flags").
    - (b) Support gaps throughout: fragment numbers with holes, clips by fragment number, and a residue mapping taken from the fragments.

    Either way, Alaric's `sequence: auto` and `rmsd.py --fragment` must be fixed (A12), or the export must never rely on them.
16. **Go-ahead items carried over from `open-questions.md`.** **Answered:** go ahead, and add them to the implementation plan (author). They are tasks of `draft-implementation-plan.md` W4 (publication and pointer bump) and W8 (the 10-12 fix).
    - Push attract-jax `9453ee6` and `22b23e4`, commit its untracked accuracy data, and bump Alaric's submodule pointer from `12bba50`. A5 cannot run from Alaric's pinned attract-jax until then.
    - Fix the 10-12 grid bug (a bead exactly on a grid node). It is a prerequisite of the energy-filter cascade.

    The former question about "skip 10-12 if less than 1000 poses" is settled by `min_cascade` (A12): the count is checked before every step, against the poses still undecided.

**Not decisions.**
- `final/state-transitions.md` §11.4 (clip and matrix) is settled by the author's clarification (P4.4, P5.6). Its two remaining sub-points, D2 and D3, are now answered.
- §11.8 (is the whole table editable in stages 4–6?) is answered by the author's own wording in `open-questions.md`: "the editable table remains visible and editable throughout". P4.3 follows it.
- §11.10–11.13 have been answered by the author: a restart is out of scope for state transitions; no assumption is made about withdrawing queued writes, and a write of the correct state is always queued; Seamless guarantees that equal-value writes revoke nothing and that read-only Alaric mounts are polled. They affect how P3–P5 and P9 are met, not what they require.
- The delivery form of each Alaric step (§3.0) is the implementer's choice, not the author's.

## Appendix A. Where each clause comes from

Abbreviations:

| Short | Document |
|---|---|
| OQ | `open-questions.md` |
| CB | `final/contract-boundaries.md` |
| ST | `final/state-transitions.md` |
| S4 | `stage4-best-fit-miniplan.md` |
| TH | `stage4-thresholds-howto.md` |
| EN | `energetics-miniplan.md` |
| EF | `alaric-energy-score-filter-plan.md` |
| AJ | `attract-jax-numpy-scoring.md` |
| AG | `alaric-gaussian-miniplan.md` |
| PG | `preparic-gaussian-miniplan.md` |
| GS | `gaussian-filter-sources.md` |
| AR | `anchor-ring-ovrmsd-mode.md` |
| TA | `test-anchor-miniplan.md` |
| PA | `preparic-architecture.md` |
| src | a source check made for this document (Appendix B) |

The third column names one way to meet each clause, using the T-, S- and B-numbers of CB §2 and §8. These are not part of the contract.

| Clause | Sources | One way to meet it |
|---|---|---|
| G1 | CB §1 Location; OQ (no remote workers) | `preparic/alaric_dir.py`, `preparic-start --alaric-dir` |
| G2, G3 | CB intro, §1, §4 rules | Python transformers with attached modules; `seamless-run` bash transformers |
| G4 | CB §1 | `fraglib_identities` pin, `META__FILE__fraglib.yaml`, `precalc` checksums |
| G5 | S4 inputs; CB shared pin cells | B5 `exclude_pdb_code` |
| G6 | CB data rules, T3; S4 | T3 `aareduce_ranges`; B4, B6 |
| G7 | CB §1 data rules; OQ energy Q2, test-anchor Q4; TH §3; TA §1 | `util/refe-thresholds.py`, `alaric/energy_threshold.py` |
| G8, G9 | CB §1 schemas, §5.1, §8.1; author (I7) | startup loaders that keep diagnostics; consumers that raise |
| §3.0 | CB §1, §4, §7.1 | module or CLI + DEPS; factories for T5 and T13 |
| A1 | CB T5; S4; TH §1; src | `seamlessify_refe_best_fit` with unbound pins |
| A2 | CB T6; S4 "PDB sets"; src | T6 `refe_fragments` |
| A3 | CB T7; TH §2, §5; AR §4 | T7 `best_fit_tables` |
| A4 | CB T8; TA §1 | T8 `best_fit_thresholds` |
| A5 | CB T13, §4; EN; AJ | T13 `energetics_range_<k>`, `energetics_seamless_support.py.DEPS.txt` |
| A6 | CB T16; EF §6 | `alaric/energy_threshold.py` attached to T16 |
| A7 | AG §1–5; GS; CB §5.1; OQ Gaussian Q1–2 | `alaric/gaussian_stacking_filter.json` |
| A8 | CB §5.1; OQ Gaussian Q2–3; AG §1, §4 | `util/gaussian-precalc.py` |
| A9 | CB §5.2; OQ stage 6 Q4; PG §7.1–7.2 | T21 `gaussian_check` over `alaric/anchor_precalc.py` |
| A10 | PG §1; CB T4, T18, T21 | B16 `ring_refe` |
| A11 | CB T23, §5.1; TA §4; AR §6; src | constants mirrored in T23, T21 |
| A12 | EF §4, §8; OQ energy Q1, Q4; AG §6; AR §1–3; src | `energy-filter` action; `anchor.py`/`resolve.py` name validation |
| P1, P2 | ST §2, §8; CB §8.1, §8.5; author (O6) | file-authoritative mounted edit cells; mounted `cif`; browser restart after a backend restart |
| P3 | ST §3, §7; author (O7, D17) | stage derived from marker cells with recorded bases; ordered orphan cleanup; no protection against racing tabs |
| P4 | ST §3.4, §4.1–4.4, §7.1 | single-write Continue/Back; `active` flag in markers; orphan deactivation |
| P5 | ST §6; CB §8.4 | echoes in computed cells |
| P6 | ST §5.4; CB §8.4 | state graph, banner scope |
| P7, P8 | CB §8.1; ST §1, §9.1 | null as the only empty value; validators T10, T19 |
| P9 | ST §7, §8 | load overlay; URL rules |
| P10 | ST §5.3, §10.1; CB T20 | group-B gate; T20 independent of S18 |
| P11 | author (2026-10-29); CB T25, §8.6.5 | T25 `share_range_pdbs_aa` → S23 `range_pdbs_aa`, echoing S5 and B5 |
| Stage 1 | ST §4.2, §7.1, §9.1; CB §8.6.1 | S21 `structure_id`; arbiter rule |
| Stage 2 | CB §8.6.2; ST §4.2; src | T0 `parse_ranges` |
| Stage 3 | ST §4.2; PA; AG §5 | S7, S8 keyed by stacking key |
| Stage 4 | S4; TH; OQ stage 4; ST author decisions, §3.4, §4.2–4.4, §9.1.5 | T5–T11; S9–S12; S22 `fragments` via T24; S11 as the stage-4 marker |
| Stage 5 | EN; OQ stage 5; EF §6; CB T12–T16; ST §4.2 (4→5), §7.1; author clarification on clip and matrix | S13–S15, S24, B14; matrix adapted on Continue 4→5 |
| Stage 6 | PG; TA; OQ stage 6, test-anchor; CB T18–T23, T28 | S17–S20, S25 |
| X1–X7 | OQ stage 6 Q2–3, test-anchor Q1 and Q5; CB §5.1, §7.3, §8.6.4; TA §5; PG §5; ST §10.5 | deferred export stage |

## Appendix B. Facts checked in the source for this document

These facts were read from the Alaric and preparic source on 2026-10-29. They are not recorded in any other plan.

1. **ovRMSD coverage** (A11, D11).
   - `GRID_SPACING = √3/3` and `MAX_OVERLAP_RMSD = 1.3` (`alaric/grow.py:31-32`).
   - `_precompute_translation_sets` builds tables for 0, 0.577, 1.155 and 1.732 Å; the last step is added after the loop ends (`grow.py:268-310`).
   - `_select_translation_set_index` clamps larger radii to the last table (`grow.py:318-320`).
   - The only validation is `ov_rmsd > 0` (`grow.py:1246`, `anchor_refe.py:272`, `middle/schema.py:162-166`).
   - `anchor-refe` imports the same table builder and kernel.
   - `crmsd` is a lookup in a dense matrix, with no such limit.
2. **Fragment numbering** (§1, A12, D15).
   - `Reference.get_fragment_positions` numbers fragments as `first_resid − resids[0] + 1` and skips windows whose residue numbers are not consecutive (`alaric/reference.py:102-111`). `_get_pos` resolves fragment N the same way (`reference.py:131-144`).
   - The fitter, `refe-best-fit-bases.py` and `anchor-refe` (`anchor_refe.py:177-211`) therefore agree.
   - `refe-best-fit-pairs.py` pairs only rows whose fragment numbers are consecutive (`:241-243`).
   - `middle/resolve.py:39-65` (`sequence: auto`) and `sequence.txt` are positional, and `rmsd.py:290-300` uses absolute residue numbers. Both disagree with the fitter after a gap.
   - A chain break with consecutive residue numbers is not detected.
3. **The best-fit `rmsd` column** (A1, A2). It is the in-place RMSD, with no superposition, of the pose conformer · R + grid · √3/3 against the reference fragment, over all heavy atoms of both nucleotides in template order, printed with `:.6f` (`util/refe-best-fit.py:136-153, 266-269`). Recomputed from the written PDBs on 1b7f fragments, it agrees to about 2e-5.
4. **Biounit and ranges** (P2.1, D7).
   - The backend never receives `biounit`. `ranges_to_pdb` takes only `cif` and `range_specs` (preparic `backend.py:146-147`).
   - The uploaded CIF is the RCSB asymmetric-unit file (`viewer.ts:182-200`).
   - `range_pdbs.py:73-82` strips copy suffixes such as `-2` and extracts asymmetric-unit chains by label ids.
   - The browser names chains per assembly copy (`chains.ts:21-25`, `viewer.ts:244-270`) and validates rows against the selected assembly (`ranges.ts:63-78`).
