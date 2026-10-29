# Preparation payload contracts, schema 1

The canonical bundle is [schemas.json](../contracts/preparation-v1/schemas.json),
with a deterministic [builder](../contracts/preparation-v1/build_artifacts.py),
one shared [fixture corpus](../contracts/preparation-v1/fixtures.json),
[golden references](../contracts/preparation-v1/golden-references.json), and
[active inventory](../contracts/preparation-v1/inventory.json). These specify
representations for the settled W0 boundaries. They do not install consumers
in Preparic or implement Alaric's future scientific steps.

The bundle uses JSON Schema draft 2020-12. Select an entry with
`schemas.json#/$defs/S14`, for example. Payloads have `schema: 1` and reject
unknown fields. Definitions without an S/B prefix are reusable embedded shapes,
not independent boundary documents. JSON numeric equality applies: schema `1`
and `1.0` are equivalent; `true` is not schema 1. Boolean grids and active flags
accept booleans only.

## Exact representation choices

These names instantiate the existing contracts; they add no author policy.
The JSON Schema is authoritative for required fields and types. All arrays
have semantic order unless explicitly described as a key set. Object property
order has no significance.

| Embedded shape | Fields and meaning |
|---|---|
| Range row | `{chain: string, start: integer, end: integer}`; normalized author residue IDs, inclusive. Range identity is this whole row, independent of its ordinal position. |
| Project basis | `{pdbcode: string, biounit: positive integer}`. |
| RNA decision basis | `{pdbcode: string, rna: range-row}`; `rna` spells the contracts' “range-1 row”. |
| Residue identity | `{chain: string, resid: integer, insertion_code: string, base: A/C/G/U}`; empty insertion code is explicit. |
| Fit echo | `{pdb_code: string, rna: range-row, resids: residue-identity[]}`; `pdb_code` is lower-case B5. |
| Energy / descriptor echo | `{best_fits: fit-echo, protein_ranges: range-row[]}`. |
| Score echo | `{bestfit_stackings: energy-echo, selected_keys: string[], clip: S12}`. |
| Class edit | `{key, rrm}` for 50/60/70/80, or `{key, rrm, tilt30: boolean, disp45: boolean}` for 90/95. No flags on smaller classes. |

The project spelling `pdbcode` follows stored decision bases; computed `pdb_code`
follows T25's specified echo. `rna` avoids treating a changing range index as
identity. Selected-key bases are unique string arrays in canonical key order.
The S8 range basis is a normalized snapshot `{pdbcode, biounit, rows}`. Range
snapshots carry assembly identity because normalized rows depend on it.

| Boundary | Exact top-level fields after `schema` |
|---|---|
| S4 `ranges` | `basis: project-basis`, `active`, `rows: range-row[]`. The underlying share remains YAML. |
| S5 `range_specs` | `pdbcode`, `biounit`, `rows`; computed, read only. |
| S7 `stackings` | `echo: {structure_id,pdbcode,biounit,rows}`, `rows: stacking[]`. |
| S8 `stacking_selection` | `basis: {pdbcode,biounit,rows}`, `active`, `selected: {stacking-key: boolean}`. |
| S9 `best_fit_thresholds` | `echo: fit-echo`, `pairs`, `bases`; every threshold also has its numeric `_raw` field. |
| S10 `best_fits` | `echo: fit-echo`, `rows: {key,fragment,sequence,conformer,rotamer,grid,pdb}[]`. |
| S11 `threshold_edits` | `basis: RNA-basis`, `active`, `pairs`, `bases`; null means a blank individual threshold. |
| S12 `best_fit_clip` | `basis: RNA-basis`, `clip1`, `clip2`; counts are nonnegative integers. Startup only: `{schema:1,basis:null,clip1:0,clip2:0}`. It always exists and has no active flag. |
| S13 `energetics` | `echo: energy-echo`, `rows: {fragment: fragment-key,range,refe: number/null,bestfit: number/null,error: string/null}[]`. |
| S14 `interaction_matrix` | `basis: selected-key[]`, `active`, `rows: fragment-key[]`, `ranges: range-id[]`, `grid: boolean[][]`. |
| S15 `energy_thresholds` | `echo: energy-echo`, `values: {fragment-key: {range-id: two-decimal-string}}`; explicit Copy source only. |
| S17 `bestfit_stackings` | `echo: energy-echo`, `rows: stacking[]`; descriptors only. |
| S18 `gaussian_classes` | `basis: selected-key[]`, `active`, `rows: class-edit[]`; persisted edits, distinct from S25. |
| S19 `gaussian_validation` | `echo: {classes,clip,selected_keys,best_fits}`, `rows: {key,state,messages: string[]}[]`. States: `green`, `amber`, `red`, `blocked`. |
| S20 `refe_validation` | `echo: {bases,clip,selected_keys,best_fits}`, same validation row shape. |
| S21 `structure_id` | `structure_id: string`. |
| S22 `fragments` | `echo: RNA-basis`, `residues`, `fragments`, `adjacent_pairs`, `candidates`; see below. |
| S23 `range_pdbs_aa` | `pdb_code`, `range_specs: S5`, `pdbs: {range,name,pdb}[]`; PDB strings pass through byte-for-byte. |
| S24 `energy_threshold_edits` | `basis: RNA-basis`, `ranges: {row: range-row,values: {fragment-key: null/two-decimal-string}}[]`. No active flag. |
| S25 `gaussian_scores` | `echo: score-echo`, `filter_set_version`, `legal_filter_names`, `candidates`, `references`, `classes`; see below. |
| B17 `gaussian_check_jobs` | `best_fit_keys: fragment-key[]`, `jobs: {key,fragment,nucleotide,sequence,conformer,rotamer,grid,range,protein_resid}[]`. |
| B18 `gaussian_check` | `filter_set_version`, `min_cell_fraction`, `jobs: {key,fragment,nucleotide,error?,filters}[]`. |
| `interactions` handoff | `clip1`, `clip2`, `kept_residues`, `ranges: {index,chain,start,end}[]`, `runs: {range,first_fragment,last_fragment,anchors: {fragment,nucleotide}[]}[]`, `interactions: {fragment-key: range-id[]}`. |
| `energy-thresholds` export | Exactly `schema` plus top-level `frag-N-SEQ: {range-K: two-decimal-string}` entries. No metadata or `thresholds` wrapper. |

S1 is text, S2 is a plain positive integer, and S3 is raw CIF text. These retain
their declared celltypes rather than acquiring JSON envelopes. S6 is an
unshared internal `list[str]`. Writable decisions and computed shares support
`null` where their schema explicitly allows it. S12, the exports and Gaussian
job/result documents do not. `null`, an inactive marker, and an empty valid
selection have separate fixtures. An empty shaped matrix is representable but
fails consumption when it does not match the current extent.

The inventory preserves T0–T16/T18–T28 and S1–S15/S17–S25 and their existing
names, including T27/S24 and T28/S25. T17/S16, activation tokens, parent tokens,
validation receipts and extra cross-result model/ring identity envelopes are
absent. T10/T19/T27 results stay unshared. `spec_digest` and `npy_sha256` keep
their existing scientific model/cache roles; no new comparison gate is added.

## Canonical fragment and Gaussian shapes

A fragment row is `{number,key,sequence,residues:[identity,identity]}`. Its key
is `frag-N-SEQ`, and `number` is the ordinal window among residues present,
independent of residue IDs. `adjacent_pairs` holds `{down,up,sequence}` for
existing adjacent fragments. Candidate membership is
`{key: stacking-key,fragment: fragment-key,nucleotide: 1|2}`. It is explicit so
all consumers use the same windows and nucleotide assignments.

The non-1-start fixture begins at residue 101. The gapped Alaric fixture has
residue IDs `[101,102,104,105]`, fragment numbers 1 and 3, and no window 2.
Preparic stage 2 refuses ranges with unobserved residues. The small numbering
check covers the corpus's integer-residue cases; it is not a general PDB
connectivity inference algorithm. Production consumers use the canonical map's
explicit adjacency and candidate membership. W8 must fix actual Reference
numbering before gapped A2/A1 acceptance.

Pair threshold rows are `{down,up,sequence,ovrmsd,crmsd}`. Base threshold rows
are `{fragment,sequence,nuc1_ovrmsd,nuc2_ovrmsd}`. Stored RMSD thresholds are
positive decimal strings with at most two decimal places, or null in S11.
S9 also requires `ovrmsd_raw`/`crmsd_raw` or
`nuc1_ovrmsd_raw`/`nuc2_ovrmsd_raw`, respectively. Raw values remain numeric;
they are never replaced by rounded values.

A stacking row carries `{key,range,protein_resid,descriptor,candidates}`.
`descriptor` is `{d,phi_deg,theta_deg}` from backend Kabsch ring geometry.
`grid` in fit/job rows is exactly three integers; conformer/rotamer are discrete
library indices. Fragment numbers in jobs are integers, not residue IDs.

S25 candidate scores are `{key,fragment: fragment-key,nucleotide,score}`;
reference scores are `{key,score}`. Each computed class is
`{key,class_score,rrm,auto_prefilters:{tilt30,disp45},filter_name,reference_score,below_rrm95}`.
The backend chooses the worst **unclipped** candidate, the smallest reached RRM
cutoff, and strict all-candidate prefilters. The reference score is display
information. S25 contains all twelve legal names and `filter_set_version`.
Names in the generated schema come directly from the authoritative model JSON;
there is no independently maintained TypeScript filter list.

Each B18 filter entry is either
`{count,capacity,fill,emitted,pose_pass,score}` or the hierarchy shortcut
`{emitted:false,implied_by: filter-name}`. A job may carry `error` and an empty
filter map. Filter names and shortcut targets are restricted to the delivered
model's names. Numerical consistency of count/capacity/fill, hierarchy and
scientific scores is future backend acceptance, separate from shape validation.

## Validation, actions and gates

[validate.py](../tests/contracts/validate.py) and
[validate.ts](../tests/contracts/validate.ts) implement the subset of JSON
Schema keywords used by this bundle, rejecting unknown keywords. They are
lightweight contract checks, not general JSON Schema libraries. Boundary
consumption checks version first, then shape, then applicable semantic rules;
unsupported documents never have their fields partly consumed. JSON parsing,
shape and semantic diagnostics are deterministic fixture categories. The
loader preserves malformed input as raw fixture text plus its diagnostic
before normalization; legacy decisions without a basis become undecided with
an explanation and are never migrated.

The semantic operation helpers have an explicit precondition: inputs are
nonnull decisions admitted by the current active stage prefix, with current
canonical context supplied by the caller. They do not implement sentinel
projection or derive the wizard stage from inactive/null markers. Loader and
standalone schema fixtures separately test those representations.

Shape validation does not establish freshness, completeness, current identity
or scientific correctness. Separate pure checks cover fragment windows,
ordered addressed matrix extent/columns, strict booleans, selection basis,
locks, per-run lock coverage, scoring errors, class selection/basis membership,
and duplicate candidate/class addresses. An error on a selected
anchor's lock refuses the matrix and requires deselection in stage 3. The
interaction handoff itself receives **shape validation only**; full anchor
checks belong to matrix read validation and later export.

The matrix adaptation fixtures preserve old interior cells, expand only at an
outer end touching the saved boundary, use supplied growth eligibility for new
rows, and leave a user-stopped end untouched. Growth eligibility stands for
fresh per-range energies/errors; these tests do not call a scorer. Clip
clamping protects all selected-anchor candidate fragments on Continue 3→4.
The small clamp helper receives already resolved ordinal extrema.

S24 starts empty. Explicit Copy requires fresh S15 values and confirmation
before overwriting existing values. Typing accepts decimal text of either sign
with at most two places; it pads to exactly two stored places **without
rounding**. More places, nonfinite text and exponent notation are rejected.
Thus UI input `-16.5` persists as `-16.50`, satisfying T27's exact string shape.
The export retains exactly two places. Range reconciliation keys by the full
protein row, preserving values across reordering/refits and resetting only a
changed range. A project/RNA-basis change empties the whole table.

Threshold completeness checks expected base and adjacent-pair identities from
the canonical map, requiring only kept rows. Missing kept rows cannot pass by
vacuous filtering. Energy completeness requires a valid stored value for every
true matrix cell; false/clipped cells may stay blank. The gate fixtures express
all other readiness requirements separately. Failed T10 vetoes 4→5, failed T27
vetoes 5→6. A nonfailed validator does **not** establish freshness or require a
receipt/complete acknowledgment. Pending/blocked nonfailed examples pass only
the failure-veto subcondition; other gate conditions remain independently
required. Red and amber export conditions require explicit Proceed anyway,
including a lowered energy threshold rejecting its best-fit energy.

## Running and evidence limits

From the Alaric checkout:

```sh
/home/agent/miniforge3/envs/preparic/bin/python tests/contracts/run_contract_checks.py
```

The runner uses Python's standard library, Node.js, and TypeScript. It finds
TypeScript through `--tsc PATH`, `CONTRACT_TSC`, `PATH`, then sibling
`preparic/node_modules/.bin/tsc`. Missing tools produce an actionable diagnostic;
no packages are installed. TypeScript is compiled in a temporary directory,
and both languages read the same JSON corpus and must return exactly the same
results. Compilation uses `--strict --target ES2020 --module commonjs`.

The run on 2026-10-29 passed **261 shared cases** in Python and TypeScript,
JSON round trips, inventory coverage, and pinned golden-reference metadata.
The parent agent may add review cases before committing; the runner prints the
current authoritative count. Generated artifacts can be recreated with:

```sh
python contracts/preparation-v1/build_artifacts.py
```

Regeneration updates the model source pin and enum names, so review that diff
before accepting an upstream model change. The runner hashes only the small
model JSON and checks its declared version, `spec_digest`, tolerance, golden
pointers and counts. It never hashes/decompresses Gaussian caches or invokes a
scientific module. Delivered score/acceptance/lattice goldens are referenced,
not copied, regenerated or numerically executed. Rounding examples reference
the settled strict-successor rule: `-16.5913 → -16.59`, exact `-16.59 → -16.58`,
and `0.123 → 0.13`; the rounding implementation remains Alaric's future A4/A6.

Classification boundary declarations cover cutoff equality, worst unclipped
candidate, ignored clipped candidates, strict prefilter equality, and the
below-RRM95 red fallback. Checks validate their legal-name references; they do
not derive scores or classes. Python must evaluate the delivered numerical
fixtures when T28 is delivered, at relative `1e-12`; its 1b7f comparison with
A9/T21 at `1e-6` is a future backend delivery test. The browser only consumes
S25 and edits S18, with no scorer, classifier or numerical self-test.

No actual A2/A4/A6 output acceptance, W1 browser integration, T28 self-tests,
A9/T21 parity, fitter/library loading, scientific generation, services, mounts,
or concurrent-tab protocol proof is claimed. A7/A8 remain accepted by the
author's earlier ruling, not by a new numerical test here. Runtime restart
correctness still assumes an immediate browser restart; idle-tab following
remains W1 work and racing edits remain outside the guarantee. The separate
prefilter-thresholding investigation is untouched. No new author ruling was
needed to define this package.
