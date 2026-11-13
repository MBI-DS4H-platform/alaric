# Export: DATA and fragment pool generators ("gens")

Implementation record and contract specification, 2026-11-07. §3 retains the generator rules; §§4–7 specify the archive, snapshot, backend/frontend integration and Alaric prerequisites; §§8–9 retain delivery and acceptance evidence; §10 is the independent 1B7F catalogue oracle. E0–E7 are implemented and verified; E7 acceptance is recorded in [export-gens-e7-results.md](export-gens-e7-results.md).

**Scope.** This specification describes the implementation in the Alaric and Preparic checkouts. The generator rules in §3 are requirements. Keep the two export transformers, the small manifest and Seamless-owned download packing. Do not reintroduce custom GET callbacks, backend capture hooks, polling generations or an exporter-written ZIP. Only change Seamless if E0 demonstrates a violation of its documented download contract; report that finding before designing a workaround.

**Relation to existing documents.** [final/oracle-contract.md](final/oracle-contract.md) §6 records the delivered export behavior and X1–X7 protections. This document supplies the detailed archive and generator contract. [final/contract-boundaries.md](final/contract-boundaries.md) records the cell and ownership boundaries. E7 updated both finalized contracts and the usage documentation to reflect the implementation. The old `alaric/export_bundle.py` API is retired; it is not an API to extend or call from Preparic.

**Transport decision.** The export is computed inside Preparic's Seamless graph and served by a Seamless **download** ([Seamless shares contract](../../seamless1/seamless/docs/agent/contracts/shares.md), *Downloads*). Two Preparic transformers compute it: `export_admission` (a shared plain cell holding readiness, hard blocks, X5 reasons and the snapshot digest) and `export_tree` (a `folder` cell holding every archive member). `build_context()` registers `ctx.shares.add_download({"alaric_project": "export_tree"}, path="export.zip", format="zip")`. A GET packs the buffers of one consistent cut and runs no Preparic code. The earlier `register_get_endpoint` callback was rejected in Seamless and is not used. No other web server, writable export-trigger cell or filesystem export destination is added.

## 1. Vocabulary

| Term | Meaning |
|---|---|
| Fragment N | The dinucleotide covering nucleotides N and N+1. |
| Range | A Preparic range PDB, `range-K` (`range-1` is the RNA; protein ranges are `range-2`, …). Written as A, B in examples. |
| Pool | The output of a pose-producing action. |
| Gen | Fragment pool generator: a folder of action dirs with final pools, named `fragN` (merge gens: `fragN-merged`; common-conformer gens: `fragN-rangeX-common-conf`; R0), and zero or more intermediate pools. Most gens have one final pool; an anchor-run gen of K nucleotides has up to K+1 before chain boundaries/clipping (`gen-frag6-7`: `frag6`, `frag7`), and a common-conformer gen has 2. |
| Main folder | The project root holding `DATA/`. The user builds a run by moving a gen's action dirs into it. |
| Gen-final vs run-final | A gen's final pool is final only within the gen. In the run it is an intermediate: the run's final pools are what remains after merging and dead-end removal. |
| Anchor run | Consecutive anchored nucleotides N … N+K−1 (see R3). |
| Inner / outer fragment | Inner: both nucleotides anchored (frag N … frag N+K−2). Outer: one nucleotide anchored (frag N−1, frag N+K−1). |
| Gen type | Anchor, grow, merge or common-conformer (R2). |
| Mode | Fixed or unfixed: whether the relative orientation of the range PDBs is assumed known (R6). |

## 2. Delivered implementation

- Preparic's pure `export_snapshot.py` normalises the transformer pins, performs admission checks, calculates backend-owned warning records and computes the canonical snapshot digest.
- `export_gens.py` plans the generator catalogue. `export_tree.py` verifies admission and digest agreement, then renders the DATA files, schema-1 manifest and action YAML into an unshared folder mapping.
- The shared plain `export_admission` cell and unshared `export_tree` folder are wired in `backend.py` from the same explicit pin registry and attached source modules. The existing Seamless share server serves `export.zip` from one consistent cut; it does not run an exporter callback for the GET.
- Stage 6 shows readiness and backend-supplied warnings, binds acknowledgment to the snapshot digest, verifies the manifest before releasing a ZIP, cancels invalidated requests and marks an earlier download stale.
- The Alaric action contract includes `apply-mask`, one-input `identity`, and schema-1 `constraints.json` resolution. The old synthetic bundle API is superseded and is removed after its remaining useful assertions transfer to real input fixtures and export trees.
- The archive has `DATA/` and nested generator folders under `alaric_project/`. Alaric loads only action directories at the project root; users choose compatible gens and move their action dirs beside `DATA/` before running a project.

## 2a. DATA, as the actions resolve it

| File | Used by |
|---|---|
| `reference.pdb` | `anchor-refe`, `sequence: auto` fallback. X2: the exact bytes of Preparic's RNA all-atom file. |
| `pdbcode.txt` | `exclude: auto` |
| `range-K.pdb`, `range-K-aa.pdb` | `protein: range-K` (plain for anchoring, all-atom for scoring; [resolve.py](../alaric/middle/resolve.py) `resolve_protein`) |
| `energy-thresholds.json` | `energy-filter` `threshold: auto` |
| `score-filter.yaml` | `energy-filter` `cascade`/`min_cascade: auto` |
| `constraints.json` | `grow` `crmsd`/`ovrmsd: auto` (`pairs`), `sequence: auto` |

## 3. Rules

**R0. Pool names.**
- A gen's final pool is named after its fragment only: `frag5`. The gen's role (anchor, fwd, bwd, merge) appears in its folder path and its intermediate pool names, never in the final name.
  - A gen that consumes fragment N−1 references `fragN-1`, whichever gen produced it.
  - The main folder can hold only one action dir per name, so the filesystem allows exactly one source per fragment in a run. Two gens with the same final pool cannot both be moved in.
  - In unfixed mode the range suffix applies where R9 requires it: `frag5-range2`.
- **Exception: merge gens** name their final pool `fragN-merged` (R5). The user chooses gens dynamically, so a merge must be able to sit in the main folder next to the gens it joins. Its restricting pool is another gen's `fragN`, and a merge that ends on an anchored fragment (e.g. `merge/fwd/anchor/gen-frag6-merged`) sits next to the anchor gen's `fragN`.
- **Exception: common-conformer gens** name their final pools `fragN-rangeX-common-conf` and `fragN-rangeY-common-conf` (R8), so they sit next to the plain pools they read.
- **Intermediate pools** are named by the pool's own fragment, then the gen role, then the range suffix if any, then a step suffix:

  | Suffix | Action |
  |---|---|
  | `-raw` | `anchor`/`anchor-refe` |
  | `-from-N` | `anchor`/`anchor-refe` of an inner fragment from nucleotide N (R3) |
  | `-1`, `-2`, … (after `-raw`/`-from-N`) | one stacking's raw anchor, when a nucleotide has several (R3) |
  | `-ident` | `identity` of raw anchors |
  | `-grown` | `grow`, before scoring |
  | `-unique` | single-pool `identity` (R11) |
  | `-pass` | non-final pool of the first pruning pass in a K-run; independent of the variant's direction |
  | `-ident-1`, `-ident-2`, … | additional binary `identity` folds when more than two raw pools must be intersected |
  | `-e1`, `-e2`, … | non-final steps of a tandem `energy-filter` chain |
  | `-scored` | last `energy-filter` of a non-final pool |
  | `-mask` | `mask-common-conformer` (R8) |

- Every variant of a gen uses the same intermediate names. Select exactly one direction/anchor-tree variant for an anchor run in an assembled run; those alternatives deliberately collide and cannot coexist.
- Examples:
  - `anchor/fwd/gen-frag6-7`: `frag6-anchor-raw`, `frag7-anchor-raw`, `frag6-anchor-scored`, `frag7-anchor-scored`, `frag7-anchor-grown` → `frag7`, `frag6-anchor-grown` → `frag6` (R12, one stacking per nucleotide).
  - Grow gen: `frag3-fwd-grown` → `frag3-fwd-unique` → `frag3`.
  - Inner fragment of a K-run: `frag5-anchor-from-nuc5`, `frag5-anchor-from-nuc6` → `frag5-anchor-ident` → `frag5-anchor-scored`.

**R1. What the exporter writes.**
- `DATA/` and gens. Each gen is a subfolder, nested in a classification path (R7), and holds the gen's actions. There are many more gens than a run uses. The user builds a run à la carte by moving a gen's action dirs into the main folder.
- Every action that accepts `fragment` states it explicitly, and each `grow` states `direction` explicitly. `fragment: auto` takes the first `frag<N>` token of the dir name and `direction: auto` needs a name ending in `-fwd`/`-bwd`, which R0's names do not provide. `identity`, mask production and mask application use their schema's input keys, without unsupported fragment/direction fields. `sequence`, `exclude`, `threshold`, `cascade`, `min_cascade` and the grow `crmsd`/`ovrmsd` stay `auto` where applicable and resolve from `DATA/`, so the edited values live in one place. `protein` is an explicit `range-K`; a portable gen cannot infer the intended range from its name.
- The export is a ZIP with one top-level folder, `alaric_project/`, which holds `DATA/` and the gen classification folders. Extraction creates the main folder. The earlier fresh-directory/non-empty-target rule is replaced by this archive contract: the download takes no target path and writes no project files. It includes `DATA/export-manifest.json`, which records:
  - `schema: 1`;
  - the snapshot digest;
  - the reference SHA-256;
  - the ranges;
  - the fixed/unfixed gens present;
  - Preparic's P5.8 hand-off and class table (X4: these travel with the export).

**Why gens are chosen dynamically.** Whether a stretch is covered by the fwd chain, the bwd chain or a mix (and where the merge falls) does not change the outcome: after the merge and dead-end removal, the final pruned pools are the same. What differs, and greatly, is the size of the intermediate pose pools along the way. The user therefore picks gens step by step, guided by the pool sizes seen so far. This is why every choice must be exported, why gens must slot into a run already in progress, and why a gen references its inputs by fragment name alone (R0).

**R2. Types.** Gens come in types: anchor and grow, plus merge (R5) and common-conformer (R8). In its simplest form a gen is its namesake action plus an `energy-filter`. ("Mode" is reserved for fixed/unfixed, R6.)

**R3. Anchor gens.**
- They live under `anchor/` (Gaussian `anchor` action) and `anchor-refe/` (`anchor-refe` action). Both trees provide the same final pool names and run variants, so each anchor run uses one or the other. Different runs may use different trees. Reference actions are deduplicated by candidate within a gen (X3): unlike Gaussian anchors, they have no protein-residue input. Several stackings of the same nucleotide can therefore require additional Gaussian raw actions/identity folds without requiring duplicate reference actions; the two trees need not have identical internal graphs.
- An anchored nucleotide k lies in frag k−1 (as its second nucleotide) and frag k (as its first).
- **Source of anchors.** Anchors come from Preparic's candidate anchors (contract P5.3, `stage5.candidate_anchors`), one per (RNA residue, range):
  - `locks` give the unclipped fragments that contain the nucleotide, as matrix row keys with nucleotide 1/2. They already use ordinal fragment numbering (contract §1), so no residue-number arithmetic is needed.
  - `range` gives `protein: range-K`.
  - `stackings` lists every selected stacking collapsed into that anchor, each with its own protein residue.
- **Several stackings of one nucleotide.** When a nucleotide stacks with several protein residues, each stacking gets its own raw Gaussian anchor action, and all of them are folded by `identity` into the single pool that one stacking would have produced: `fragN-anchor-raw`, or `fragM-anchor-from-nucM` for an inner fragment. The reference tree emits the corresponding candidate only once. The recipes below are unchanged downstream of that pool.
  - Fixed mode: this covers all stackings of the nucleotide, whatever their range (R6).
  - Unfixed mode: only stackings on the same range are folded. Stackings on different ranges belong to different range-rooted gens (R6, R9).
  - Names of the individual raw anchors: `-1`, `-2`, … appended, ordered by (range, protein residue), e.g. `frag5-anchor-raw-1`, `frag5-anchor-raw-2` → `frag5-anchor-raw`.
- **Single fragment** (the anchor is at a chain end, e.g. nuc1): `anchor/gen-frag1/` contains `frag1-anchor-raw` (the anchor action) and `frag1` (`energy-filter`, input `frag1-anchor-raw`).
- **Single interior anchor** (e.g. nuc7, giving frag6 and frag7): variants `anchor/fwd/gen-frag6-7/` and `anchor/bwd/gen-frag6-7/`. Both first run anchor + `energy-filter` for frag6 and frag7 (intermediates).
  - fwd: `frag7` = grow the scored frag6 pool forward, restricted by the scored frag7 pool; then `frag6` = grow `frag7` backward, restricted by the scored frag6 pool.
  - With one stacking per anchor pool, "restricted by" here is an unrestricted `grow` followed by an `identity` with the scored anchor pool (R12): `frag7-anchor-grown` + `frag7-anchor-scored` → `frag7`. When nuc7 has several stackings, it is a `grow` with `restrict_input`.
  - bwd: the same with the order reversed (frag6 first).
- **K consecutive anchors on the same range** (nucleotides N … N+K−1). The folder is named by the first and last fragment, e.g. `anchor/fwd/gen-frag3-6`.
  1. Each of the K−1 inner fragments, whether there is one (K=2) or several, is anchored from both its nucleotides: inner fragment M gets `fragM-anchor-from-nucM` and `fragM-anchor-from-nuc(M+1)`. The two pools are merged by `identity`, then energy-filtered.
  2. If K>2, the inner fragments are pruned against each other by a forward pass and then a backward pass, as in `gen-frag6-7`:
     - Forward pass: grow the first inner fragment's scored pool forward, restricted by the next inner fragment's scored pool, and repeat up to the last inner fragment. The last inner fragment's forward-pass pool is its final pool.
     - Backward pass: grow the last inner fragment's final pool backward, restricted by the previous inner fragment's forward-pass pool (for the first inner fragment, its scored pool), and repeat down to the first inner fragment. Each backward-pass pool is that fragment's final pool.
     - Without the backward pass, only the last inner fragment would be pruned against the whole run. Inner pools are often very small after the forward pass, but not always, so the backward pass is always done.
     - Both passes use `restrict_input`, not R12's grow + `identity`: inner-fragment pools combine several stackings and are much smaller than single-stacking pools, so restricting by them is cheap.
     - Names (fwd variant, inner frag4–frag6): `frag5-anchor-pass`, `frag6` (forward pass); `frag5`, `frag4` (backward pass). The bwd variant uses the same `-pass` convention for its non-final first-pass pools, satisfying R0's variant-independent naming rule.
     - With K=2 there is one inner fragment and no passes: its energy-filtered pool is its final pool.
  3. Each outer fragment is grown from the adjacent inner fragment's final pool: frag N−1 backward from the first inner fragment, frag N+K−1 forward from the last. Each grow is combined with a raw anchor action for that fragment by an `identity` merge, and the merged pool is then energy-filtered. The raw anchor pool gets no `energy-filter` of its own. No separate unique step is needed, since the anchor action yields unique poses by construction.
  - Variants `fwd/` and `bwd/` as for K=1. bwd is the mirror of the above: the first pass runs backward from the last inner fragment, the second forward.
- **Clipping.** Clipped fragments get no actions. A run is recomputed on the kept fragments, so a K=1 run that loses one outer fragment degenerates to the single-fragment recipe (1B7F: nuc11's frag11 is clipped, giving `anchor/gen-frag10/`). A longer run retains its inner fragments and whichever outer fragments survive.

**R4. Grow chains.**
- fwd: `grow/fwd/gen-frag3/` produces `frag3` from input `frag2`; `grow/fwd/gen-frag4/` produces `frag4` from `frag3`. bwd: `grow/bwd/gen-frag4/` produces `frag4` from `frag5`.
- The user moves e.g. `grow/fwd/gen-frag3/*` into the main folder.
- Chains stop at the clip boundary.

**R5. Merge gens.**
- Example: `merge/fwd/gen-frag4-merged` grows `frag3` (fwd side) forward, restricted by `frag4` (bwd side). Its final pool is `frag4-merged` (R0 exception). `merge/bwd/gen-frag3-merged` is its mirror image.
- With anchors at nuc1 and nuc7 on the same range, the merges run:
  - fwd: from `merge/fwd/gen-frag2-merged` (grow `frag1`, restricted by `frag2`) to `merge/fwd/anchor/gen-frag6-merged` or `merge/fwd/anchor-refe/gen-frag6-merged` (grow `frag5`, restricted by `frag6`, the anchor pool; R5 content, R7). The latter is the "all fwd, no bwd" choice.
  - bwd: the mirror image.
- A chain does not carry on past the next anchor run on its own range (fixed: on any range). The anchor pool is the latest restriction target.
- Abutting runs (e.g. nuc4 and nuc6): `merge/fwd/gen-frag5-merged` grows `frag4`, restricted by `frag5` (the nuc6 anchor pool). `merge/bwd/gen-frag4-merged` is the mirror.
- Merges are allowed only when both anchors are on the same range, or in fixed mode.
- Content:
  - Restricted by a grown pool: a `grow` named `frag4-merged-grown`, with `input: frag3` and `restrict_input: frag4`, followed by an `identity` of `frag4-merged-grown` with `frag4` (its `restrict_input`), named `frag4-merged`. There is no `energy-filter`. `grow` keeps only output poses present in the restricting pool ([grow.py](../alaric/grow.py) `--restrict-poses`), and that pool is already scored. The `identity` is always required: the grown pool must be identity-merged with its restricting pool before chains can be built on `frag4-merged`.
  - Restricted by a single-stacking Gaussian anchor pool (`merge/.../anchor/`): an unrestricted `grow` named `frag6-merged-grown`, then the `identity` with `frag6` (R12). The pool is too large to serve as `restrict_input`.
  - Restricted by the same pool made by `anchor-refe` (`merge/.../anchor-refe/`): the reference pool is small, so this merge is like the grown-pool case, with `restrict_input` on the `grow`, followed by the `identity`.
  - An anchor pool of several stackings (e.g. a K-run's outer fragment) keeps `restrict_input` on the `grow` in both trees (still followed by the `identity`), so there is no `anchor/` / `anchor-refe/` split.
  - In unfixed mode, with a range suffix (R9): `frag8-merged-range2`.

**R6. Fixed versus unfixed.**
- **Fixed:** all range PDBs share one trusted frame.
  - A fragment interacting with several ranges is scored by a tandem `energy-filter` chain, one per range, starting with the range that has the lowest (most negative, i.e. strictest) energy threshold. Ties go to the lower range index.
  - The range does not matter for anchoring. Consecutive anchored nucleotides form one K-run whatever their ranges, and all stackings must hold: an anchored fragment's raw anchor pools are all combined by `identity`, including several stackings of one nucleotide on different ranges (R3, several stackings).
  - A chain may grow into any fragment that interacts with at least one range, so the two sides always meet and a merge is always possible, barring clipping.
- **Unfixed:** for anchor and grow, each range PDB is an independent, unrelated molecule.
  - A pose is meaningful only in the frame of its chain's root range R. Its `energy-filter` scores against R only, and the fragment's interactions with other ranges play no part in that chain.
  - An R-rooted chain grows into each fragment that interacts with R. It stops at the first that does not, and only the matrix stops it, not another range's anchors.
  - Consequently, two R-anchors separated by a fragment that does not interact with R give two halves that never meet, and no merge gens between them. This is expected in unfixed mode and is not reported.
  - K-runs form only within one range.
- **Illegal:** a fragment that interacts with no range at all.

**R7. Classification folders.**
- A gen that comes out the same in both modes stays at `X/Y/Z`.
- A gen unique to one mode goes to `X/Y/fixed/Z` or `X/Y/unfixed/Z`.
- In unfixed mode, where gens for different ranges would collide, the path becomes `X/Y/unfixed/range-2/Z`.
- With a single protein range, fixed and unfixed coincide, so no `fixed/`/`unfixed/` levels are written.
- Gen folder naming:
  - X is the gen type's tree (`anchor`, `anchor-refe`, `grow`, `merge`) and Y the direction (`fwd`/`bwd`). Z is the gen folder, `gen-fragN`, or `gen-fragN-M` for an anchor run (a hyphen, since `frag67` would be ambiguous once fragment numbers reach 10). A merge gen's folder is named after the final pool it generates: `gen-fragN-merged`.
  - Single-fragment anchors have no direction level: `anchor/gen-frag1`.
  - Merges have a direction level like the other trees: `merge/fwd/gen-frag4-merged`.
  - The gen folder is always the last component, e.g. `grow/fwd/unfixed/range-2/gen-frag8`. The user moves `…/gen-*/*` into the main folder.
  - Common-conformer gens exist only in unfixed mode and have no direction or mode level: `common-conformer/gen-fragN-range-X-Y/` (R8).
  - A merge that ends on a single-stacking anchor pool (R12) is split by anchor tree: its graph differs between the Gaussian `anchor` tree (pool too large for `restrict_input`) and the `anchor-refe` tree. It has an `anchor/` or `anchor-refe/` source-tree level immediately before its `gen-*` folder, after any mode/range levels. For example: `merge/fwd/fixed/anchor/gen-frag6-merged` and `merge/fwd/fixed/anchor-refe/gen-frag6-merged`. Select the merge matching the chosen anchor tree, and declare only that tree's producers for the anchor-pool external input; leave other merge-input producer choices unchanged. Merges that end on a grown pool or on a multi-stacking anchor pool have identical graphs in both trees and keep the ordinary generic merge path.

**R8. Common-conformer gens (unfixed, different ranges).**
- For every fragment N that interacts with, and is reachable by, ranges X and Y (X < Y), i.e. every fragment whose pools carry a range suffix (R9), the exporter writes a common-conformer gen `common-conformer/gen-fragN-range-X-Y/`.
- Inputs: always the plain pools `fragN-rangeX` and `fragN-rangeY`, whichever gens produced them.
- Actions:
  - `mask-common-conformer` with `input1: fragN-rangeX` and `input2: fragN-rangeY`, giving the intermediate `fragN-rangeX-rangeY-mask`. It produces one mask per input.
  - Two apply-mask actions (C1), one per input.
- Outputs (the gen's two final pools): `fragN-rangeX-common-conf` and `fragN-rangeY-common-conf`.
- More than two interacting ranges (extremely rare): one gen for every pair X < Y among them. The outputs are then `fragN-rangeX-rangeY-common-conf` and `fragN-rangeY-rangeX-common-conf`.
- This is not a merge. No chain spans the two anchors.
- The outputs have names of their own, so the gen never replaces a plain pool. The user moves it in once both plain pools exist in the run.
- Whether the filtered pools are end products or are grown from is up to the user. Exported gens always read plain pools. To grow from a filtered pool, the user edits the input of the downstream gen by hand. For example, in 1B7F, `grow/fwd/unfixed/range-3/gen-frag6` can read `frag5-range3-common-conf` instead of `frag5-range3`.

**R9. Range suffix.** In unfixed mode the range suffix is a differentiator. Each range is a separate molecule, and plain `fragY` suffices whenever it is unambiguous which range is attached. `fragY-rangeX` is used only for a fragment that interacts with more than one range and is reachable by chains (or anchors) of more than one range. This holds whether the pool is anchored or grown.
- The suffix is decided per fragment, so every gen uses the same name for a given (fragment, range), whichever gen produced it (R0). A chain therefore gains or drops the suffix as it crosses fragments: in 1B7F, the range-2 fwd chain runs `frag8-range2` → `frag9` → `frag10`, and the bwd chain from nuc11 runs `frag10` → `frag9` → `frag8-range2`.
- Example (1B7F): the range-3 run's outer frag5 and the range-2 nuc6 anchor both produce frag5. R7's range folders separate them in the export tree, but both belong in the same run (R8 filters each against the other), so their names must differ in the main folder: `frag5-range3` and `frag5-range2`.
- The pool-name suffix has no hyphen before the number: final pools `frag5-range2`, `frag6-range3`; intermediates `frag5-fwd-range2-grown` etc.
- Folders keep Preparic's label: `unfixed/range-2/`. `DATA/range-2.pdb` and `protein: range-2` are unchanged.

**R10. Sentinel.** Gens downstream of an `anchor-refe` with `ovrmsd: -1` are still generated (they fail at run time by design; X3).

**R11. Unique final pools** (optimisation). Every gen's final pool holds unique poses.
- Anchor gens and merge gens already produce unique pools: the anchor action yields unique poses by construction, and so does an `identity` with an anchor pool. Common-conformer outputs are subsets of unique pools, so they are unique too.
- Grow gens need an explicit unique step, placed directly after the `grow` and before `energy-filter`. The filter keeps a subset, so its output stays unique, and scoring runs on fewer poses.
- The `identity` action does this today only if both inputs point to the same dir. It is to be modified to work on a single pool (C2).

**R12. Restriction by a single-stacking anchor pool is implemented as grow + `identity`** (optimisation).
- Wherever R3/R5 say "restricted by" an anchor pool that comes from a single stacking, the gen runs an unrestricted `grow` followed by an `identity` merge with the scored anchor pool. That gives the same poses (and unique ones, R11) at a much lower cost than `restrict_input`, because single-stacking pools are large.
- An anchor pool that combines several stackings, or any `anchor-refe` pool, is much smaller, and restricting by it is cheap, so it keeps `restrict_input`. This holds whether the stackings are on consecutive nucleotides (a K-run, inner and outer fragments alike) or are several stackings of one nucleotide (R3). The stackings are always on one range, or in fixed mode.
- Single-stacking anchor pools occur in:
  - the `gen-frag6-7` steps (both variants), when nuc7 has one stacking;
  - the Gaussian-tree merges that end on such a pool (`merge/fwd/anchor/gen-frag6-merged` into a one-stacking nuc7, the abutting `merge/fwd/anchor/gen-frag5-merged` into a one-stacking nuc6, and their mirrors). The `anchor-refe` merges to the same target keep `restrict_input`: a reference pool is small.
- Merges restricted by a grown pool (e.g. `merge/fwd/gen-frag4-merged` restricted by the bwd side's `frag4`) also keep `restrict_input` on the `grow` (R5), followed by the `identity` with the restricting pool.
- No `energy-filter` after the `identity`: the result is a subset of the scored anchor pool.
- Names: `frag7-anchor-grown` → `frag7` (identity with `frag7-anchor-scored`); `frag6-merged-grown` → `frag6-merged` (identity with `frag6`). With `restrict_input`, a non-merge gen's step is a single `grow` named after its output (`frag7`). A merge always has the `-merged-grown` → `-merged` pair: its `grow` keeps `restrict_input` where R12 does not apply, and the `identity` with the restricting pool is always present.

## 4. Ownership, inputs and snapshot admission

### 4.1 Implementation boundary

Put export configuration generation in **Preparic**, as two transformers in its graph (§7.1). Create three root-level Python modules beside `backend.py`: `export_snapshot.py` (normalisation, admission, digest), `export_gens.py` (the generator planner) and `export_tree.py` (rendering `DATA/`, manifest and action files to a folder mapping). They use the standard library and Preparic's own validators/helpers; YAML emission may use its existing YAML dependency. All their functions are pure: they take the transformer's input values and never touch a Context, so fixtures do not fabricate Seamless objects. Include their source as transformer modules, using the same pattern as `preparic_stage4` in `backend.py`; do not load helpers from the installation at execution time.

Alaric owns action schemas, resolution, execution and cache identities (§6). Preparic emits configuration for those actions; it does not execute them during export. No exporter imports `alaric`, dynamically loads installed Alaric source into the backend, calls a library/scoring tool, recomputes Gaussian values, or reimplements threshold rounding. This preserves G2 and the boundary document's ownership rule. Constants mirrored in Preparic remain covered by parity tests against Alaric's definitions.

E7 retired `alaric/export_bundle.py` after transferring its useful acceptance assertions to tests of real input pins, rendered archive members and selected generator trees. Its synthetic `selected_anchors`, `reference_candidates`, `handoff.fragment_rows`, browser-consent fields and `ready` result are not the backend input API. No compatibility wrapper or second export path remains.

### 4.2 Input inventory

Both export transformers take the following cells as input pins. They never reread deployment mounts or the browser's state graph. Non-shared cells are wired like any other; no folder needs to become shared.

| Captured values | Source and purpose |
|---|---|
| Structure and stage decisions | S1 `pdbcode`, S2 `biounit`, S3 `cif`, S4 `ranges`, S8 `stacking_selection`, S11 `threshold_edits`, S12 `best_fit_clip`, S14 `interaction_matrix`, S18 `gaussian_classes`, S24 `energy_threshold_edits`; validate bases, active flags, stage and the structure/code agreement |
| Normalised structure/range identity | S5 `range_specs`, S21 `structure_id`, B5 `exclude_pdb_code`; retain original numbering and assembly identity |
| Exact prepared files | B1 `range_pdb_files`, B3 `range_pdb_files_aa`, B4 `rna_pdb`; folder pins deliver path-to-bytes mappings inside the transformer, with B4 equal to B3's `range-1-aa.pdb` |
| Fragment and pair evidence | S22 `fragments`, S9 `best_fit_thresholds`, S10 `best_fits`, B11 `clipped_fragments`; full fragment identities, kept PDBs/edited pairs/clip, computed raw and rounded pair/base values |
| Anchor and interaction evidence | S7 `stackings`, S8 selection, S14 matrix, B14 `interactions`; derive candidate anchors and verify all current locks against the actual matrix |
| Energetics | S13 `energetics`, S24 edits; true-cell energies and edited values. S15 computed energy thresholds are a Copy source, not exported thresholds |
| Gaussian evidence | S17 `bestfit_stackings`, S25 `gaussian_scores`, S19 `gaussian_validation`, B18 `gaussian_check`, B15 `gaussian_model_json`, B7 `fraglib_identities`, the six `ring_refe_*` cell identities; class validity, chosen-filter results, model/library identities and minimum cell fraction |
| Validators | T10/T19/T27 result pins, used only as dependency gates. A successful check returns `None`; an exception blocks downstream execution. These values are not validation reports and must not be interpreted as undecided decisions |

S23 `range_pdbs_aa` is also checked for freshness and agreement with the exact B3 bytes the browser displays (X2/P11). S26 `stage4_active_nucleotide`, other view state and the local Proceed anyway checkbox are excluded from export identity. Computed S15 is not a new prerequisite when unused. A per-ligand error on a false matrix cell does not block export; a true-cell error does.

Reuse Preparic's `stage5.py:candidate_anchors` definitions (mirrored in `src/lib/stage5.ts`): records are `{resid, range, stackings, locks:[{fragment, nucleotide}]}`, with fragment keys and nucleotide 1/2 in the locks. Collapse selected stackings by (RNA-residue identity, protein range), retain every selected stacking key/protein residue, and convert kept locks to action fragment ordinals/`first|second` exactly once. Do not infer them from `interactions.runs[].anchors` alone: that hand-off omits the individual stacking/class associations required for Gaussian actions. Compare the independently derived locks with the hand-off and matrix. Use ordinal fragment addresses and the RNA residue identity carried by the fragment list, never PDB-residue arithmetic. Refuse a protein-residue identity the current integer `resid` action API cannot represent; never silently strip an insertion code or select a different residue.

### 4.3 Consistency (X7)

X7 asks for one consistent snapshot. It follows from the graph, not from capture code:

- **Each export cell is consistent with its inputs.** A transformation computes from one set of input checksums, so `export_admission` and `export_tree` each see a single state of their pins. Neither reads anything outside its pins.
- **The download takes one consistent cut.** A GET reads `export_tree` in one read turn of the Context, holds its buffers until the archive is built and never computes. If `export_tree` is not complete (an edit is still propagating, or a producer is waiting, computing or failed), the GET answers `409` with the reason. It never serves a mixed or stale archive as a success.
- **Accepted writes invalidate the export before a later cut.** For a changed PUT, Seamless installs the input and derives the graph in a controller turn before returning `200`. A subsequent download cut therefore sees the dependent tree pending until recomputation completes; it does not fall back to the last complete folder. An equal-value PUT need not invalidate anything. E0 tests this ordering through HTTP, not through a simulated capture loop.
- **Freshness echoes are still checked.** Wiring guarantees that the export consumed the current cells, not that those cells agree with one another. Where a decision carries a basis echo, or a result echoes the decisions it was computed for, admission checks the echo against the current decision values (§4.4). A result is not fresh merely because its fragment keys match.
- **Admission and tree agree.** `export_tree` takes `export_admission` as an extra pin. It rebuilds the normalised snapshot with the same pure function and raises unless admission is `ready` and its own digest equals admission's `digest`. The manifest records that digest (§5.2).

An edit after the cut cannot alter the archive being built. Whether the archive is the state the user approved is checked in the browser by comparing digests (§7.2). No project edits are locked or delayed for export. There is no generation counter, observer hook, recapture guard or `snapshot_changed` refusal, and `ctx.compute()` plays no part.

This differs from a **share** of `export_admission`: a share can keep serving its last complete value while its graph node is pending or failed. The download checks the live node state. Use the existing state graph for UI status and the download for authoritative admission; do not add a second capture protocol to compensate for stale share records.

### 4.4 Normalised snapshot and hard blocks (X1/X6)

Define an internal `ExportSnapshot` with `schema: 1`, the export-relevant decision values, full and kept fragment maps, kept edited pairs, candidate anchors/stacking classes, true-cell matrix and edited energies, the interaction hand-off, prepared file bytes, Gaussian/model/library identities, and the Gaussian check's minimum cell fraction. `export_snapshot.py` builds it from the transformer's input values.

Admission checks, in addition to ordinary decision/freshness validation:

- The exclude code is the same non-empty lower-cased code used by the fit and Gaussian check. No generated environment sets `ALARIC_ANCHOR_MIN_CELL_FRACTION`; the Gaussian check's fraction equals the canonical default (⅓), as in X1/A11. A mismatch is a hard block.
- Full reference bytes equal B4, B3's RNA member, and the fitted/displayed input's identity. Record SHA-256 of the bytes, never of a re-serialised PDB.
- Every required boundary JSON has `schema: 1`. Validate fragment keys, two-letter sequences, adjacent pair addresses, trinucleotide identities, positive finite pair values with at most two decimals, and the exact current clip. Reject booleans as numeric values. Kept numbers stay their original ordinals after clipping.
- Every kept pair has edited cRMSD/ovRMSD values. Every kept pair ovRMSD is at most L_ov; no checkbox overrides this. Clipped pairs need no filled values and contribute no action or warning. No pair/base threshold is rounded again.
- Every selected stacking has at least one kept candidate; every kept anchor lock is true for its range; hand-off runs, ranges, clip and interactions agree with the actual matrix. Enforce the stage 5 run/shape/error rules again, including anchors whose true cell has a scoring error.
- Each kept fragment interacts with at least one range (R6). Every true cell has a finite best-fit energy and a valid edited two-decimal threshold. Match S24 entries by their range row, not by a historical range index. Missing false/clipped entries are allowed.
- Class keys equal selected stacking keys, filter names come from the current model, and Gaussian evidence is fresh for the current classes, kept candidates, model, rings and library. Red/amber filter results are warnings under X5, not invalid decisions. Installation/job errors remain hard blocks.

When admission can run, report malformed or conflicting evidence as a block with the fragment/range/stacking address and remedy; never silently repair it, suppress an anchor, fill a threshold or regenerate installed data. A pending or failed upstream input can prevent admission from running at all: then the state graph, not an old admission value, supplies the pending/error explanation (§7.1). Because `export_tree` raises unless admission is ready, the checks bind a direct GET as well as the button.

Whenever its inputs are available, admission also re-derives the stage-6 gate from its decision inputs (the same predicates as `project-state`), so a project below stage 6 is reported as not ready, not as invalid. Do not promise an admission result before its producers can execute.

### 4.5 Function contracts for the implementer

Use ordinary Python mappings/lists, primitive values and bytes for these interfaces. `ExportSnapshot` and the recipe records below are internal mappings; the only new boundary outputs are admission's schema-1 JSON and the folder mapping. Keep the two thin transformer wrappers in `backend.py`; put testable logic in the three modules.

| Module/function | Input | Return and responsibility |
|---|---|---|
| `export_snapshot.normalise_export_inputs(inputs)` | Named pin values from §4.2, keyed by their pin names | `(snapshot_or_none, blocks)`. Parse/normalise and perform §4.4 checks. A block is `{code, address, message, remedy}`; `address` is a stable fragment/range/stacking key, or null. No action emission or Context reads |
| `export_snapshot.snapshot_digest(snapshot)` | A normalised snapshot | Canonical SHA-256 string under §5.2. The snapshot includes every value needed for emission and X5 decisions, but no validator-return sentinel or view state |
| `export_snapshot.export_warnings(snapshot)` | A valid, normalised snapshot | Ordered X5 records `{kind, address, message}`. Use `kind` values `gaussian`, `pair-boundary`, `energy`. Never round values or make base/sentinel warnings |
| `export_snapshot.build_admission(inputs)` | The same pin-value mapping | Schema-1 admission (§7.1). `ready` means no hard blocks; warnings do **not** make it false. `digest` is null if no snapshot can be constructed. This function owns normalisation, checks, warnings and digest for admission |
| `export_gens.plan_gens(snapshot)` | A valid snapshot | Ordered generator recipes under §5.3, after graph/name/path checks. No filesystem writes or YAML serialization |
| `export_tree.render_tree(snapshot, recipes, digest)` | Valid snapshot, checked recipes and matching digest | `dict[str, bytes]` containing exactly §5.1's members, without the outer `alaric_project/` prefix |
| `export_tree.build_tree(inputs, admission)` | The same pin-value mapping plus schema-1 admission | Re-run normalisation/digest; require no blocks, `admission.ready == True` and equal digests; call planner and renderer. Raise `ExportNotReady` for a known refusal; propagate unexpected errors. Never return null or an empty success tree |

`build_admission` computes the digest whenever a snapshot can be constructed. It computes warnings only when that snapshot has no hard blocks; otherwise `warnings` is `[]`. Return `ready = not blocks` only when a snapshot exists. A failed normalisation must never accidentally produce `ready:true` from an empty block list; add a block explaining the missing snapshot.

Define `ExportNotReady` in `export_snapshot.py` and include that module in both wrappers. Do not catch `Exception` and turn a programming error into a not-ready result. Catch expected input-validation errors at the validator/parser that knows their address and convert those to blocks. An installation failure that already failed an upstream transformer remains an upstream failure; a materialised job record containing an error is an addressed hard block.

A generator recipe is `{path, modes, range, direction, final_pools, external_inputs, actions}` with the meanings in §5.2. Each action record is `{name, parameters, comments}`: `parameters` is the complete action mapping including `action`, and `comments` is a list of literal YAML comment lines. `actions` is in dependency order. `external_inputs` contains the exact canonical pool names plus expected fragment/sequence/frame; it is not a second producer-selection mechanism. The manifest contains the compact inventory, not a second copy of every action's YAML parameters.

The wrappers have explicit parameters for the input pins; do not use an opaque Context pin or construct transformers inside transformers. They only gather those values, import their attached helper modules and call `build_admission` or `build_tree`. Both wrappers must pass the same named values to `normalise_export_inputs`; validator dependency pins are deliberately ignored as data.

Existing helper signatures to reuse, with source included as modules rather than imported from an installed checkout:

- `stage4._validate_threshold_edits(edits, fragments)` and `stage5.validate_energy_threshold_edits(edits, fragments, range_specs)` validate decision shapes; both accept their documented undecided forms, return `None` on success and raise on malformed decisions.
- `stage6.validate_gaussian_classes(marker, model_json, selection, gaussian_filter)` checks the editable class marker. Supply its existing attached filter helper as in the current backend; do not create a new filter-name table or execute an installed Alaric import in the backend process.
- `stage5.candidate_anchors(fragments, selection, stackings, matrix_rows)` supplies the anchor/lock shape of §4.2. Reuse it; do not reconstruct anchors from the first stacking or only the interaction hand-off.

The extra export checks (coverage, retained locks, limits, file agreement, X5 reasons and digest) belong in `export_snapshot.py`. Translate only the decision-only stage predicate needed from `src/lib/project-state.ts`, and test it against the existing stage fixtures; do not introduce another persisted stage or a new navigation state machine.

## 5. Archive and generator implementation

### 5.1 Archive members

`export_tree` is a `folder` cell whose keys are the following paths; the download serves them under `alaric_project/`. Nothing is written to a mounted folder and there is no persistent export job.

| Member | Content/source |
|---|---|
| `DATA/reference.pdb` | Full, exact B4 RNA bytes; clipping never rewrites it |
| `DATA/pdbcode.txt` | Validated lower-cased exclude code, followed by one newline |
| `DATA/range-K.pdb`, `DATA/range-K-aa.pdb` | Exact B1/B3 bytes for each range, including range 1; preserve assembly frame, atom order and residue numbering |
| `DATA/constraints.json` | Versioned sequence map and kept edited grow-pair values, as specified below |
| `DATA/energy-thresholds.json` | Exactly `{schema:1, "frag-N-SEQ":{"range-K":"-16.59"}, …}`; true kept cells only, no provenance or wrapper fields |
| `DATA/score-filter.yaml` | Exactly A12's default min_cascade/cascade data, below |
| `DATA/clipped-fragments/reference/frag-N-SEQ-refe.pdb` | Exact kept reference PDB bytes from B11's `reference/` members |
| `DATA/clipped-fragments/bestfit/frag-N-SEQ-bestfit.pdb` | Exact kept best-fit PDB bytes from B11's `bestfit/` members |
| `DATA/best-fit-statistics.json` | Schema 1. Per kept fragment: best-fit RMSD, computed raw/rounded base ovRMSD per nucleotide, and the best fit's ATTRACT energy (or scoring error) against every protein range. Per kept pair: raw, computed (rounded) and edited cRMSD/ovRMSD. Informational; no action reads it |
| `DATA/anchor-refe-bases.pdb` | One residue per RNA position used by an `anchor-refe` action in the catalogue: the nucleobase atom records that anchor-refe fits (Alaric's `only_base` template atoms: ring atoms plus in-plane exocyclic substituents), copied verbatim from `reference.pdb` in template order, each preceded by a REMARK naming its candidates and ovRMSD. Informational; no action reads it |
| `DATA/export-manifest.json` | Digest and generator inventory (§5.2), including the hand-off and class table |
| `<classification>/gen-*/<action-name>/alaric.yaml` | One schema-valid action mapping, plus real YAML comments for sentinel actions |

The generator catalogue is configuration, not a runnable choice of every action. No action directories are emitted directly under `alaric_project/`. `middle/project.py:action_dirs` must continue to ignore the nested catalogue. After extracting, the user moves action directories from chosen gens alongside `DATA/`, as explained in the Alaric usage docs (§8), not in a bundled README; the exporter neither chooses a merge location nor performs run-final dead-end removal.

The two kept-PDB sets are required by X4. They are payload, not provenance, and must not disappear when reducing the manifest. Include exactly one reference and one best-fit PDB for each kept fragment, and no clipped PDB. B11's `thresholds.json` and `clip.json` remain inputs for validation; do not duplicate them in the archive because their edited pairs and clipping information already travel through `constraints.json` and the manifest's hand-off.

`constraints.json` uses the format actually read by `middle/resolve.py`: top-level `fragN: {sequence: "AG"}` entries for **kept** fragment numbers, plus `pairs: [{down:N, up:N+1, sequence:"AGU", crmsd:<edited>, ovrmsd:<edited>}]`, and `schema: 1`. Addresses are original ordinals; `pairs` contains exactly the kept adjacent pairs. Values are the edited values, serialised as decimal strings without changing their numeric value. `sequence:auto` reads these explicit fragment entries; the full `reference.pdb` remains its consistent fallback. No `sequence.txt` with shifted, clipped positions is written. Add strict schema/shape validation to the Alaric consumer (§6 C3), which currently accepts an unversioned, permissive mapping.

`score-filter.yaml` is fixed configuration, not a Preparic edit cell:

```yaml
min_cascade: 1000
cascade:
  - {plateau_distance: 5, neighbour_distance: 7, max_error: 1.0}
  - {plateau_distance: 10, neighbour_distance: 12, max_error: 0.05}
```

All new boundary JSON files have top-level `schema: 1`; a nested marker is preserved as a value, not silently migrated to a new decision shape. No fragment library, Gaussian CSR caches, reduced beads, energy grids, conformer pools, absolute installation paths or generated backend deployment is bundled.

### 5.2 Manifest, digest and tree paths

The manifest holds only what the browser check and the user's later work need:

- `schema: 1`, `snapshot_sha256`, `reference_sha256`;
- the normalised `range_specs`;
- the validated P5.8 `handoff` and the `class_table` (X4);
- `gens`: ordered records `{path, modes, range, direction, final_pools, external_inputs, actions}`. Use null where range/direction is inapplicable; `modes` is `["fixed"]`, `["unfixed"]` or both. `actions` is the ordered list of action-directory names, not the YAML mappings. Describe each external input's fragment/range and permitted producer choices, rather than claiming the whole catalogue is one DAG;
- `disabled_reference_anchors`: candidate address, raw/rounded values, limit, and action paths for every sentinel.

Input checksums, producer provenance and per-member hashes are not repeated: Seamless records the transformations, and the archive bytes come from cell buffers.

All manifest paths are relative to the project folder, exactly like `export_tree` keys; they do not repeat the `alaric_project/` prefix. The download adds that prefix once. Define `EXPORTER_VERSION` in `export_snapshot.py` (2 since the best-fit statistics and anchor-refe base members, 2026-11-13); bump it when emission semantics change and keep a digest golden tied to that version. The manifest's `schema` describes its format; `EXPORTER_VERSION` participates only in snapshot identity.

Compute `snapshot_sha256` in `export_snapshot.py` from canonical JSON of the normalised snapshot: sorted mapping keys, fixed separators, UTF-8, `allow_nan=False`, byte buffers represented by their SHA-256 and length, ordered lists in defined range/fragment/stacking order, plus an `EXPORTER_VERSION` constant that changes with the emitted configuration. Exclude timestamps, local install paths and view state. `export_admission` publishes it as `digest`; `export_tree` writes it into the manifest after checking that it recomputes the same value (§4.3). Never use a Seamless checksum as if it were the raw file's SHA-256.

Build the `export_tree` mapping from validated identifiers only, never from untrusted chain/stacking text. Keys are relative paths with `/`, no absolute paths, `..`, backslashes or empty segments; reject duplicates and a key that is a directory prefix of another. Seamless refuses an unsafe key at GET time (`409 unsafe-folder-key`), but the generator rejects it first. Values are bytes. Prepared PDBs are passed through unchanged, so their leaves keep the input buffers (X2). Packing, member order, timestamps and permissions are Seamless's deterministic packing; Preparic does no ZIP writing.

### 5.3 Build logical recipes before assigning paths

Implement pure helpers in `export_gens.py`: normalise candidate anchors, compute mode reachability, assign canonical pool names, build anchor/grow/merge/common-conformer graphs, classify variants, and validate/render recipes. Internally use records for action name/type/parameters, local dependencies, external pool contracts, final pools and restriction provenance. Do not construct names by concatenation throughout the code: centralise fragment/range/role/step naming and check uniqueness.

Build in this order:

1. Apply the clip to the full fragment/candidate map. Form consecutive nucleotide runs in fixed mode across all ranges, and separately per range in unfixed mode. Determine retained inner and outer fragments from actual locks. A K=1 run with one retained fragment uses the single-fragment recipe; a K≥2 run always retains its inner fragments and may lose either/both outer fragments. Its folder uses the first/last **retained fragment**, with a single-fragment folder when they coincide.
2. Compute per-range reachability along contiguous true matrix cells, starting at that range's anchors, stopping at false/clipped rows. Fixed reachability uses the union of true ranges, with anchor runs from any range as boundaries. Keep interaction components distinct: never bridge an unfixed false-cell gap. This pass is independent of action names.
3. For each fragment, count unfixed ranges that both interact with it and can produce its pool. Apply R9's suffix only when this count exceeds one, consistently to every producer and consumer. Naming decisions must precede any graph emission.
4. Build both anchor trees and fwd/bwd variants using R3, with a shared recipe builder for mirrored passes. Gaussian raw actions are per selected stacking and kept candidate, ordered by numeric range then full protein-residue identity and stacking key. Reference raw actions are deduplicated by (fragment, first/second) within a gen. Resolve class/filter names from stacking keys, never from the collapsed anchor's first stacking.
5. Emit raw-pool intersections as deterministic binary `identity` folds. Use the unsuffixed logical raw/from pool as the last fold's output and numbered `-ident-*` temporaries for earlier folds. Intersect the two nucleotide pools of each inner fragment before scoring. For K>2, run the first restricted pruning pass through all inner fragments, then the reverse pass, retaining `-pass` intermediates until all dependents consume them. Outer-fragment grows read the adjacent **final pruned** inner pool, intersect with the raw outer anchor, then score. Do not separately score that raw outer pool.
6. Enumerate ordinary fwd/bwd grow gens outside anchor-run final pools. Walk to the clip/matrix limit or the next anchor run of the mode/range. Each ordinary grow is `grow` → single-input `identity` → scoring. Keep only directions with a reachable adjacent input; do not emit a consumer whose external pool has no possible producer.
7. Between successive compatible anchor runs in the same reachable component, enumerate a merge at every joining edge, including either run's boundary anchor target. Example: if the left final anchor pool is frag1 and the right boundary pool is frag6, emit fwd merges for targets frag2…frag6 and bwd merges for targets frag1…frag5. Do not emit merges on an open tail, across a false interaction gap, or between unrelated unfixed ranges. Abutting runs give just the two boundary merges. A merge reads plain pool names and produces `-merged`; it never overwrites its restricting pool or automatically redirects downstream inputs.
8. For each fragment reached by at least two unfixed ranges, emit R8's pairwise mask/apply-mask gens. For two reachable ranges use `fragN-rangeX-common-conf`; with three or more reachable ranges use pair-qualified output names for **all** pairs, to avoid collisions. These gens always read plain pools and create separate filtered pools.
9. Compare canonical rendered action graphs, comments, final names and external input contracts across fixed/unfixed modes. External contracts describe the required pool kind/fragment/sequence/frame, not the history of how its poses were produced. Hoist only genuinely identical recipes to their shared R7 path. Otherwise retain fixed/unfixed paths. For boundary merges, compare the R12 graph for each actual anchor source tree; use the conditional `anchor/` or `anchor-refe/` path level from R7 only when the merge graphs differ (target is a single-stacking Gaussian anchor pool), and keep the generic merge path when they match. Use range folders consistently within an unfixed (type, direction) branch when multiple root ranges populate that branch, even where one particular fragment has only one producer; this preserves R8's `grow/fwd/unfixed/range-3/gen-frag6` example. Comparing only final names, a fragment's current scoring ranges, or the number of protein ranges is insufficient. With one protein range all recipes are common and no mode levels are emitted.

Validate each gen as a DAG with distinct local names, declared finals, resolved local references, and an explicit external-input list. Validate each selected assembled run as a DAG separately in tests; the unselected catalogue contains mutually exclusive alternatives and is not itself a deployable graph. Reject classification-path collisions rather than choosing one recipe silently.

### 5.4 Action emission details

Action files are `<action-name>/alaric.yaml`, the current `ACTION_FILE`. Use only schema-supported keys. Raw Gaussian anchors write explicit `fragment`, `protein: range-K`, integer protein `resid`, `nucleotide: first|second`, `precomputed_filter_name: gaussian-rrm` and the stacking row's `precomputed_filter_threshold`; `sequence`/`exclude` are `auto`. Reference anchors write the candidate's explicit fragment/nucleotide/ovRMSD and `reference: reference.pdb`, without a protein field.

Render JSON/YAML with deterministic ordering, UTF-8 and a final newline. Add action comments as actual `# ...` lines before the YAML mapping, prefixing every line; never add a `comment` parameter that the action schema rejects. Keep PDB byte buffers completely outside the text serializers. The renderer checks duplicate member names before constructing the final dictionary, since assigning the same dictionary key twice would otherwise silently overwrite an action.

`score_pool(fragment, input, output, mode, root_range)` emits the reusable energy-filter chain. Fixed mode sorts true ranges by numeric threshold (most negative first), then numeric range index; unfixed mode uses just its root range. A one-range filter uses the requested output name directly; longer chains use `-e1`, `-e2`, … and put the last filter at the requested final or `-scored` name. Every filter has `threshold: auto`, `epsilon: 15`, `cascade: auto`, `min_cascade: auto`, explicit `fragment`/`protein`, and `sequence`/`exclude: auto`. The chosen sorting uses edited thresholds without re-rounding them. Keep intermediate stems distinct for different scoring sites in the same gen.

`restrict_pool` chooses R12's optimisation from the **actual restricting recipe**, not the pool name: a single raw-anchor constraint made by the Gaussian `anchor` action uses grow→identity; a pool combining distinct anchor constraints, a pool made by `anchor-refe`, or a grown pool uses `restrict_input`. Both paths preserve the scored restricting pool's subset, so add no energy filter or ordinary-grow unique step after them. Preserve R3's deliberate `restrict_input` for inner pruning passes.

When the boundary merge's target is a single-stacking Gaussian anchor pool, emit the matching `anchor`/`anchor-refe` merge alternatives using R7's conditional source-tree path level. Restrict the anchor-pool external input's permitted producers to the matching tree; retain all compatible producers for the other merge inputs. A generic merge recipe may be shared across the two anchor trees only when its rendered graph and external contract are identical.

For all grows, `fragment` is the **output** fragment, direction is explicit, and the resolved pair is (N−1,N) for fwd or (N,N+1) for bwd. Do not use a source fragment's thresholds by accident. Keep `sequence`, `exclude`, `crmsd`, `ovrmsd` as `auto`.

For reference thresholds, look up the S9 base row by fragment/sequence and select nuc1 for `first`, nuc2 for `second`. Emit the rounded value unchanged at or below L_ov; otherwise emit integer `-1` and X3's actual YAML comment containing raw value, rounded value, L_ov and the disabled-positive-control explanation. Only say the pose itself exceeds coverage when **raw > L_ov**. Deduplication must reject conflicting threshold evidence. Generate all descendants of sentinel actions normally; a sentinel is neither a hard block nor a Proceed anyway reason.

## 6. Alaric prerequisites

**C1. Add `apply-mask`.** Reuse the streaming machinery in `alaric/pose_filter.py`/`select-poses.py`; do not revive energy score/filter actions removed by W8. The new action mapping is:

```yaml
action: apply-mask
input: frag5-range2
mask_input: frag5-range2-range3-mask
mask: 1
```

Add allowed/required keys `{input, mask_input, mask}`, dependency kinds POSE/MASK, output kind POSE, and strict `mask` integer 1 or 2 validation to `middle/schema.py`. Update graph cross-validation and inherited metadata in `middle/graph.py`/`middle/resolve.py`, sigil handling in `middle/sigil.py`, execution in `middle/backend.py` and the supported `middle/templates/` variants, and result discovery/deployment in `middle/result.py`/`middle/deploy.py`. Resolve mask side 1 to `mask1.npy`, side 2 to `mask2.npy` from the mask-pair output directory; the existing ordinary-mask `mask.npy` result path is insufficient. Validate the input against the mask producer's corresponding source pool/ordering; reject wrong sides, lengths, incompatible fragments or malformed masks rather than selecting unrelated rows. Preserve pool metadata and pose order as required by the mask. Include selector and both dependency identities in its sigil. Update `mask-common-conformer`'s comment/documentation to point to this consumer.

**C2. Support one-input `identity`.** Make `input2` optional in the schema; absence means deduplicate `input1`, bucket by bucket, reading each file once. Update `middle/graph.py` dependency/cross-validation, `middle/resolve.py` fragment/sequence traversal, `middle/pool_graph.py` deduplication classification, `identity_filter.py` execution/provenance and backend/template dispatch to handle absence without inserting `input2: null`. Add an explicit single-input CLI mode so omission is not confused with the old three positional arguments. Keep the two-input CLI argument order.

For two-input identity with distinct dependency result sigils, preserve the existing parameter/dependency labels, sigil, and result artifacts byte-for-byte; add no new mode/default field to that branch. When both dependencies resolve to the same result sigil, use single-pass deduplication and emit both legacy maps, but add `same_input_dedup_version: 1` to the hashed parameter record. This explicitly invalidates old duplicate-producing self-intersection results, whose cache identity did not include executable code. The single-input form has only the `input1` dependency and a distinct body; it remains marker-free. Prove the distinct-input compatibility with a saved two-input sigil/artifact fixture, and prove same-result and single-input output equivalence. Use the existing pose-record equality/bucket ordering conventions, not coordinate rounding, to deduplicate.

**C3. Version and validate `constraints.json`.** Centralise its load/validation in `middle/resolve.py` so both `sequence:auto` and pair-threshold resolution reject absent/unsupported schemas and malformed keys/pairs/values. Accept the canonical lowercase `crmsd`/`ovrmsd` written by the exporter; retain documented uppercase aliases only with unambiguous, nonconflicting values. Validate adjacency, duplicate pair addresses and agreement between sequence entries and pair sequences. Missing requested entries remain errors. Update existing constraints fixtures/docs to schema 1 rather than silently migrating them. This implements G8 for the newly specified boundary file.

**C4. Supersede the old bundle (E7 complete).** E7 audited every caller/import, transferred X1/X2/X3/X6/digest assertions to real input pins, rendered archive members and selected catalogue trees, then removed `export_bundle.py`, its synthetic fixture and export-only tests. The useful W8 energy-filter/action and backbone checks remain. The impossible `frag-N-bestfit` pool inputs and generic `validations: fresh` dictionaries are absent from production admission.

## 7. Backend wiring and browser behavior

### 7.1 Export cells and download

Wire the export in Preparic `backend.py:build_context()`, after the cells it reads and the share namespace are set up. Read Preparic's `AGENTS.md` before editing that checkout. Use its existing module inclusion and pin wiring; a folder pin arrives in the function as `dict[str, bytes]`, not as a checksum index to resolve manually.

Use these pin names for **both** wrappers; each names `context.<pin>` and has the given transformer celltype:

| Pin names | Celltype |
|---|---|
| `pdbcode`, `cif`, `exclude_pdb_code`, `fraglib_identities` | `str` |
| `biounit`, `ranges`, `stacking_selection`, `threshold_edits`, `best_fit_clip`, `interaction_matrix`, `gaussian_classes`, `energy_threshold_edits` | `plain` |
| `range_specs`, `structure_id`, `fragments`, `best_fit_thresholds`, `best_fits`, `stackings`, `interactions`, `energetics`, `bestfit_stackings`, `gaussian_scores`, `gaussian_validation`, `gaussian_check`, `range_pdbs_aa` | `plain` |
| `range_pdb_files`, `range_pdb_files_aa`, `clipped_fragments` | `folder` |
| `rna_pdb`, `gaussian_model_json`, `ring_refe_a`, `ring_refe_c`, `ring_refe_phe`, `ring_refe_his`, `ring_refe_arg`, `ring_refe_trp` | `bytes` |

Use the existing T19 model pin conversion for `gaussian_model_json`: the source cell is plain, but the existing validator consumes its JSON buffer as bytes. Decode that buffer once for the normalised snapshot and retain the original bytes when calling the existing helper; do not pass a dictionary to a helper expecting JSON bytes.

Additionally wire `validated_pairs` to `context.validate_threshold_edits.result`, `validated_classes` to `context.validate_gaussian_classes.result`, and `validated_energies` to `context.validate_energy_threshold_edits.result`, each as `plain`. These three successful values are null; they are dependency gates, not entries in the normalised snapshot. A required `plain` pin accepts a complete null and delivers `None`; a non-nullable pin does not. Invalid validators fail upstream and block both wrappers. Preserve the validators' existing return behavior and cover the null case in E0.

Define `EXPORT_INPUT_CELLTYPES` from this table once in `backend.py`; use it to wire and declare celltypes for both wrappers and to check their explicit parameter lists in a binding test. Do not maintain two independent pin lists. Follow this actual existing connection idiom:

```python
context.export_admission_transformer = _export_admission
context.export_admission_transformer.modules.export_snapshot = export_snapshot_module
context.export_admission_transformer.pins.best_fits = context.best_fits
context.export_admission_transformer.celltypes.best_fits = "plain"
context.export_admission_transformer.pins.clipped_fragments = context.clipped_fragments
context.export_admission_transformer.celltypes.clipped_fragments = "folder"
context.export_admission_transformer.celltypes.result = "plain"
```

Construct `export_snapshot_module` using the existing source-module loader used for `stage4_module`. Attach the Preparic helper modules it imports as well. Wire every other pin from the table in the same way. Then expose and register the results:

```python
context.export_admission = context.export_admission_transformer.result   # plain
context.export_admission.share(readonly=True)

context.export_tree_transformer = _export_tree
# Attach export_snapshot, export_gens and export_tree modules, and their imports.
# Wire the same EXPORT_INPUT_CELLTYPES and validator dependencies as above.
context.export_tree_transformer.pins.admission = context.export_admission
context.export_tree_transformer.celltypes.admission = "plain"
context.export_tree_transformer.celltypes.result = "folder"
context.export_tree = context.export_tree_transformer.result

context.shares.add_download(
    {"alaric_project": "export_tree"}, path="export.zip", format="zip"
)
```

The comments stand for the repetitive bindings defined by the tables, not for additional design choices. `_export_admission` calls `export_snapshot.build_admission`; `_export_tree` calls `export_tree.build_tree`. Include precisely the source modules those functions import and exercise both wrappers on the configured spawn workers. The browser URL is `/preparic/export.zip`.

**`export_admission`** is `{schema:1, ready, blocks:[{code, address, message, remedy}], warnings:[{kind, address, message}], digest}`. `blocks` holds the §4.4 hard blocks and, when it can execute, the not-yet-stage-6 reason; `warnings` holds the X5 reasons (§7.2); `digest` is `snapshot_sha256`, null when no snapshot can be built. Expected semantic blocks return `ready:false`; unexpected errors fail the transformer. Upstream failures prevent it from running and are shown using the state graph. A share may still serve its preceding value in any non-complete state; never treat that old value as a new report.

**`export_tree`** raises `ExportNotReady` unless admission is `ready` and the digests agree (§4.3). It must never return null for a blocked export: a null cell is left out of the archive and the GET would answer `200` with an empty ZIP. Suppress its expected refusal only while a current admission says `ready:false`. If admission is ready but rendering, planning or digest agreement fails, show the tree exception as an export error. Never hide every `export_tree` failure under a permanent recomputing message.

Implement these distinct UI states:

| Actual state | UI behavior |
|---|---|
| Required upstream producer or validator is pending/failed/blocked | Disable Export; show the pending reason or existing upstream error. Admission may not execute; do not reuse its old blocks/ready flag as authoritative |
| Admission is complete and `ready:false` | Disable Export; show its addressed blocks. The tree's corresponding `ExportNotReady` is expected |
| Admission is complete and ready; tree is waiting/computing | Disable Export with “export is being recomputed” |
| Admission is complete and ready; tree failed | Disable Export and show the actual tree exception |
| Admission is complete and ready; tree complete | Apply stage/connectivity/local-write and X5 checks, then enable Export |

These are UI explanations, not a second server-side freshness mechanism. The download's cut remains the authoritative check of current graph state.

**The download** answers as the Seamless contract specifies:

| Status | Meaning for export |
|---|---|
| `200` | the ZIP, `Content-Type: application/zip`, `Content-Disposition: attachment; filename="export.zip"` |
| `409` | `export_tree` is not complete; JSON `{"error", "cells": {"export_tree": <reason>}}`. `waiting`/`computing` mean recomputation; `failed` can mean expected refusal **or an unexpected export error**; `blocked` means an upstream dependency prevents execution. Use the state table above to explain it |
| `503` | a buffer could not be resolved, the Context did not answer, or the build timed out |
| `404` | no download registered (backend restarting) |

The GET is read-only: no decision writes, no saved archive, no recomputation. Seamless builds downloads on its own worker pool. Both export transformers are cheap (configuration plus PDB pass-through), so they simply recompute whenever an input changes.

### 7.2 Browser control and Proceed anyway (X5)

Add the Export control to `src/components/Stage6Gaussian.svelte`, wired through the existing stage-6 state in `App.svelte`. Put pure readiness/warning/digest logic in `src/lib/export.ts` and transfer/download handling in `src/lib/export-download.ts`; reuse `stage6.ts`, `project-state.ts` and `seamless.svelte.ts` definitions. Do not add a stage marker or persistence for export or consent.

Export is enabled when the project is in stage 6, the client is connected, no write is pending or debounced, existing stage-6 result-echo checks pass, admission is ready, and the state graph shows the admission and tree complete. Match the admission cell's state-graph checksum to the checksum of the admission share value the tab has read; do not mistake an older shared value for the reported complete result. Reuse the client's existing checksum/state lookup conventions. Otherwise disable Export and use §7.1's state table for the reason. The button never arms itself to fire later. Gaussian classes are the only editable decision in stage 6 (author clarification, 2026-11-06), so a class edit or Back is the usual change during an export.

The state graph publication can lag; these checks explain and gate the UI, but are not an atomic server admission check. A GET can still answer `409` after an enabled click, and that is normal. After a local edit, use the acknowledged write and subsequent state-graph updates rather than assuming the old complete report is current. The backend download remains the final authority; no input-checksum vector, browser copy of the full snapshot or additional shared freshness cell is required.

`export_admission` lists the X5 reasons, computed from the current kept evidence:

- Any chosen Gaussian validation is amber/red, including non-emitted cells, pose failure and below-RRM-95 computed classification.
- An edited pair cRMSD or ovRMSD has `threshold − raw < 1e-5`. Apply this to kept pairs only, using S9's raw values; bases never participate.
- A true kept energy cell has `bestfit_energy >= edited_threshold`; strict `<` is the acceptance rule and energies have no boundary zone.

When any reason is listed, show one checkbox beside the exact red bold message **“The nearest-to-native structure doesn't pass the filters. Proceed anyway”**, with the reasons available for inspection. Keep a local `acknowledgedDigest`, set only by checking the box; the rendered checked value is `acknowledgedDigest === admission.digest`. A different digest therefore needs acknowledgment. Do not erase the remembered digest merely while results are pending: a change followed by a revert restores the same evidence and may restore its acknowledgment. This does not revive a cancelled request. A sentinel alone neither shows the checkbox nor closes readiness. Hard blocks remain hard blocks even with the checkbox checked.

On an enabled click:

1. Remember admission's non-null `digest` D and create a local active-request record `{digest: D, cancelled: false}`. Disable repeat clicks for that tab and show “Preparing export…”. If warnings are present, require acknowledgment for D before starting.
2. Fetch the URL through the existing backend URL/proxy helper, with `cache: "no-store"` and an `AbortController`. Do not navigate to the URL: that would release a download before the digest check.
3. For `409`, `503` or `404`, show an actionable message using §7.1's state table and release no download. For any other non-200 status, malformed response or invalid ZIP, show an export error and release no download. A `409 failed` is not automatically an ordinary admission block. Never auto-retry later against a different snapshot.
4. For `200`, read `alaric_project/DATA/export-manifest.json` with a ZIP reader (`fflate` in the download module; add it to the manifest/lockfile if no ZIP dependency exists). Require `schema: 1`, `snapshot_sha256 == D` and D still equal to the current admission digest. Also require this to remain the active uncancelled request, stage 6, a connected tab, no pending/debounced writes and the current readiness/acknowledgment checks above. Otherwise discard the response with “Project changed during export; export again.”
5. On Back, disconnect, a new local class edit, an observed export-relevant decision change or a digest change while the request is pending, mark its record cancelled and abort it. Cancellation is local request handling, not a Context generation counter. A late response must remain discarded even after reconnect, re-entering stage 6 or reverting to D. Seamless correctly retains the old cut; the browser decides whether that already-started request should still be downloaded.
6. Only then create an `application/zip` Blob, download it as `<pdbcode>-alaric-<digest-prefix>.zip` and revoke the object URL. Record the downloaded digest; when admission's digest later differs, show “Export is stale”. Two tabs have independent request/checkbox state.

The GET carries no consent. X5 is browser admission, as the contract specifies; X6 is enforced in the backend because `export_tree` raises while blocked. A direct GET receives the archive of a valid snapshot without consent semantics. No shared cell records success or consent.

Restore the button in a `finally` path only for the request that is still active. Handle abort separately from transfer errors, and release response buffers/object URLs. Unit-test the request state transitions independently of the ZIP parser and UI component. This requires no extra persistent state or backend API.

## 8. Delivery sequence and contract updates

Implement in reviewable steps with the following dependencies. File names for new modules are chosen here; extend existing tests in the repositories' normal layout.

| Step | Work and deliverable | Depends on |
|---|---|---|
| E0 | Add a tiny real-Context download probe: nested `folder` members under `alaric_project/`; exact bytes; `409` for pending/failed/blocked cells; successful null on required `plain` pins; immediate downstream invalidation after a changed PUT returns `200`; same-value PUT behavior; edits after the cut; missing-buffer `503`; recomputation and resolvable folder results after restart; proxy URL. Use Seamless's existing `test_share_downloads.py` and nullable-pin tests as references. Record the Seamless revision and observed results | Seamless downloads (`add_download`) |
| E1 | Alaric C1/C2/C3 schema, graph, execution and sigil changes with action tests; document schema 1 constraints and new action keys | None |
| E2 | `export_snapshot.py`: implement §4.5's normalisation, admission, X5 reasons and digest over real cell shapes, including freshness-echo checks and addressed blocks. No Context reads | None; E0 verifies the binding assumptions |
| E3 | Pure generator planner, name allocator, mode classification and DAG validation; table-driven R0–R12 recipes and the §10 golden catalogue | E2's normalised model; E1 action contract |
| E4 | `export_tree.py`: DATA files, manifest and the folder mapping with path validation; validate emitted YAML/JSON with Alaric consumers in tests | E3 |
| E5 | Wire both transformers with the single pin registry and attached source modules; share admission, keep tree unshared, register the download. Check explicit signatures, celltypes, successful validator-null inputs and actual spawn-worker execution; verify through the share server and proxy | E0, E2, E4 |
| E6 | Stage 6 readiness/warning/consent UI, guarded fetch, digest check, download and stale indication | E5; pure UI helpers can start earlier |
| E7 | End-to-end extraction/selection/deployment/run checks, race/failure tests, old-bundle retirement and final contract/document updates | E1–E6 |

Use the existing test layout. Add Preparic `tests/backend/test_export_snapshot.py`, `test_export_gens.py`, `test_export_tree.py` and `test_export_download.py`; add `tests/frontend/export.test.ts` and `tests/frontend/export-download.test.ts` for the two TypeScript helpers. Extend Alaric `tests/middle/` for action schema/graph/execution/sigils and constraints resolution. Store the §10 input/golden as committed test data, not as a test that reads the current mounts. Keep pure fixture tests separate from real Context/spawn, HTTP and action-execution tests so failures identify the layer.

For each step, deliver its implementation, the applicable §9 tests and a short record of the exact command/result. Run targeted Python tests in the repository's declared environment; in Preparic also run `npm test -- tests/frontend/export.test.ts tests/frontend/export-download.test.ts`, `npm run check` and `npm run build` after UI integration. Do not delete existing acceptance tests simply because their old bundle API was removed: transfer the assertions first, then remove obsolete bindings. Do not report a probe or runtime test as passed if only its pure helper was tested.

E7 updated the finalized contracts and usage documentation to match these delivered behaviors:

- Oracle scope/§6 and gate 6→export: replace the deferred design with the export cells, the download and the admission rules, retaining X1/X2/X5/X6/X7 protections. X3 specifies raw actions per stacking/candidate in Gaussian gens, deduplicated reference candidates, both alternative trees and all descendants. Clarify that repeated variants are alternatives, not one action per candidate globally across the ZIP.
- A12/X4 and the boundary's T16/T27 export wording: replace “one energy-filter per true matrix cell” with “score each scored pool against its mode's required ranges”: fixed tandem AND filters; unfixed root-range-only; repeated gen alternatives; no redundant scoring after restricting to a scored pool. Keep the exact energy-threshold file, ε=15 and cascade defaults. Document explicit fragment fields, grow directions and protein ranges according to supported action schemas.
- Add schema 1 `constraints.json`, manifest, kept-fragment archive paths, canonical naming and the C1/C2 action contracts. P6.6/X3 permits the deduplicated reference graph to differ internally from the Gaussian graph.
- X5 and boundary §8.6.4 currently assign computation of the warning condition to the browser. This plan deliberately puts the single warning calculation in `export_admission`; amend those clauses to say that the backend supplies reasons and the browser displays them and enforces local acknowledgment. Do not implement a second warning algorithm in TypeScript. The numerical tests belong to `export_snapshot.export_warnings`; frontend tests verify display, digest-bound consent and download admission.
- Boundary §7.3/§8.6.4: replace deferred export with the input inventory (§4.2) and consistency rules (§4.3). Add `export_admission`/`export_tree` and their transformers to the cell and transformer tables. §8.1's “no custom routes” now permits one Seamless download, `export.zip`, on the existing share server, without a callback, writable trigger or new persistent decision. Export blocks are reported in `export_admission`, distinct from transformer errors in the state graph (§8.4); `export_tree`'s failure while blocked is expected.
- Add the browser digest/consent checks, including that the GET carries no consent. Update Preparic `STAGES.md` and the Alaric action/project-layout docs to explain that exported gens are a catalogue to move into a run; there is no bundled README.

Resolved draft ambiguities are recorded explicitly: a Seamless download of `alaric_project/` supersedes target-directory writing; protein ranges are explicit; fragment/direction fields are limited to schemas that accept them; `-pass` names the non-final first pruning pass regardless of direction; extra binary folds have numbered intermediate names; reference candidates are deduplicated independently of Gaussian stackings. These are implementation clarifications of the contracts and R0–R12, not pending user choices. E0 probed the Seamless download behavior and confirmed the selected transport.

## 9. Acceptance and completion checks

### 9.1 Pure archive and recipe checks

- Use real schema-shaped fixtures for S7/S9/S10/S11/S12/S13/S14/S18/S19/S24/S25, B11/B14/B18 and range files. Assert all `export_tree` keys and manifest inventory, exact reference/range/member bytes, checksum/digest stability, strict schemas, true-cell-only energy edits, edited grow pairs and no original-number shift under clipping. Change each output-relevant input and require a changed identity; view-only changes must not alter it.
- Cover K=1 terminal/interior, either retained candidate, K=2 and K>2, both traversal variants, either/both clipped outers, several stackings of one nucleotide, protein residues on different ranges, fixed cross-range runs and unfixed independent runs. Inspect local dependency edges and final names, including the backward pruning pass and outer grows from pruned finals.
- For folds with three or more raw pools assert every stacking is represented exactly once, the intersections form a binary DAG, and no action names collide. Reference actions deduplicate repeated stacking candidates and reject inconsistent S9 evidence.
- Cover numerical scoring order including negative values, ties and range indices 2/10. Assert all schema-supported explicit/auto fields, unique-before-score for ordinary grows, root-range-only unfixed filters, tandem fixed filters, no energy summation and no redundant filtering after scored restrictions.
- Enumerate gap/open-tail/abutting-run merges, anchor-target and grown-target restrictions, false-cell separation in unfixed mode, range suffix gains/losses, exact mode hoisting, single-range layout and 2-/3-range common-conformer output names. Assert every external input has a possible canonical producer and no merge uses an unrelated unfixed frame.
- Sentinel tests include raw below L_ov but rounded above it, raw above it, rounded within coverage, clipped failing candidates and conflicting evidence. Assert real YAML comments and unchanged descendants; only the correct raw-above-limit case claims that the pose exceeds coverage. Pair ovRMSD above the limit always refuses export.
- Parse every emitted `alaric.yaml` with Alaric's schema. Select representative compatible gens into temporary roots with `DATA/`, construct/resolve the real middle-layer graph, and assert all auto fields/dependencies work. Loading the untouched `alaric_project/` folder discovers no action directory. Conflicting source alternatives are documented and cannot silently overwrite one another in the selection test.
- Build the tree mapping twice from the same inputs and compare keys and bytes. Malformed/missing/version-2 input JSON and invalid source identifiers must produce addressed admission blocks where inputs permit admission to execute. Inject duplicate, prefix-colliding and unsafe **generated** paths into planner/renderer tests: require an export error, not a partial/empty tree or a fictitious admission block. Admission does not run the renderer. Make the frontend show such an unexpected tree error while admission is ready.

### 9.2 Action execution checks

- C1: exercise a real paired common-conformer mask through both `apply-mask` sides, checking retained pose records and metadata against direct streaming selection. Cover empty/all-true masks, incorrect side/input/length/type and mask dependency changes. A selector or source change must invalidate the sigil.
- C2: compare single-input dedup with same-result two-input identity on records with duplicates across file boundaries/buckets, empty and already-unique pools. Assert exact pose-set/order semantics and one source read. For distinct dependency result sigils, preserve the two-input sigil **and artifacts** byte-for-byte: pose order, `map-1.npy`, `map-2.npy` and `identity-filter.json`. For equal dependency sigils, require `same_input_dedup_version: 1` in the hashed record so an old duplicate-producing self-intersection result cannot be reused; keep both maps in the output. The single-input form has no such marker and writes only `map-1.npy`.
- On 1B7F, run small representative selected projects containing Gaussian and working reference anchors, ordinary grow+unique+score, both merge restriction paths and common-conformer masks. Check unique final poses, fixed/unfixed filter semantics and R12 pose-set equivalence on the **same** test inputs. Use bounded fixtures rather than execute every catalogue alternative.
- Deploy a sentinel-containing reference choice. Its sentinel action must deploy but fail at runtime with the invalid-threshold message; independent actions remain valid/runnable. Deployment must not require `ALARIC_ANCHOR_MIN_CELL_FRACTION` or a bundled library/cache.

### 9.3 Live download and browser scenarios

- Through the real share server and proxy, GET returns a readable ZIP whose `alaric_project/` members equal the `export_tree` leaves byte for byte; PDB members equal B1/B3/B4. A blocked, pending or failed export answers `409`, never an archive or an empty ZIP. GET the URL without the UI to prove hard blocks are enforced in the backend.
- Change a class, a polled model/ring input or (by direct PUT or mounted-file edit, since the stage-6 UI cannot) an earlier decision while an export request is pending. Before the cut, require pending/failed `409` or a newly completed consistent tree; after the cut, the old consistent archive may still complete. The browser's request/digest checks decide whether to release it. A missing leaf buffer may give `503`, never mixed contents. Test change→revert: a fresh request can use the restored digest/acknowledgment, but a cancelled request must never resume.
- Hold an old computed share with matching fragment keys but changed protein range/decision/code input. Export remains blocked until the echo checks are current. No mounted file is read behind the input cells.
- Fail each validator/producer and verify that admission may be blocked without publishing a new value. The UI must display the upstream error, not old ready/blocks. Separately test semantic `ready:false`, normal recomputation, and an unexpected tree failure while admission is ready. A required `plain` validator pin receiving successful null must allow both wrappers to execute.
- Show the exact Proceed anyway checkbox for each Gaussian amber/red/below-95, pair boundary and failing energy case; require acknowledgment to release the download. No warning for bases/sentinel alone. Reset acknowledgment when admission's digest changes. A kept-pair L_ov violation or missing true-cell threshold cannot be overridden.
- Two tabs may export concurrently. An edit in one invalidates the other's pending request/consent; late replies after Back, disconnect or abort do not download. A manifest digest different from the click-time digest is rejected even if the archive itself is valid. Browser response/ZIP/schema failures show an actionable message and restore the button. A later edit marks an existing downloaded digest stale.
- After backend and browser restart as required by P2/O6, Export waits for recomputed fresh results. No export trigger, consent, view state or job file is mounted. Successful and refused GETs change no project decisions or files. Verify namespace URL resolution and cache bypass through the proxy.

Completion means E0–E7 pass, the §10 catalogue is a checked golden rather than a prose-only example, real selected projects deploy with the new actions, X1–X7 assertions run against downloaded exports, the old bundle API is gone, and the contract/usage docs describe the delivered download.

## 10. Worked example: 1B7F (current Preparic mounts)

**Inputs.**
- RNA: chain A, 1–12. Clip 2/1, so the kept fragments are frag3–frag10.
- Selected stackings:

  | Nucleotide | Base | Range | Protein residue |
  |---|---|---|---|
  | nuc4 | G | range-3 | 93 |
  | nuc5 | U | range-3 | 135 |
  | nuc6 | U | range-2 | 10 |
  | nuc11 | U | range-2 | 49 |

- Interaction matrix (range-2, range-3):

  | Fragments | range-2 | range-3 |
  |---|---|---|
  | frag3, frag4 | F | T |
  | frag5–frag8 | T | T |
  | frag9, frag10 | T | F |

**Unfixed.**
- range-3: nuc4–5 form a K=2 run. Inner frag4; outer frag3 and frag5.
  - fwd chain: frag6 → frag7 → frag8, stopping at frag9 (no range-3).
  - bwd chain: none (frag2 is clipped).
- range-2: nuc6 is K=1 (frag5-6). nuc11 has frag11 clipped, so it is `frag10` alone (R3, clipping).
  - nuc6 bwd: stops at once (frag4 has no range-2).
  - fwd from nuc6 and bwd from nuc11 meet in frag7–frag9, so there are merges.
- frag5–frag8 interact with and are reachable by both ranges, so their pools are `fragN-range2` / `fragN-range3` (R9), and each gets a common-conformer gen, `common-conformer/gen-frag5-range-2-3/` … `common-conformer/gen-frag8-range-2-3/` (R8):
  - frag5: both anchored.
  - frag6: range-2 anchored, range-3 grown.
  - frag7, frag8: range-3 grown; range-2 from whichever gen the user picks.
- frag3, frag4 (range-3 only) and frag9, frag10 (range-2 only) interact with one range, so their pools are unsuffixed: `frag3`, `frag4`, `frag9`, `frag10` (R9).

**Fixed.**
- nuc4–6 form one K=3 run: inner frag4 and frag5, outer frag3 and frag6. nuc11 gives `frag10`.
- Chains and merges cover frag7–frag9.
- Tandem energy filters for frag5–frag8.

**Catalogue oracle.** Turn these inputs into a committed fixture, independent of the user's current mounts. The expected logical choices are:

| Mode/root | Anchor gens (in each of the two trees) | Ordinary grows | Merge targets |
|---|---|---|---|
| Fixed | `gen-frag3-6` fwd/bwd; `gen-frag10` single | fwd frag7–9; bwd frag7–9 | fwd frag7–10; bwd frag6–9 |
| Unfixed/range-3 | `gen-frag3-5` fwd/bwd | fwd frag6–8; no bwd | None |
| Unfixed/range-2 | `gen-frag5-6` fwd/bwd; `gen-frag10` single | fwd frag7–9; bwd frag7–9 | fwd frag7–10; bwd frag6–9 |

Unfixed names at frag5–8 carry their range suffix; frag9/frag10 stay plain. There are four common-conformer gens, for frag5–8. Compare rendered graphs when hoisting: `anchor/gen-frag10` and `anchor-refe/gen-frag10`, `grow/bwd/gen-frag9`, `merge/fwd/anchor/gen-frag10-merged`, `merge/fwd/anchor-refe/gen-frag10-merged` and `merge/bwd/gen-frag9-merged` are common to both modes. The other listed choices require their mode-specific paths. The boundary merges into the frag6 and frag10 anchor pools are split by anchor tree (R5). Retaining both directional anchor variants gives **49 physical gen folders** after these six shared choices are hoisted; assert their paths, actions, external inputs and final pools in the golden. Merges include the anchor targets at the gap ends even though ordinary grows stop before them.
