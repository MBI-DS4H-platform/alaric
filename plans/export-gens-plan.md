# Export: DATA and fragment pool generators ("gens")

Draft. §3 holds the agreed rules, §4 the code and contract changes they require, and §5 the open holes. Nothing here is implemented yet.

**Relation to existing documents.** [final/oracle-contract.md](final/oracle-contract.md) §6 (X1–X7) defers the design of the export stage. This document is that design. Its obligations still apply: same exclude code and minimum cell fraction (X1), byte-identical reference (X2), the anchor-refe sentinel (X3), proceed-anyway (X5), hard blocks (X6) and one consistent snapshot (X7). §4 (C4) lists where X3/X4 need amending. [alaric/export_bundle.py](../alaric/export_bundle.py) is the current in-memory bundle builder, and §2 says what it does and does not cover.

## 1. Vocabulary

| Term | Meaning |
|---|---|
| Fragment N | The dinucleotide covering nucleotides N and N+1. |
| Range | A Preparic range PDB, `range-K` (`range-1` is the RNA; protein ranges are `range-2`, …). Written as A, B in examples. |
| Pool | The output of a pose-producing action. |
| Gen | Fragment pool generator: a folder of action dirs with final pools, named `fragN` (merge gens: `fragN-merged`; common-conformer gens: `fragN-rangeX-common-conf`; R0), and zero or more intermediate pools. Most gens have one final pool; an anchor-run gen of K nucleotides has K+1 (`gen-frag6-7`: `frag6`, `frag7`), and a common-conformer gen has 2. |
| Main folder | The project root holding `DATA/`. The user builds a run by moving a gen's action dirs into it. |
| Gen-final vs run-final | A gen's final pool is final only within the gen. In the run it is an intermediate: the run's final pools are what remains after merging and dead-end removal. |
| Anchor run | Consecutive anchored nucleotides N … N+K−1 (see R3). |
| Inner / outer fragment | Inner: both nucleotides anchored (frag N … frag N+K−2). Outer: one nucleotide anchored (frag N−1, frag N+K−1). |
| Gen type | Anchor, grow, merge or common-conformer (R2). |
| Mode | Fixed or unfixed: whether the relative orientation of the range PDBs is assumed known (R6). |

## 2. What exists today

- `build_export_bundle(snapshot)` returns a Python dict:
  - validation, hard blocks and the `ready` flag;
  - the reference bytes and their checksum;
  - one `anchor` per selected anchor and the `anchor-refe` actions;
  - a chained `energy-filter` per true matrix cell, starting from `frag-N-bestfit`;
  - `energy_thresholds` and the grow thresholds.
- It writes no files, produces no grow actions, and has no notion of gens.
- Nothing in Preparic calls it, and Preparic has no export code.
- The middle layer reads `DATA/` plus the action dirs directly under the root ([middle/project.py](../alaric/middle/project.py) `action_dirs`). A gen in a nested folder is therefore ignored until it is moved, as intended.

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
- **Exception: merge gens** name their final pool `fragN-merged` (R5). The user chooses gens dynamically, so a merge must be able to sit in the main folder next to the gens it joins. Its restricting pool is another gen's `fragN`, and a merge that ends on an anchored fragment (e.g. `merge/fwd/gen-frag6`) sits next to the anchor gen's `fragN`.
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
  | `-e1`, `-e2`, … | non-final steps of a tandem `energy-filter` chain |
  | `-scored` | last `energy-filter` of a non-final pool |
  | `-mask` | `mask-common-conformer` (R8) |

- Every variant of a gen uses the same intermediate names.
- Examples:
  - `anchor/fwd/gen-frag6-7`: `frag6-anchor-raw`, `frag7-anchor-raw`, `frag6-anchor-scored`, `frag7-anchor-scored`, `frag7-anchor-grown` → `frag7`, `frag6-anchor-grown` → `frag6` (R12, one stacking per nucleotide).
  - Grow gen: `frag3-fwd-grown` → `frag3-fwd-unique` → `frag3`.
  - Inner fragment of a K-run: `frag5-anchor-from-5`, `frag5-anchor-from-6` → `frag5-anchor-ident` → `frag5-anchor-scored`.

**R1. What the exporter writes.**
- `DATA/` and gens. Each gen is a subfolder, nested in a classification path (R7), and holds the gen's actions. There are many more gens than a run uses. The user builds a run à la carte by moving a gen's action dirs into the main folder.
- Every action states `fragment` and `direction` explicitly. `fragment: auto` takes the first `frag<N>` token of the dir name and `direction: auto` needs a name ending in `-fwd`/`-bwd`, which R0's names do not provide. `sequence`, `exclude`, `protein`, `threshold`, `cascade`, `min_cascade` and the grow `crmsd`/`ovrmsd` stay `auto` and resolve from `DATA/`, so the edited values live in one place.
- Each export writes a fresh directory and refuses a non-empty target. It also writes `DATA/export-manifest.json`, which records:
  - `schema: 1`;
  - the snapshot digest;
  - the reference SHA-256;
  - the ranges;
  - the fixed/unfixed gens present;
  - Preparic's P5.8 hand-off and class table (X4: these travel with the export).

**Why gens are chosen dynamically.** Whether a stretch is covered by the fwd chain, the bwd chain or a mix (and where the merge falls) does not change the outcome: after the merge and dead-end removal, the final pruned pools are the same. What differs, and greatly, is the size of the intermediate pose pools along the way. The user therefore picks gens step by step, guided by the pool sizes seen so far. This is why every choice must be exported, why gens must slot into a run already in progress, and why a gen references its inputs by fragment name alone (R0).

**R2. Types.** Gens come in types: anchor and grow, plus merge (R5) and common-conformer (R8). In its simplest form a gen is its namesake action plus an `energy-filter`. ("Mode" is reserved for fixed/unfixed, R6.)

**R3. Anchor gens.**
- They live under `anchor/` (Gaussian `anchor` action) and `anchor-refe/` (`anchor-refe` action). The two trees are identical apart from that action and produce the same names, so each anchor run uses one or the other. Different runs may use different trees.
- An anchored nucleotide k lies in frag k−1 (as its second nucleotide) and frag k (as its first).
- **Source of anchors.** Anchors come from Preparic's candidate anchors (contract P5.3, `stage5.candidate_anchors`), one per (RNA residue, range):
  - `locks` give the unclipped fragments that contain the nucleotide, as matrix row keys with nucleotide 1/2. They already use ordinal fragment numbering (contract §1), so no residue-number arithmetic is needed.
  - `range` gives `protein: range-K`.
  - `stackings` lists every selected stacking collapsed into that anchor, each with its own protein residue.
- **Several stackings of one nucleotide.** When a nucleotide stacks with several protein residues, each stacking gets its own raw anchor action, and all of them are folded by `identity` into the single pool that one stacking would have produced: `fragN-anchor-raw`, or `fragM-anchor-from-M` for an inner fragment. The recipes below are unchanged downstream of that pool.
  - Fixed mode: this covers all stackings of the nucleotide, whatever their range (R6).
  - Unfixed mode: only stackings on the same range are folded. Stackings on different ranges belong to different range-rooted gens (R6, R9).
  - Names of the individual raw anchors: `-1`, `-2`, … appended, ordered by (range, protein residue), e.g. `frag5-anchor-raw-1`, `frag5-anchor-raw-2` → `frag5-anchor-raw`.
- **Single fragment** (the anchor is at a chain end, e.g. nuc1): `anchor/gen-frag1/` contains `frag1-anchor-raw` (the anchor action) and `frag1` (`energy-filter`, input `frag1-anchor-raw`).
- **Single interior anchor** (e.g. nuc7, giving frag6 and frag7): variants `anchor/fwd/gen-frag6-7/` and `anchor/bwd/gen-frag6-7/`. Both first run anchor + `energy-filter` for frag6 and frag7 (intermediates).
  - fwd: `frag7` = grow the scored frag6 pool forward, restricted by the scored frag7 pool; then `frag6` = grow `frag7` backward, restricted by the scored frag6 pool.
  - With one stacking per anchor pool, "restricted by" here is an unrestricted `grow` followed by an `identity` with the scored anchor pool (R12): `frag7-anchor-grown` + `frag7-anchor-scored` → `frag7`. When nuc7 has several stackings, it is a `grow` with `restrict_input`.
  - bwd: the same with the order reversed (frag6 first).
- **K consecutive anchors on the same range** (nucleotides N … N+K−1). The folder is named by the first and last fragment, e.g. `anchor/fwd/gen-frag3-6`.
  1. Each of the K−1 inner fragments, whether there is one (K=2) or several, is anchored from both its nucleotides: inner fragment M gets `fragM-anchor-from-M` and `fragM-anchor-from-M+1`. The two pools are merged by `identity`, then energy-filtered.
  2. If K>2, the inner fragments are pruned against each other by a forward pass and then a backward pass, as in `gen-frag6-7`:
     - Forward pass: grow the first inner fragment's scored pool forward, restricted by the next inner fragment's scored pool, and repeat up to the last inner fragment. The last inner fragment's forward-pass pool is its final pool.
     - Backward pass: grow the last inner fragment's final pool backward, restricted by the previous inner fragment's forward-pass pool (for the first inner fragment, its scored pool), and repeat down to the first inner fragment. Each backward-pass pool is that fragment's final pool.
     - Without the backward pass, only the last inner fragment would be pruned against the whole run. Inner pools are often very small after the forward pass, but not always, so the backward pass is always done.
     - Both passes use `restrict_input`, not R12's grow + `identity`: inner-fragment pools combine several stackings and are much smaller than single-stacking pools, so restricting by them is cheap.
     - Names (fwd variant, inner frag4–frag6): `frag5-anchor-fwd`, `frag6` (forward pass); `frag5`, `frag4` (backward pass).
     - With K=2 there is one inner fragment and no passes: its energy-filtered pool is its final pool.
  3. Each outer fragment is grown from the adjacent inner fragment's final pool: frag N−1 backward from the first inner fragment, frag N+K−1 forward from the last. Each grow is combined with a raw anchor action for that fragment by an `identity` merge, and the merged pool is then energy-filtered. The raw anchor pool gets no `energy-filter` of its own. No separate unique step is needed, since the anchor action yields unique poses by construction.
  - Variants `fwd/` and `bwd/` as for K=1. bwd is the mirror of the above: the first pass runs backward from the last inner fragment, the second forward.
- **Clipping.** Clipped fragments get no actions. A run is recomputed on the kept fragments, so a run that loses one outer fragment degenerates to the single-fragment recipe (1B7F: nuc11's frag11 is clipped, giving `anchor/gen-frag10/`).

**R4. Grow chains.**
- fwd: `grow/fwd/gen-frag3/` produces `frag3` from input `frag2`; `grow/fwd/gen-frag4/` produces `frag4` from `frag3`. bwd: `grow/bwd/gen-frag4/` produces `frag4` from `frag5`.
- The user moves e.g. `grow/fwd/gen-frag3/*` into the main folder.
- Chains stop at the clip boundary.

**R5. Merge gens.**
- Example: `merge/fwd/gen-frag4` grows `frag3` (fwd side) forward, restricted by `frag4` (bwd side). Its final pool is `frag4-merged` (R0 exception). `merge/bwd/gen-frag3` is its mirror image.
- With anchors at nuc1 and nuc7 on the same range, the merges run:
  - fwd: from `merge/fwd/gen-frag2` (grow `frag1`, restricted by `frag2`) to `merge/fwd/gen-frag6` (grow `frag5`, restricted by `frag6`, the anchor pool). The latter is the "all fwd, no bwd" choice.
  - bwd: the mirror image.
- A chain does not carry on past the next anchor run on its own range (fixed: on any range). The anchor pool is the latest restriction target.
- Abutting runs (e.g. nuc4 and nuc6): `merge/fwd/gen-frag5` grows `frag4`, restricted by `frag5` (the nuc6 anchor pool). `merge/bwd/gen-frag4` is the mirror.
- Merges are allowed only when both anchors are on the same range, or in fixed mode.
- Content:
  - Restricted by a grown pool: a single `grow` named `frag4-merged`, with `input: frag3` and `restrict_input: frag4`, and no `energy-filter`. `grow` keeps only output poses present in the restricting pool ([grow.py](../alaric/grow.py) `--restrict-poses`), and that pool is already scored.
  - Restricted by a single-stacking anchor pool: a `grow` plus an `identity` instead (R12). An anchor pool of several stackings (e.g. a K-run's outer fragment) keeps `restrict_input`.
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
  - X is the gen type's tree (`anchor`, `anchor-refe`, `grow`, `merge`) and Y the direction (`fwd`/`bwd`). Z is the gen folder, `gen-fragN`, or `gen-fragN-M` for an anchor run (a hyphen, since `frag67` would be ambiguous once fragment numbers reach 10).
  - Single-fragment anchors have no direction level: `anchor/gen-frag1`.
  - Merges have a direction level like the other trees: `merge/fwd/gen-frag4`.
  - The gen folder is always the last component, e.g. `grow/fwd/unfixed/range-2/gen-frag8`. The user moves `…/gen-*/*` into the main folder.
  - Common-conformer gens exist only in unfixed mode and have no direction or mode level: `common-conformer/gen-fragN-range-X-Y/` (R8).

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
- An anchor pool that combines several stackings is much smaller, and restricting by it is cheap, so it keeps `restrict_input`. This holds whether the stackings are on consecutive nucleotides (a K-run, inner and outer fragments alike) or are several stackings of one nucleotide (R3). The stackings are always on one range, or in fixed mode.
- Single-stacking anchor pools occur in:
  - the `gen-frag6-7` steps (both variants), when nuc7 has one stacking;
  - merges that end on such a pool (`merge/fwd/gen-frag6` into a one-stacking nuc7, the abutting `merge/fwd/gen-frag5` into a one-stacking nuc6, and their mirrors).
- Merges restricted by a grown pool (e.g. `merge/fwd/gen-frag4` restricted by the bwd side's `frag4`) also keep `restrict_input` (R5).
- No `energy-filter` after the `identity`: the result is a subset of the scored anchor pool.
- Names: `frag7-anchor-grown` → `frag7` (identity with `frag7-anchor-scored`); `frag6-merged-grown` → `frag6-merged` (identity with `frag6`). With `restrict_input` the step is a single `grow` named after its output (`frag7`, `frag6-merged`).

## 4. Required changes

**C1. Apply-mask action.** No action applies a mask; one is to be added (R8).
- The `filter` action with `mask_input` and `mask: 1|2` was removed in commit 40efa4f ("Implement W8 energy filters and export admission"). That commit replaced score+filter with `energy-filter`, and the mask route went with it.
- `mask-common-conformer` has had no consumer since. [schema.py:83](../alaric/middle/schema.py#L83) now says to pass the mask to `select-poses.py` by hand.
- The streaming code is still there ([pose_filter.py](../alaric/pose_filter.py)), so the action needs restoring, under its old name or a new one.

**C2. `identity` on a single pool.** The action is to be modified to deduplicate a single pool (R11).
- Today `identity` requires `input1` and `input2` ([schema.py:42](../alaric/middle/schema.py#L42), [identity_filter.py](../alaric/identity_filter.py) takes two pose dirs). Deduplicating one pool means pointing both inputs at the same dir, which reads every file twice and intersects it with itself.
- Proposal: make `input2` optional. Without it, the action deduplicates `input1`, bucket by bucket, in one pass. The two-input form is unchanged, so its sigils and cached results stay valid.

**C3. Replace the action emission of export_bundle.py.** Its emission does not fit gens:
- It emits one `anchor` per selected anchor, not two raw anchors per anchored nucleotide (one per fragment).
- It emits no grows, merges, identity merges or conformer filters.
- It chains `energy-filter` with fixed names from a nonexistent `frag-N-bestfit`.
- It has no fixed/unfixed modes.
- Keep: validation, hard blocks, the X3 sentinel and comments, X5/X6 and the snapshot digest.
- Replace: action emission and the handoff (`fragment_rows`).
- Replace its anchor input as well. It reads a `selected_anchors` list (fragment, first/second, protein, one protein `resid`) and keeps one candidate fragment per entry. Nothing in Preparic produces that list. The new emission reads Preparic's candidate anchors instead (R3, source of anchors): all their locks, and one raw anchor per collapsed stacking. Its numbering is ordinal like theirs, so only the input shape changes. Only the tests build `selected_anchors` today ([test_w8_export_seam_contract.py](../tests/test_w8_export_seam_contract.py), [test_w8_contract_followup.py](../tests/test_w8_contract_followup.py)), and they change with it.

**C4. Amend the oracle contract** ([final/oracle-contract.md](final/oracle-contract.md)).
- X3 ("one `anchor` per selected anchor") and X4 (one `energy-filter` per true matrix cell, per-action layout) describe the old shape and are to be rewritten for gens.
- Also to be added: the new `DATA/constraints.json` and manifest contents (R1), and the apply-mask action (C1).

## 5. Holes found

None open.

## 6. Worked example: 1B7F (current Preparic mounts)

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
