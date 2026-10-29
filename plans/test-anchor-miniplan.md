# Test-anchor mini-plan: `anchor-refe` (reference ovRMSD) mode in preparic

This is a plan only; nothing is implemented. It builds on `anchor-ring-ovrmsd-mode.md`, `stage4-best-fit-miniplan.md` (S4), `stage4-thresholds-howto.md` and `preparic-gaussian-miniplan.md` (S6). The mode is Alaric's `anchor-refe` action. It is an alternative to the Gaussian filter, chosen per anchor. Despite the brief's wording, it fits **nucleobase heavy atoms** (2-character names) of one nucleotide against `DATA/reference.pdb`. It does not use the ring sets or `ring-refe/`.

## 1. The threshold, and how much stage 4 already covers
- **Definition.** There is one scalar per anchor action, meaning per (fragment N, `nucleotide` first|second). It is the unaligned RMSD between that nucleotide's base atoms in the best-fit library pose and the same atoms in the reference. The threshold is that value rounded up to 0.01 Å, and Alaric accepts a pose when RMSD **<** threshold.
- **Mapping.** This is exactly row N of S4's bases table: `nuc1_ovrmsd` for `first` and `nuc2_ovrmsd` for `second`. Bases rows exist only for this mode, since `grow` uses the pairs table.
- **Stage 4 already covers the computation, so no new transformer is needed.** These planned S4 pieces produce the values, through Seamless, on the full fit:
  - `seamlessify_refe_best_fit_tables`, which wraps `refe-best-fit-bases.py`;
  - `refe-thresholds.py` / `seamlessify_refe_thresholds`, which rounds the values and also writes the `*_raw` values;
  - the editable `threshold_edits.bases` table.

  The only change to S4 is the rounding rule below.
- **Rounding fix (amends S4).** `ceil` leaves an exact multiple of 0.01 unchanged, and the strict `<` then rejects the best-fit pose. `refe-thresholds.py` should instead return the **smallest multiple of 0.01 strictly greater than raw**, i.e. `(floor(raw·100 + ε) + 1)/100`. This differs from `ceil` only at exact multiples, so the 18 golden YAML values are unchanged. The same fix applies to the pairs values used by `grow`.

## 2. Selecting the mode (stage 6 UI, separate cell)
- **Granularity.** The mode is chosen per stacking anchor, using S6's `stackingKey` and S6's recommended granularity. An anchor in refe mode turns into one `anchor-refe` action per unclipped best-fit dinucleotide of its nucleotide r: fragment i+1 with `first`, and fragment i with `second`. The protein residue is irrelevant here. If two refe-mode stackings share the same r, they produce identical actions, and export deduplicates them.
- **State.** `ctx.anchor_modes` is a new plain cell, mounted file-authoritative at `deployment-mounts/anchor-modes.json` and shared read-write. It follows the `stacking_selection`/clip pattern: a startup loader rewrites a malformed file to the default, and writes go through the pending-write/revision code.
  - Shape: `{schema:1, refe:[key,…]}`. The default `{schema:1, refe:[]}` means every anchor is Gaussian.
  - It is validated inline and by a backend transformer that raises: keys must be selected stackings, with no duplicates.
  - It is reset to the default by the same stage-3 hooks that reset `gaussian_classes`.
- **Why not a "refe" class value inside `gaussian_classes`?** A separate cell keeps S6's schema, its validator and its "Copy computed → editable" button untouched. Copying classes never wipes the mode choices. The refe mode can also ship before stage 6 is finished.
- **Mutual exclusion.** A key listed in `refe` makes its `gaussian_classes` row inert. The row is kept and greyed with the label "refe mode", so its edits survive if the user toggles back. `_gaussian_validation_jobs` skips it, and export ignores it. Exactly one mode per anchor is enforced by construction.
- **UI.** The S6 editable table gets a leading "Mode: Gaussian | Refe" toggle per row, plus a global "All Gaussian / All refe" control.

## 3. Showing and editing thresholds
- **Panel.** A small `RefeAnchorTable.svelte` sits in stage 6. For each refe-mode anchor it shows a row for each candidate (fragment, nucleotide), with these columns:
  - sequence;
  - the computed threshold (rounded);
  - the **editable** value, bound to the same `threshold_edits.bases` cell through S4's cell component, with its "differs from computed" marker and per-cell revert;
  - the validation status (§4).

  There is no second copy of the thresholds, so stage 4 and stage 6 always show the same value. If `threshold_edits` is still `null`, the panel offers S4's "Copy computed → editable".
- **Clipping.** This uses the shared `best_fit_clip`. A clipped candidate is greyed and excluded from validation and export, but stays editable (the S4 rule). An anchor whose candidates are all clipped gets no action and a greyed row.
- **Hover.** Hovering a row overlays the best-fit and reference dinucleotide PDBs in Mol*, as in S4.

## 4. Server-side validation (cheap, preparic env)
`_validate_refe_anchors(anchor_modes, threshold_edits, best_fit_thresholds, best_fit_clip, stackings, stacking_selection, range_specs)` is a Python transformer that needs no Alaric code and no library. Its output, `refe_validation`, is mounted read-only. For each candidate, with thr = the edited value and raw = the `*_raw` value (6 decimals, so error ≤ 5e-7):

| Check | Status |
|---|---|
| thr − raw ≥ 1e-5 | green: the best-fit pose passes |
| \|thr − raw\| < 1e-5 | amber: inside Alaric's boundary-recompute zone (`grow.py:35`), so the outcome rests on float details |
| thr ≤ raw − 1e-5 | red: the best-fit pose is **rejected** |
| 1.3 < thr ≤ 1.73 | amber: above `MAX_OVERLAP_RMSD`, an untested range |
| thr > 1.73 | red: the translation offsets clamp (`grow.py:319`), so poses are silently missed |

Like S6's flags, red and amber are warnings that need one confirmation at export. Export is blocked only while validation is stale or errored. An optional stronger oracle would run `anchor-refe` with `--conformer-range c-c` and check that the best-fit pose (rotamer, grid) is emitted. That is already covered by `tests/1b7f/test_anchor_refe.py:220`, so I would not run it live.

## 5. What the export stage must emit (not built here)
For each unclipped candidate of each refe-mode anchor, deduplicated:
```yaml
action: anchor-refe
fragment: N
sequence: XY        # bases row sequence
exclude: <pdbcode, lower-case>   # same pin as the fit
nucleotide: first|second
ovrmsd: <threshold_edits value>
```
- `DATA/reference.pdb` **must be byte-identical** to the fitter's `rna.pdb` (`pdbs-aa/range-1-aa.pdb`), because the fit and the action read different files. Export should copy the file and record its checksum. No `ring-refe` file is needed, and Alaric computes the reference coordinates just in time.
- Do not use `ovrmsd: auto`, because it reads one project-wide value.
- Downstream wiring, such as `identity` merges of the fwd/bwd candidates (as in `tests/1b7f-anchor-refe`), plus score and filter, is an export-stage decision.

## 6. Implementation steps
1. **alaric** `util/refe-thresholds.py`: apply the strict-greater rounding (§1) and add a test with an exact-multiple input. This folds into S4 step 3.
2. **preparic** `refe_anchor.py`: the `anchor_modes` loader and validator, `_validate_refe_anchors`, and the candidate enumeration, shared with `_gaussian_validation_jobs`.
3. **preparic** `backend.py:_add_refe_anchor_pipeline`: cells, mounts and shares. Add the skip-refe filter to `_gaussian_validation_jobs`.
4. **frontend**: `src/lib/refe-anchors.ts` (candidates, mode toggle, validation display) with Vitest; the share keys `anchor_modes` and `refe_validation`; `RefeAnchorTable.svelte`; and the mode column in `GaussianClassTable.svelte`.
5. **Tests**: `tests/backend/test_refe_anchor.py`. On the 1b7f fixtures, the six `tests/1b7f-anchor-refe` values must come out green, the boundary/red cases must be flagged, clipping must be honoured, and the mode reset must work.

## 7. Risks
- **Chain breaks.** `reference.py` counts residues by position from the first residue, while fitter fragments may skip gaps. Check that both index the same way on a gapped range before export relies on it.
- **Large thresholds.** End fragments reach 2.25 Å (1b7f fragment 11), which is beyond the offset coverage. The UI flags this; Alaric itself does not.
- **No protein term.** Refe-mode anchors depend entirely on a downstream score/filter for protein clashes.

## 8. Open questions for the user
1. Should the mode be chosen per stacking anchor (recommended) or per anchor action (fragment × first/second)?
2. Should the strict-greater rounding replace `ceil` in S4, for both bases and pairs (recommended)?
3. Thresholds above 1.73 Å: warn (recommended), block export, or extend Alaric's `MAX_OVERLAP_RMSD`?
4. Should refe mode be offered for all anchors, or only as an explicit per-anchor override with Gaussian as the default (recommended)?
