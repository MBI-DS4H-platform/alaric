# Preparic roadmap — open questions

Collected 2026-10-28 from the planning session. Stage numbering: 1 PDB code, 2 ranges, 3 stacking selection, 4 best-fit conformers, 5 energetics, 6 Gaussian stacking filter, then a later export stage. The subagent's recommendation is given in brackets where it made one.

Questions here that bear on the contract documents and are still open are gathered in [contract-open-questions.md](contract-open-questions.md), marked below with their ID there.

## Stage 4: best-fit conformers ([stage4-best-fit-miniplan.md](stage4-best-fit-miniplan.md))

1. In the editable threshold table, should clipped rows stay editable or be locked? [Editable, so edits survive if a clip is undone] [Author: I confirm]
2. Must the computed thresholds be copied into the editable table before moving on to stage 5? [No; require it only at the export stage] [Author: No, they must be copied before stage 5, but the editable table remains visible and editable throughout]
3. The plan adopted these recommendations without your explicit confirmation:
   - **Pose to score:** use the poses the fitter already writes, i.e. the closest pose the docking grid can reach, rather than an exact superposition. [Open: D10]
   - **Residue numbering:** keep the best-fit PDBs' template numbering (chain B, residues 1–2) and renumber in preparic only if needed for display. [Author: never renumber at all]
   - **Template loading:** add an optional templates argument to Alaric's `Reference`. [Author: defer this]

## Stage 5: energetics ([energetics-miniplan.md](energetics-miniplan.md))

1. Protein ranges without a stacking anchor never get interactions. Is that intended? [Author: yes]
2. How should clashes (large positive energies) be displayed? Recompute currently counts them as interacting. [Author: yes]
3. **Unbound protein structures** (open issue 1): [use the bound range now, other PDB entries in phase 2, AlphaFold after that; align by sequence and superimpose on matched CA atoms]. Should unbound energies stay display-only, or also feed Recompute? [Author: defer this]
4. Backend validation that the matrix contains every anchor cell is deferred, as you said. When should it be designed?
[Author: mark it as deferred to the final export stage (not part of this plan)]

## Alaric spin-off: merged score+filter action ([alaric-energy-score-filter-plan.md](alaric-energy-score-filter-plan.md))

1. **Cascade:** keep 5-7 → 10-12 → gridless with margins of 1.0 and 0.1, or drop 10-12 and go 5-7 → gridless? Dropping it costs about 115–160 µs per pose instead of 105–128 µs and needs only one grid per range. [Author: decide at runtime; exit cascade if less than 1000 poses]
2. **Rounding versus the strict `<`:** rounding away from zero makes negative thresholds more negative, so the best-fit pose itself is rejected (−16.5913 becomes −16.60). The choices are: round up with `<` (today's practice), round up with `<=`, or the rule as stated. [Author: my bad, rounding must be '<' ]
3. Should energy thresholds come from the best-fit or the reference energies? [Best-fit, which is today's practice][Author: I confirm]
4. Keep the old `score`, `filter` and `score_add` actions for existing projects and fixtures, or remove them? This also decides whether Alaric's `score-*.sh` scripts move to the gridless path. [Author: remove them]

## Alaric Gaussian port ([alaric-gaussian-miniplan.md](alaric-gaussian-miniplan.md))

1. At RRM-95, the tilt<30 and both-filters cutoffs can't be reached (432 and 428 positives against 437 needed). Is the fallback acceptable: keep the RRM-95 Gaussian-only cutoff and add the prefilter, giving 92.4% and 92.0% recall? The alternative is the fixed-cutoff reading for all variants. [Author: you misread /srv/shared/stacking-prediction-plots/envelope-analysis-5-report.md. The thresholds are to be unchanged, the prefilter is to be added on top, by definition dropping recall under 90 or 95. This is described at the very end]
2. Apply the envelope (distance < 5.7 Å, displacement < 60°) as a hard mask? It doesn't restrict distance at any of the cutoffs. PrepaGaussian asks the same question. [Author: Yes, at the precomputation stage]
3. Should the precomputed arrays (about 350 MB compressed for 12 filters) be generated on demand through Seamless or committed to git like the legacy r90–r99 set? [On demand] [Author: neither. Precomputed arrays must be built by an Alaric generator script, and the folder must be added to .gitignore]

## Stage 6: Gaussian stacking filter ([preparic-gaussian-miniplan.md](preparic-gaussian-miniplan.md))

1. One class per stacking anchor, or one per anchor action (each fragment and first/second nucleotide)? [Per stacking anchor] [Open: D9]
2. Anchors scoring worse than RRM-95 (class 95, shown red): export with a warning, block the export, or prompt deselection in stage 3? [Export with a warning][Author: In the export page, there will be a checkbox next to a red bold text "The nearest-to-native structure doesn't pass the filters. Proceed anyway" . This checkbox must also appear if the energetics or cRMSD or ovRMSD thresholds are not met. This can be computed on the frontend ]
3. Server validation results of 0% fill, fill below ⅓, or a failed pose: warnings with a single export confirmation, or should any of them block the export? [Warnings] [Author: see above]
4. Validate every anchor against all 12 filters at once? That takes about 20–40 s and re-runs only when the fit changes, but edits become instant and the UI can suggest the strictest filter that still emits the best-fit cell. The alternative is validating only the chosen filter, at about 2–6 s per edit. [Author: Alaric must provide a tool to check a small list of anchor poses against all filters, knowing their hierarchy to allow early cutoff. It should run within a few seconfs]
5. Should stage 3 switch to the `ring-refe` ring orientation so its numbers match stage 6? The two currently differ by up to about 0.7° in tilt. [Open: D8]
6. *(My suggestion)* Stage 3 descriptors are actually computed in the backend (`detect_stacking.py`, a Seamless transformer), not in the browser. Should stage 6 run that same transformer on the best-fit conformers, keeping only the Gaussian in the browser, rather than adding a new TypeScript descriptor module? Doing so would also settle question 5.
[Author: I accept your suggestion]

## Cross-cutting

1. **Existing stage 3 gap:** anchors can change without a stage 3 edit when the stored `pdbcode`, `biounit` or `ranges` files are edited outside the UI or in another tab, when a biounit changes, or through auto-selection of new stackings (`src/lib/stackings.ts:127-142`). Downstream read validation catches this and offers Recompute. Should stage 3 itself be fixed? [Author: I will follow-up this in a separate issue]
2. **Modified nucleotides:** aareduce rejects RNA ranges that contain them, so those entries can never reach stage 4. Accept that as a known limitation? [Author: defer this to a later plan]
3. **Export stage:** not yet planned. It would write the Alaric project `DATA/`, `constraints.json`, the interaction map, the RMSD and energy thresholds, and the Gaussian classes. [Author: not quite. Defer this to a later plan ]
4. **Housekeeping:**
   - [alaric-project-layout.md](alaric-project-layout.md) still wrongly says the best-fit tables feed the DATA thresholds. Correct it? [Author: yes]
   - Several mini-plans are well over their length targets. Should they be trimmed before the roadmap is assembled? [Author: not right now]

## Actions awaiting your go-ahead

[Open: D16]

- Push the attract-jax commits that contain the numpy gridless path and the accuracy analysis (`22b23e4`), and bump Alaric's submodule pointer, which is still at `12bba50`. The per-pose data `docs/numpy-grid-free-2048*` is untracked.
- Fix the 10-12 grid bug in attract-jax: the identity pose scores 0.23 too high because a bead sits exactly on a grid point. The spin-off plan lists this as a prerequisite.

## Anchor ring-ovRMSD mode, Alaric `anchor-refe` ([test-anchor-miniplan.md](test-anchor-miniplan.md))

Stage 4's bases table and editable thresholds already cover the thresholds, so no new transformer is needed.

1. Should the mode be chosen per stacking anchor or per anchor action (fragment × first/second)? [Per stacking anchor][Author: no choice at all. Both modes will be eventually be exported in parallel]
2. Should the refe mode be a per-anchor override with Gaussian as the default, or an equal choice for every anchor? [Override][Author: no. See above]
3. Should the choice live in a separate `anchor_modes` cell, or as a "refe" value in stage 6's Gaussian class table? [Separate cell; it leaves stage 6 untouched and can ship first][Author: no. See above]
4. **Rounding, which affects stage 4:** plain `ceil` leaves a value that is already a multiple of 0.01 unchanged, and the strict `<` then rejects the best-fit pose. Should stage 4 round to the next multiple strictly above, for both the bases and the pairs thresholds? [Yes; the 18 hand-copied 1b7f values don't change][Author: my bad, rounding must be '<' ]
5. **Thresholds above ~1.73 Å:** Alaric's translation offsets stop at about 1.73 Å (`MAX_OVERLAP_RMSD`), so poses beyond that are silently missed, and end fragments reach 2.25 Å. Warn, block the export, or extend Alaric's limit? [Warn][Author: block the export]
