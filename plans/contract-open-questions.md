# Contract open questions

Collected 2026-10-29, when `final/contract-boundaries.md` (CB) and `final/state-transitions.md` (ST) were finalized. This file holds every question still open about the preparation contract:
- the unresolved questions of CB and ST;
- the questions in `open-questions.md` (OQ) that bear on them;
- the open decisions of `oracle-contract.md` (OC) and `draft-implementation-plan.md` (DIP);
- the proposals of OC and DIP that go beyond CB, ST and the author's decisions. They are subject to the author's revision, so each one is a question in its own right (§2.1, §3).

IDs:
- D-numbers are those of OC §7 and DIP §12. D2 is split into D2a and D2b.
- O-numbers are OC's own proposals, I-numbers DIP's.
- V-numbers are checks that need no author decision.

The documents mark each open point **Unresolved** with its ID. Until it is answered, they state the provisional behaviour given here. Recommendations from the plans are in brackets. Answer inline, as in OQ: [Author: …].

## 1. Unresolved in the final contract documents

1. **D2a. What "beyond a stacking anchor" means** (ST §3.4; OC §1 clip, P4.4, P5.3). The clip may never reach beyond a selected anchor (author).
   - **Strict:** no candidate fragment of a selected anchor may be clipped. Locks then never change with the clip, and the matrix reuse touches only the ends of runs. [Recommended by OC and DIP. ST applies it provisionally: the clip control stops at the outermost anchor fragments.]
   - **By residue:** only the anchor's residue must stay kept. Its outer candidate fragment may be clipped, so its lock shrinks with the clip.

2. **D2b. An anchor newly selected inside the kept clip** (ST §4.2 3→4, §11.4b; OC P4.4).
   - Stage 4 flags the clip, and Continue 4→5 stays closed until the user reduces it. [Recommended by OC and DIP; ST's provisional text]
   - Continue 3→4 clamps the clip to the outermost anchor. That costs one more awaited write, and it is an automatic correction (G9).

3. **D3. Rows uncovered when the kept matrix is adapted on Continue 4→5** (ST §4.2 4→5, §11.4a; OC P5.6). Which new rows become true?
   - (a) Apply Recompute's growth rule to the new rows only (|E| ≥ 1, stopping at error rows), and only for a run that touched the old clip boundary. A run that ended earlier stays as the user left it. [Recommended by OC and DIP]
   - (b) The end run extends to the new boundary, and every new row is true.
   - (c) The new rows stay false, and the user extends the run by hand.

4. **D4. Confirm the reading of your answer to ST §11.5.** The question was:
   - Is "every unclipped threshold filled (copied and/or typed)" the right condition for 4→5?
   - Should rows of clipped fragments be required too?

   The answer was "yes". The final documents read it as: **every pair and base value must be filled, clipped rows included**, by Copy and/or typing. A later clip change then never uncovers an empty row. The other reading, unclipped rows only, was what OC and DIP recommended. Which is meant?

5. **D7. Assembly copies** (CB §8.6.5; ST §3.1, M2's basis; OC P2.1, App. B 4).
   - **The mismatch:** the backend never receives the biounit. It strips copy suffixes and extracts asymmetric-unit chains. The browser names, validates and highlights rows per assembly copy.
   - **Your answer to ST §11.9:** chain naming depends on the assembly, because chains may be duplicated, so `biounit` stays in M2's basis.
   - **Still open:**
     - (a) Ranges name asymmetric-unit chains only. Copy suffixes are refused, and the biounit is display-only.
     - (b) The backend applies the assembly operators, so a picked copy is extracted where it was picked. [DIP recommends (b) if picking copies stays part of the workflow; your §11.9 answer points the same way]
   - **Until decided:** from stage 4 on, a range picked on a copy is shown at its asymmetric-unit position (CB §8.6.5).

6. **D8. Kabsch ring normals in stage 3** (CB T4, §7.2; OC P3.2, A10; OQ stage 6 Q5, never answered).
   - **The difference:** stage 3 (T4) uses SVD plane normals. The Gaussian model and A9 use Kabsch fits onto `ring-refe`. The descriptors differ by up to 0.7°.
   - **The question:** should T4 switch to Kabsch normals? Stage 3's numbers would change, and `ring-refe` would become a stage 3 input. [DIP: yes. It is needed before browser/A9 score parity at 1e-6 (O5).]
   - **Not settled by Q6:** your acceptance of OQ stage 6 Q6 makes T18 reuse T4's code, so stages 3 and 6 share whatever normals T4 uses. It does not choose them.

7. **D9. Gaussian class granularity** (CB T20, T22; ST S18; OC P6.3; OQ stage 6 Q1, never answered). The options are:
   - one class per selected stacking, scored on the worst of its unclipped candidates [recommended; every plan assumes it];
   - one class per candidate (fragment × first/second).

8. **D10. Which pose stage 5 scores** (CB T13; OC P5.1; OQ stage 4 Q3, adopted without confirmation). The options are:
   - the discrete best-fit library pose that the fitter writes [recommended: docking can sample it, and the existing `.score` fixtures use it];
   - an exact superposition onto the reference fragment.

9. **D16. Go-ahead items** (OQ "Actions awaiting your go-ahead"). CB T13 and §4 need Alaric's pinned attract-jax to contain the numpy path. The items:
   - Push attract-jax `9453ee6` and `22b23e4` and commit its untracked accuracy data. Then bump Alaric's submodule pointer from `12bba50`.
   - Fix the 10-12 grid bug (a bead exactly on a grid node). It is a prerequisite of the energy-filter cascade.

## 2. Open decisions of the oracle contract

10. **D11. ovRMSD above L_ov = √3 Å** (OC A11, P6.6, X6; CB T23). You ruled that `anchor-refe` thresholds above the limit block the export. Still open:
    - **Scope:** does the block also cover `grow`'s pair ovRMSD, which clamps the same way? [DIP: yes]
    - **Workflow:** an end fragment can exceed √3 (1b7f fragment 11, nucleotide 2: 2.246 Å), and the clip can no longer remove an anchor's fragments. The user's remedies are then:
      - lower the threshold, which fails the best-fit pose and needs "Proceed anyway";
      - deselect the anchor.

      Is that the intended workflow?
    - **Alaric:** should it raise on an `ovrmsd` above √3 instead of clamping silently [DIP: as the first delivery]? Or should it extend its offset tables, which moves L_ov?

11. **D12. Interacting cells without an energy** (OC P5.7). The editor could let a run cover a fragment whose scoring failed, and such a cell has no energy to derive a threshold from. The options are:
    - the editor forbids it and read validation refuses it [recommended];
    - the export is blocked on such a cell.

12. **D13. What triggers "Proceed anyway"** (OC X5; your answer to OQ stage 6 Q2).
    - **Energetics:** energy thresholds are computed only, so the best-fit pose always meets them. Was the energetics trigger meant for user-editable energy thresholds, which no plan has? [DIP: keep them computed-only]
    - **Amber states:** do they trigger the checkbox like red ones? They are fill < ⅓, a pose that fails, and a threshold inside the 1e-5 boundary zone.

13. **D15. Gaps in the RNA range** (OC §1, A12). Alaric numbers fragments from the first residue present and leaves holes. Preparic's plans count from the range's `start`. The options are:
    - (a) the stage-2 gate refuses an RNA range with any unobserved residue [DIP: as the first delivery];
    - (b) holes are supported throughout: clips, addressed matrix rows, runs and the residue mapping.

    Either way Alaric's `sequence: auto` and `rmsd.py --fragment` must be fixed (O3). DIP adds that a chain break with consecutive residue numbers needs its own check.

### 2.1 The oracle contract's own proposals

14. **O1. The oracle contract as the acceptance contract.** Adopt OC's clauses, with their oracles as the test definitions, at this precedence: author decisions, then final CB and ST, then the mini-plans.
15. **O2. Fragment-key parity of A2** (OC A2). The reference cutter must yield exactly the fit's fragment keys, including on an RNA with a numbering gap. Its atom lists must match the fit's atom for atom, and its in-place RMSD must equal the TSV `rmsd` within 1e-4 Å. This is what makes opening stage 4 before the fit safe (your answer to ST §11.7).
16. **O3. One fragment numbering in Alaric** (OC A12, App. B 2). `sequence: auto` (`middle/resolve.py`, `sequence.txt`) and `rmsd.py --fragment` disagree with the fitter after a gap, or when the first residue is not numbered 1. Fixing them is new Alaric work.
17. **O4. A row means the same atoms everywhere** (OC P2.1). The atoms the backend extracts are the ones the user picked, and a row is valid in the browser iff the backend can extract it. Today this fails for assembly copies (D7).
18. **O5. Score parity** (OC P6.2). If a browser score differs from A9's by more than 1e-6, a bug error is shown. This depends on D8.
19. **O6. Backend restart.**
    - **Your answer:** a restart is out of scope for state transitions (ST §11.10).
    - **What OC still requires:**
      - decisions and the stage survive a restart (P2);
      - results read before a restart are not fresh until read again (P5);
      - the viewer waits for S23 to be re-read (P11's oracle).
    - **DIP** tests the same in W1 and W9.

    Keep these as contract requirements, or drop them as well?
20. **O7. The gate guarantee under any interleaving** (OC P3, oracle 4). Stage k is never reached unless its gate held when its decision was last written or reactivated. DIP argues that the per-cell protocol cannot guarantee this (D17).

## 3. The implementation plan's own proposals

21. **D17. Transition admission** (DIP §3, W1, §12).
    - **The problem:** DIP found two races that ST's per-cell marker protocol cannot close:
      - a delayed Continue into a later stage, retried after the orphan cleanup;
      - a threshold edit in one tab while another tab's Continue 4→5 is in flight.
    - **The proposal:** serialise transition acceptance with gate-affecting writes, and check the expected snapshot and freshness at commit time. Ideally this is a guard on the existing share PUTs. Otherwise it is one atomic project-decision envelope, which would change CB §8's writable shares and mounts.
    - **The choice:** approve that correction, or keep ST's protocol and weaken O7?
22. **I1. Activation tokens** (DIP W1). Every Continue gives its marker a fresh activation token and records its parent's token. The valid prefix then also requires matching linkage. This extends ST's bases.
23. **I2. An addressed matrix extent** (DIP W0). S14 stores explicit ordered fragment-row keys and range ids, not just a boolean grid, because a row count cannot tell a 5′ clip from a 3′ clip. ST and your clarification say the clip is read from the matrix's extent; this fixes how.
24. **I3. A write adapter** (DIP W1).
    - **The proposal:** preparic sends its own conditional PUTs, with the marker frozen per intent. On disconnect it drops intents that are unsent or retrying, instead of relying on the retries of `seamless-client.js`.
    - **To check against your answer to ST §11.11:** make no assumption about withdrawing a write, and always queue a write of the correct state.
25. **I4. Validation receipts** (DIP W1). Small results of T10 and T19 would be shared, so that a gate can confirm a backend validation of the current value. CB §8.3 keeps them unshared.
26. **I5. Content identity of the Gaussian model** (DIP W1). A content identity of the model and rings is carried through T17, T18, T21 and T22 and compared across results. B15 and B16 are polled (your answer to ST §11.13), so they can change live. CB echoes only `filter_set_version` in S16.
27. **I6. The fit gate** (DIP W1). Group B starts on the valid stage-3 prefix, through a cheap gate projection, not on S8's `active` flag. Under ST §5.3, a stale but active S8 after a range change (ST §7.2a) restarts the fit while the stage is 2.
28. **I7. Legacy and malformed stored decisions** (DIP W0).
    - A stored decision without a basis loads as undecided, with an explanation. It is never migrated silently.
    - The diagnostics of a malformed file are kept before the startup loader resets it (CB §8.1).
29. **I8. attract-jax's full dependency list** (DIP W0). `minfor.py.DEPS.txt` omits `attract-par.npz`, so A5's inclusion test (CB §4) fails at first. The proposal is to fix that list in attract-jax rather than weaken the test.

## 4. Checks that need no decision

30. **V1. Spawn workers and module-form transformers** (CB §1, T16).
    - **Your execution decision** (DIP §1): eight local spawn workers, and no persistent cache, hashserver or database.
    - **What CB assumed:** it was written for Seamless's default, in which a Python transformer with attached modules runs in-process.
    - **To check:** T16 (with `energy_threshold.py`) and every other module-form step must run on the spawn workers. If one cannot, it moves to staged code or CLI + DEPS (DIP W0, item 6).

## 5. Answered and applied when CB and ST were finalized

| Question | Answer | Applied in |
|---|---|---|
| ST §11.3, D1: Back keeps every stage decision | yes | ST §4.3, §10.3; CB §8.1; OC P4, §7; DIP §12 |
| ST §11.5, D4: the 4→5 condition | "yes", read as every value including clipped rows (confirm: D4) | ST §4.2, §4.3; CB T9–T11, §8.2; OC gate 4→5 |
| ST §11.6, D5: the stage-4 skeleton | explicit Copy; S9 clearly visible beside the table | ST §5.4; CB §8.2 (S9); OC P4.3; DIP §12 |
| ST §11.7, D6: S22 from T6's output | acceptable | CB T24; OC A2, §7; DIP §12 |
| ST §11.8: the table editable in stages 4–6 | yes | ST §4.4 |
| ST §11.9: `biounit` in M2's basis | stays: chain naming depends on the assembly | ST §3.1 (unchanged); D7 |
| ST §11.10, D14: restart and cache | restart out of scope for state transitions; the cache probably does not survive; no persistent cache (execution decision) | ST §1.11, §7, §8, §10.6; OC §7 |
| ST §11.11: withdrawing queued writes | no assumption; always queue a write of the correct state | ST §7.2a, §10.6 |
| ST §11.12: equal-value PUTs | Seamless guarantees no downstream revocation | ST §1.7, §10.6 |
| ST §11.13: read-only Alaric mounts | polled while the backend runs | ST §6, §7; CB §8.4 |
| ST §9.1: proposed changes to CB §8 | applied with the finalization | CB §2, §3, §8 |
| Gaussian precomputation decisions | mask d < 5 Å; cutoffs from the report, rounded down to five decimals | CB §5, §6 |
