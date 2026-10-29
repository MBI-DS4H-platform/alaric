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

None left; see §5.

## 2. Open decisions of the oracle contract

None left; see §5.

### 2.1 The oracle contract's own proposals

None left; see §5.

## 3. The implementation plan's own proposals

None left; see §5.

## 5. Answered and applied (when CB and ST were finalized, and since)

| Question | Answer | Applied in |
|---|---|---|
| ST §11.3, D1: Back keeps every stage decision | yes | ST §4.3, §10.3; CB §8.1; OC P4, §7; DIP §12 |
| ST §11.5, D4: the 4→5 condition | every value of the kept rows; rows of clipped fragments are not required (confirmed 2026-10-29) | ST §4.2, §4.3, §9.1, §11.5; CB T9–T11, §6, §8.2; OC P4.3, gate 4→5, §7; DIP W3, §12 |
| D2a: "beyond a stacking anchor" | strict: no candidate fragment of a selected anchor is ever clipped | ST §3.4, §11.4; CB §8.1; OC §1 clip, P4.4, P5.3, §7; DIP W3, §12 |
| D2b: an anchor newly selected inside the kept clip | Continue 3→4 clamps the clip to the outermost anchors (awaited write), automatically and possibly silently: the only exception to G9 | ST §3.4, §3.5, §4.2 (3→4), §10.4, §11.4b; CB §8.1, §8.2 (S12), §8.5; OC G9, P4.4, §7; DIP W3, §12 |
| D3: rows uncovered when the kept matrix is adapted | (a) Recompute's growth rule on the new rows only, at run ends that touched the old clip boundary | ST §4.2 (4→5), §10.4, §11.4a; OC P5.6, its oracle, §7; DIP W5, §12 |
| D7: assembly copies | (b): the backend applies the assembly operators, so a range picked on a copy is extracted, prepared and shown where it was picked (clarified 2026-10-29: today it does not, so this is new T0/T1 work) | ST author list, §4.4 (S2), §11.9; CB §2, T0, T1, §8.2 (S2), §8.6.5; OC P2.1, P2.2, §7; DIP §2, W3, §12 |
| O4: a row means the same atoms everywhere | moot: D7 makes it hold by design | OC P2.1 |
| D15: gaps in the RNA range | (a): the stage-2 gate refuses an RNA range with any unobserved residue. In Alaric, tools that assume fragment N is residues N and N+1 are fixed | ST author list, §4.2 (2→3); CB T1; OC §1, P2.1, A12, §7; DIP §2, W3, W8, §12 |
| O3: one fragment numbering in Alaric | fragment N is the Nth and (N+1)th residue, whatever the residue numbers; `Reference` renumbers ordinally (clarified 2026-10-29), and a window across a gap is not a fragment | OC §1, A12; DIP W8, §12 |
| O5: score parity between browser and A9 | the Gaussians are computed by the backend alone and the browser reads them (clarified 2026-10-29: a new step T28 scores T18's and T4's descriptors and also derives the computed classes; T17 and S16 go; T28 vs A9 parity is a backend test on 1b7f, not a runtime check) | ST author list, §4.2 (2→3, 5→6), §5.2–§5.4, §6, §7; CB §2, T17, T18, T19, T28, §6, §8.2 (S7, S8, S12, S16, S17, S25, B15, B16), §8.4; OC A7, A10, P6.1–P6.3, gate 5→6, App. A; OQ stage 6 Q6; DIP W1, W7 |
| I1: activation tokens | rejected: protection against racing tab actions lies outside the weakened O7/D17 guarantee; keep the semantic bases and orphan cleanup | ST §3.5; DIP W1 (W0 already records it) |
| I4: validation receipts | rejected: browser validation is sufficient for navigation; backend validation and error reporting stay as a second check. The gates keep "T10 not failed" (4→5) and "T27 not failed" (5→6), read from the state graph (author) | DIP W1; ST §4.2; CB §8 (W0 already records the rejection) |
| I5: content identity of the Gaussian model | rejected | DIP W1 (W0 already records it) |
| O1: the oracle contract as the acceptance contract | agreed: OC's clauses are the requirements and their oracles define the tests; precedence: author decisions, then final CB and ST, then the mini-plans | OC sources and precedence; DIP §1, W9 |
| O6: backend restart | only mounted cells survive a restart. Correctness is required only if the browser restarts right afterwards; otherwise behaviour is undefined. So P2 stays (every decision is mounted), P5's restart bullet goes, and P11's restart oracle assumes a browser restart | OC P2, P5, P11, §7 (D14), App. A; ST author list, §8; CB §1 (execution), §8.6.5; DIP §1, W1, W9 (W0 already records it) |
| O7, D17: the gate guarantee and transition admission | the commit-time admission correction is too complex and is rejected. The gate guarantee holds only if tabs do not race; when tab A presses Continue while tab B is not editing, B proceeds to the next stage as if it had pressed Continue too | OC P3 (oracles 4, 5), App. A; ST author list, §3.5, §7.1; DIP §2, W1, W9, §12 (W0 already records it) |
| I7: legacy and malformed stored decisions | agreed: a stored decision without a basis loads as undecided, with an explanation, and is never migrated silently; a malformed file's diagnostics are kept before the startup loader resets it | OC G9, App. A; ST author list, §7; CB §8.1; DIP W1 (W0 already records it) |
| I3: a write adapter | rejected: writes go through `seamless-client.js` and its retries; ST §11.11's rule applies (no assumption about withdrawing a write; always queue a write of the correct state) | ST author list; DIP W1 |
| I6: the fit gate | rejected: group B stays gated on S8's `active` flag, so a stale but active S8 after a range change restarts the fit while the stage is 2 | ST author list, §5.3; OC P10; DIP §1, W1 |
| D8: Kabsch ring normals in stage 3 | yes: T4 (and so T18) use Kabsch normals fitted onto `ring-refe`, which become stage-3 inputs | ST author list; CB §2, T4, T18, §7.2; OC A10, P3.2, §7; OQ stage 6 Q5; DIP §2, W3, W7, §12 |
| D9: Gaussian class granularity | one class per selected stacking, scored on the worst of its unclipped candidates (the recommendation) | CB T22; OC P6.3, §7; OQ stage 6 Q1; DIP W7, §12 |
| O2: fragment-key parity of A2 | agreed | OC A2 |
| D10: which pose stage 5 scores | the discrete best-fit library pose (the recommendation) | CB T13; OC P5.1, §7; OQ stage 4 Q3; DIP §2, §12 |
| D11: ovRMSD above L_ov = √3 Å | the block also covers `grow`'s pair ovRMSD; the remedy is to lower the threshold, with an explicit "Proceed anyway"; Alaric raises on an `ovrmsd` above √3 | CB T23, §6; OC A11, A12, P6.6, X6, §7; DIP §2, W8, §12 |
| D13: what triggers "Proceed anyway" | amber states trigger it like red ones. Energy thresholds are user-editable, in a table that works like stage 4's: it starts empty, is filled by explicit Copy and/or typing, and needs a value in every true kept cell before 5→6. A protein-range edit resets only that range's values (clarified 2026-10-29). A best-fit energy ≥ its edited threshold triggers the checkbox | ST answers list, §2.2, §3.4, §3.5, §4.2 (2→3, 3→4, 4→5, 5→6), §4.3, §4.4, §5.1–§5.4, §6; CB T16, T27, §2, §6, §7.3, §8.1, §8.2 (S10–S15, S24), §8.4, §8.5, §8.6.4; OC §1, A6, P5.2, P5.7, its oracle, gate 5→6, P10, X4, X5, §7, App. A; DIP §2, W1, W5, W8, W9, §12 |
| D16: go-ahead items | go ahead, and add to the implementation plan | CB §4; OC A12, §7; OQ go-ahead list; DIP §2, W4, W8, §12 |
| D12: interacting cells without an energy | the editor forbids them and read validation refuses them. Confirmed consequence: if an anchor's own lock is an error row, no matrix passes, and the anchor must be deselected in stage 3 | ST §4.2; CB T16; OC P5.5–P5.7, §7; DIP W5, §12 |
| I2: an addressed matrix extent | agreed: S14 stores ordered fragment-row keys and range ids beside the grid | ST §3.1, §4.2, §7.1, §10.4; CB §8.2 (S14); OC P5.6; DIP W0 |
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
| Legacy Gaussian filters (r90, r95) | compatibility is a non-goal; no legacy model/JSON/generator/verifier/reproduction requirement | final Gaussian decisions and plan; OC A7/A8; DIP W6; W0 audit |
| Gaussian cache directory | rename from `precomputed-filters/gaussian-rrm/` to `precomputed-filters/gaussian/` confirmed | final Gaussian decisions and plan; CB §1, §5.1; OC A8; W0 audit |
| A7/A8 audit | successful by explicit author ruling; prerequisite handoff accepted | final Gaussian decisions; W0 audit and check record |
