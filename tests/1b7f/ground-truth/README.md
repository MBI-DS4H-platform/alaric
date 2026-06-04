# 1b7f frag4 stack ground truth

The ground truth for the `1b7f` frag4 stack test is the pair of checksum files:

- `frag4-fwd.arc.CHECKSUM`
- `frag4-bwd.arc.CHECKSUM`

Each contains the canonical sha256 of the corresponding organized `.arc`
output. We deliberately do **not** commit the decoded pose rows: they are
~688 MB of TSV and the checksum pins the result exactly.

## Canonical checksum recipe

The checksum is the sha256 over the raw bytes of the organized **plain**
`.arc` files (`poses-1.arc`, `poses-2.arc`, ...) concatenated in
`poses.discover_organized` numeric order. `.arc.zst` is never hashed: zstd
is not byte-reproducible across versions/levels, which is why `organize.py`
emits plain `.arc` for organized output. `tests/1b7f/arc-checksum.sh` is the
single implementation of this recipe; both test scripts call it.

Because the hash is over the binary `.arc` layout (spec'd in
`pose-writer-plan.md`), it is insensitive to TSV formatting / numpy version
drift. It *is* sensitive to any change in the `.arc` byte layout or the
canonical organized ordering -- those are exactly the regressions it guards.

## Provenance: how these checksums were blessed

A checksum over alaric's own output is self-referential, so the values here
were validated **once** against an independent implementation (crocodile,
read-only reference) before being recorded:

1. Regenerated crocodile old-format output anew for the original random
   `--test-conformers 100 --test-rotamers 1000` script:
   `frag4-fwd/` (15,908,584 poses), `frag4-bwd/` (17,643,720 poses).
2. Decoded fresh crocodile via crocodile's own `unpack_poses`, sorted rows
   by `x, y, z, conformer, rotamer`; its sha256 reproduced the previously
   committed crocodile reference exactly (crocodile is deterministic).
3. Ran the alaric stack + `organize.py` scripts as-is, decoded the
   organized `.arc` via `poses.decode_pool`, sorted by the same key.
4. The alaric rows were **identical** to fresh crocodile for both fwd and
   bwd (independent codepaths) -- alaric blessed.
5. Recorded the sha256 of the blessed organized `.arc` here.

Blessed on 2026-05-18 against crocodile at `crocodile/tests/1b7f`
(`stack-frag4-fwd.sh` / `stack-frag4-bwd.sh`,
`--test-conformers 100 --test-rotamers 1000`).

The bwd checksum was updated on 2026-05-19 after `stack-frag4-bwd.sh`
changed to deterministic first-index selection
(`--conformer-range 1 100 --rotamer-range 1 1000`), producing
18,837,845 poses.

Equality means identical sorted physical pose rows -- independent of `.arc`
file boundaries -- which is then frozen as the binary `.arc` checksum.

## Re-bless on 2026-06-04: stacking-distance filter fix

The original blessing above is **invalid for the `mask_a` axial-distance
filter** and was replaced on 2026-06-04.

Root cause: `stack.py`'s distance filter gated the axial stacking distance
with `center_z = |center_vec[:, 2]|` (the raw lab-frame Z component) instead
of the projection of `center_vec` onto the protein ring normal
(`axial = |center_vec . protein_plane|`). The two are equal only when the
protein ring normal happens to point along the lab Z axis; for a general
complex the filter accepts/rejects the wrong poses, and the surviving pose
count becomes a function of the input PDB's (arbitrary) orientation.

Why the original blessing missed it: the cross-validation against crocodile
was not independent *with respect to this bug* -- `crocodile/code/stack.py`
contains the identical `center_z` line, so both implementations agreed on
the wrong result and that agreement was frozen here.

Physical validation of the fix (independent of any self-reference): on the
full 1b7f frag6 stack (`UU`, dom1, resid 131, all conformers) the corrected
filter recovers the `refe-best-fit` near-native pose
(conformer 4005 / rotamer 17663 / grid (52,94,155), RMSD 0.749 A) that the
old `center_z` filter discarded (its best was 1.343 A). See the project
discussion of this fix; the corrected `mask_a` uses `axial`.

Re-blessed checksums (corrected `axial` filter, same deterministic
`--conformer-range 1 100 --rotamer-range 1 1000` selection):

- `frag4-fwd`: 18,936,008 poses (was 16,536,049 under the bug; resid 214
  ring normal is far from lab Z, so the effect is large, +14.7%).
- `frag4-bwd`: 19,095,353 poses (was 18,837,845 under the bug; resid 256
  ring normal is closer to lab Z, so the effect is small, +1.4%).

The corrected output is deterministic: two independent regenerations
produced identical checksums. Note this frag4 subset test
(`--conformer-range 1 100`) is a byte-level regression guard and does *not*
itself exercise native-pose recovery (the frag4 native conformer is outside
range 1..100); the native-pose anchor is the frag6 full run described above.

If you ever re-derive these from crocodile, patch crocodile's
`center_z` -> `axial` first, or the old (buggy) checksums will return.

## Debugging a mismatch

`compare-ground-truth.sh` / `stack-frag4-determinism.sh` only report
pass/fail. To localize *which* poses differ, rebuild the instrumentation on
demand (it is intentionally not part of the permanent suite):

```bash
PATH=/path/to/alaric-env/bin:$PATH   # python3 must be the alaric env

# 1. Regenerate the independent crocodile reference
( cd crocodile/tests/1b7f && bash stack-frag4-fwd.sh && bash stack-frag4-bwd.sh )

# 2. Decode crocodile and alaric to sorted rows and diff, e.g.:
#    crocodile: PYTHONPATH=crocodile/code, read_pose_files + unpack_poses
#    alaric:    PYTHONPATH=alaric/code,    discover_organized + decode_pool
#    column_stack((translations, conformer, rotamer)), then
#    np.lexsort((c4, c3, c2, c1, c0)); diff the two TSVs.
```

(Environment note: in some sandboxes `/usr/bin` precedes the conda env on
`PATH`, so bare `python3` is not the alaric python. Prepend the env `bin`
explicitly, as above, or the scripts will fail importing numpy.)
