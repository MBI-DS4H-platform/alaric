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

1. Regenerated crocodile old-format output anew:
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
(`stack-frag4-fwd.sh` / `stack-frag4-bwd.sh`, now expressed as
`--conformer-range 1 100 --rotamer-range 1 1000`).

Equality means identical sorted physical pose rows -- independent of `.arc`
file boundaries -- which is then frozen as the binary `.arc` checksum.

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
