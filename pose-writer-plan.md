# Detailed Handoff Plan: `.arc` Pose Writer, Organizer, and Bash Stack Tests

## Summary
Implement the new `.arc` pose format, deterministic `organize.py`, and alaric `stack.py` support for unorganized `.arc` pose fragments. Tests are bash-only and must run in the `alaric` conda env.

Boundary: `crocodile/` is reference-only. Do not modify any crocodile files, including files also named `crocodile/code/*.py`. Copy tests/data from crocodile into alaric, then edit only alaric copies.

## Authoritative `.arc` Format
Binary layout, little-endian:
```text
bytes 0..6    magic: b"alaric1"
bytes 7..9    M: 3 raw bytes containing signed int8 two's-complement values
bytes 10..11  nO_raw: uint16 number of offsets, with sentinel 0 => 65536
bytes 12..15  nP: uint32 number of poses
then          O: int8[nO, 3]
then          C: uint32[nO]
then          P: uint16[nP, 3]
```

Rules:
- `nO = 65536` when `nO_raw == 0`; otherwise `nO = nO_raw`.
- Empty `.arc` files are not written. If there are zero poses, the pose directory may contain no pose files.
- `M` is signed int8 semantically, even though stored in three header bytes.
- Absolute grid coordinate is `grid = O[offset_index] + 256 * M`.
- `C[i]` is the run length for offset `O[i]`; `sum(C) == nP`.
- In organized files, `P` rows for `O[0]` come first, then `O[1]`, etc.; `C` gives those contiguous mini-bucket lengths.
- A mini-bucket cannot contain more than `2**32 - 1` poses. If one offset has more poses than that, `organize.py` must return a clear error.
- Readers accept `.arc` and `.arc.zst`; organized canonical outputs should be plain `.arc` for deterministic append and byte comparison.

Packing rule:
```text
M = ((grid + 128) // 256)
O = grid - 256 * M
```
Raise if `M` does not fit int8, `O` does not fit int8, conformer/rotamer values do not fit uint16, `nP > 2**32 - 1`, or invariants fail.

Canonical organized order:
- Buckets sorted by `M[0], M[1], M[2]`.
- Offsets sorted by `O[:,0], O[:,1], O[:,2]`.
- Poses inside each mini-bucket sorted by `conformer, rotamer`.
- Organized files named `poses-1.arc`, `poses-2.arc`, etc.

## Implementation Changes
1. **Regenerate crocodile ground truth first**
   - From `crocodile/tests/1b7f/`, run:
     ```bash
     conda run -n alaric bash stack-frag4-fwd.sh
     conda run -n alaric bash stack-frag4-bwd.sh
     ```
   - Decode freshly generated old-format outputs to TSV rows:
     `x, y, z, conformer, rotamer`.
   - Sort numerically by all five columns and store in alaric:
     `tests/1b7f/ground-truth/frag4-fwd.sorted.tsv`,
     `tests/1b7f/ground-truth/frag4-bwd.sorted.tsv`,
     with `.sha256` and README.

2. **Copy tests from crocodile into alaric**
   - Copy relevant bash scripts/data from `crocodile/tests/1b7f/` to `tests/1b7f/`.
   - Create `tests/1b7f/code -> ../../code`, pointing to alaric code.
   - Do not edit the source crocodile scripts.

3. **Rewrite alaric `code/poses.py`**
   - Replace old `.npy/.dat` APIs with `.arc` APIs:
     `split_M_O`, `pack_pool`, `write_arc_file`, `read_arc_file`, `read_arc_header`, `decode_pool`, `discover_unorganized`, `discover_organized`, `PoseWriter`.
   - `PoseWriter` writes random `unorganized-{8 hex}.arc.zst`, flushes the largest bucket at `cache_poses`, and supports multiple processes sharing an output dir.

4. **Add alaric `code/organize.py`**
   - CLI:
     ```bash
     python3 code/organize.py POSE_DIR \
       --capacity 2000000000 \
       --max-poses-per-file 100000000 \
       --nprocs N \
       [--debug]
     ```
   - If `.ORGANIZED-DONE` exists, delete remaining `unorganized-*`, delete marker, exit 0.
   - If no unorganized files exist, exit 0.
   - If organized and unorganized files coexist without marker, error.
   - Merge duplicate offsets, validate each merged `C[i] <= 2**32 - 1`, sort canonically, split files before `nP` or `nO` exceed representable limits, gather/remap/sort mini-buckets, then delete unorganized files only after success.

5. **Adapt alaric `code/stack.py`**
   - Replace `PoseStreamAccumulator` with worker-local `PoseWriter`.
   - Remove `--max-poses-per-chunk`.
   - Add `--cache-size` default `50_000_000` and `--nprocs` default `os.cpu_count()`.
   - Keep stack filtering math unchanged.
   - `stack.py` emits only unorganized `.arc.zst`; scripts call `organize.py`.
   - Update alaric `code/stack.py.DEPS.txt` to include `organize.py`.

6. **Bash tests**
   - `tests/1b7f/stack-frag4-fwd.sh`: copied crocodile command for `--resid 214 --first`, then `python3 code/organize.py frag4-fwd/`.
   - `tests/1b7f/stack-frag4-bwd.sh`: copied crocodile command for `--resid 256 --second`, then organize.
   - `compare-ground-truth.sh`: decode organized `.arc`, sort TSV rows, compare to committed ground truth.
   - `stack-frag4-determinism.sh`: run same case twice with different `--nprocs`/`--cache-size`, organize, and compare `poses-*.arc` byte-for-byte.

## Test Plan
Run from alaric repo root:
```bash
conda run -n alaric bash tests/1b7f/stack-frag4-fwd.sh
conda run -n alaric bash tests/1b7f/stack-frag4-bwd.sh
conda run -n alaric bash tests/1b7f/compare-ground-truth.sh
conda run -n alaric bash tests/1b7f/stack-frag4-determinism.sh
```

Also verify:
- `organize.py` on already organized dir exits 0.
- Mixed organized/unorganized files without `.ORGANIZED-DONE` error clearly.
- Every organized `.arc` validates `sum(C) == nP`, `P[:,2] < nO`, and `P` mini-buckets are contiguous according to `C`.

## Assumptions
- `~/.alaric/fraglib.yaml` is valid for the `alaric` env.
- Ground truth equality means identical sorted physical pose rows, not byte equality with old crocodile files.
- No pytest is added in this pass.
- Crocodile remains read-only reference material throughout.
