# Bridge: Handoff-Ready Implementation Plan

## 1. Context And Goal

Add a production middle action named `bridge` for connecting `fragN` and `fragN+2` through an exact shared `fragN+1` pose pool.

The brute-force workflow grows both endpoints into the middle, materializes two huge intermediate pose dirs, then runs exact `identity`. `bridge` avoids materializing the full grown pools by using Bloom filters over exact middle pose keys, followed by the existing exact identity merge.

Important repo entry points:
- `alaric/grow.py`: growth semantics, source/target layout, cRMSD/ovRMSD filtering, rotamer/conformer handling.
- `alaric/poses.py`: organized pose format, `PoseReader`, `PoseWriter`, bucket translation layout.
- `alaric/organize.py`: organized pose sorting and `--order-array`.
- `alaric/identity_filter.py`: exact identity semantics and `map-1.npy` / `map-2.npy`.
- `alaric/middle/schema.py`, `resolve.py`, `backend.py`, `deploy.py`, `sigil.py`: action integration and non-load-bearing parameter handling.

## 2. Semantics

Bloom key is exact middle pose identity:

```text
(conformer:uint16, rotamer:uint16, tx:int16, ty:int16, tz:int16)
```

This is the same identity domain used by organized pose dirs and `identity_filter.py`.

`bridge` must produce the same final exact middle pose pool as:

```text
grow lower -> middle
grow upper -> middle
identity_filter(lower_middle, upper_middle)
```

Bloom filters may keep false positives. They must not create false negatives. Guardrails must fail loudly and never truncate.

Final output:

```text
poses-*.arc.zst
connections-lower.npy   # [lower_endpoint_pose_index, middle_pose_index]
connections-upper.npy   # [middle_pose_index, upper_endpoint_pose_index]
bridge.json
```

## 3. Action Interface

```yaml
action: bridge
input1: fragN-action
input2: fragNplus2-action
memory: 600G
max-intermediate-poses: 100000000
max-final-poses: 1000000
nprocs: 8
rotamer-chunks: 4
estimator-seed: 0
```

Rules:
- `input1` resolves to the lower fragment.
- `input2` resolves to the higher fragment.
- Fragments must differ by exactly 2.
- Middle fragment is `(lower + higher) // 2`.
- Lower grows `forward`; upper grows `backward`.
- Middle sequence and thresholds are inferred from existing constraints.
- `memory` is the Bloom memory budget.

Non-load-bearing, excluded from sigil:
- `memory`
- `nprocs`
- `max-intermediate-poses`
- `max-final-poses`
- `rotamer-chunks`
- `estimator-seed`
- any future Bloom hash-count override

## 4. Pipeline

For each rotamer chunk, run:

1. Estimate lower and upper full grown sizes from deterministic 1000-pose samples.
2. Choose orientation that minimizes expected first intermediate size:
   ```text
   expected = n_probe * (1 - exp(-k * n_insert / m)) ** k
   ```
3. Grow side2 as Bloom B2.
4. Grow side1 filtered by B2, materializing intermediate I1 with provenance.
5. Build Bloom BI1 from actual I1 keys.
6. Estimate/check expected side2 intermediate size using `len(I1)`.
7. Grow side2 filtered by BI1, materializing I2 with provenance.
8. Organize I1 and I2 with `organize.py --order-array`.
9. Run `identity_filter.py` on I1/I2.
10. Compose final lower/upper connection maps from identity maps plus provenance.

Implication: the second Bloom is built from the actual first intermediate set, which is much smaller than a full grown side. This is the key selectivity improvement.

## 5. Rotamer Chunks

If `rotamer-chunks=N`, repeat the full pipeline sequentially for each rotamer chunk.

For each conformer, split *middle* rotamers (NOT source rotamers):

```python
first = (nrot * chunk) // N
last = (nrot * (chunk + 1)) // N
```

Only target rotamers in `[first, last)` are considered during that pass.

Implications:
- Chunks partition exact middle key space because rotamer is part of the key.
- Running all chunks must equal `rotamer-chunks=1`.
- This reduces intermediate sizes by at least about `N`; often more because Bloom array occupancy decreases.
- It costs repeated passes over endpoint data.
- Chunking by conformer is not chosen because growth rate is too uneven.

## 6. Bloom Implementation

Implement deterministic vectorized hashing in `alaric/bridge.py`.

When processing organized pose buckets, hash with:
- seed from translation bucket `M`
- local payload:
  ```text
  conformer:uint16, rotamer:uint16, Ox:uint8, Oy:uint8, Oz:uint8
  ```

Bloom math:
```text
fpr = (1 - exp(-k * n / m)) ** k
```

Optimize integer `k` over a bounded range, e.g. `1..16`, unless a future non-load-bearing override is provided.

Milestone v1 should implement single-process/direct Bloom first:
- allocate one bitset of `memory`
- set/probe in vectorized blocks
- node-local intermediates only

After semantics pass, add sharded multiprocessing:
- workers produce bounded key/position blocks
- reducers own disjoint Bloom word ranges
- no full `(N,k)` materialization
- no per-worker full Bloom copies by default

## 7. Provenance

Filtered grow emits pose rows plus:

```python
emitted_origin[i] = endpoint_pose_index
```

After organizing:

```python
organized_origin = emitted_origin[order_array]
```

Identity maps:
```text
map-1.npy: [I1_pose_index, final_middle_pose_index]
map-2.npy: [I2_pose_index, final_middle_pose_index]
```

Compose maps according to which physical side I1/I2 represents, but final orientation is always:

```text
connections-lower: [lower_endpoint_pose_index, middle_pose_index]
connections-upper: [middle_pose_index, upper_endpoint_pose_index]
```

Sort final connection arrays. The identity filter deduplicates.

## 8. Milestones

### Milestone 1: Core Hash/Bloom Library
- Add `alaric/bridge.py`.
- Implement key packing/hash functions.
- Implement Bloom set/probe block operations.
- Implement Bloom metadata and FPR/k optimization helpers.
- Tests: deterministic hashes, no false negatives, formula checks.

### Milestone 2: Grow-As-Bloom And Filtered Grow
- Add bridge growth routines, using `grow.py` as the semantic reference.
- Support pose ranges, direction, thresholds, source/target sequence, pdb excludes, and rotamer chunk selection.
- Emit Bloom bits for generated candidates.
- Emit filtered pose shards and provenance for Bloom hits.
- Tests compare against brute-force `grow` on tiny fixtures.

### Milestone 3: Estimation And Orientation
- Deterministic 1000-pose sampling via `estimator-seed`.
- Estimate grown counts per side and rotamer chunk.
- Choose orientation from expected intermediate formula.
- Enforce expected intermediate guardrail.
- Tests cover lower-first and upper-first choices.

### Milestone 4: Intermediate Organization And Identity
- Organize I1/I2 with `--order-array`.
- Build `organized_origin.npy`.
- Run existing `identity_filter.py`.
- Compose connection maps.
- Tests verify provenance after organization and identity.

### Milestone 5: Rotamer Chunk Merge
- Run full bridge pipeline per rotamer chunk.
- Merge chunk-local identity outputs into final result.
- Remap chunk-local middle pose indices to final organized middle indices.
- Verify `rotamer-chunks=1` equals `rotamer-chunks=N`.

### Milestone 6: Middle Integration
- Add `bridge` schema and resolver support.
- Add `connected_pose` output kind.
- Accept `connected_pose` where pose inputs are expected.
- Add single-node local/remote templates.
- Exclude runtime knobs from sigil records.
- Use node-local work dir, promote only final result.

### Milestone 7: Checksums
- Use `seamless-checksum-index <result_dir>` for pose-like results, including existing pose outputs and `connected_pose`.
- Keep score/mask array checksum behavior unchanged.
- Tests prove changing maps/manifests changes checksum.

## 9. Work Layout

Node-local work dir:

```text
bridge-work/
  chunk-000/
    estimate.json
    bloom-full/
    intermediate-first/
      emitted_origin.npy
      order-array.npy
      organized_origin.npy
    bloom-intermediate/
    intermediate-second/
      emitted_origin.npy
      order-array.npy
      organized_origin.npy
    identity/
      map-1.npy
      map-2.npy
      identity-filter.json
    connections-lower.npy
    connections-upper.npy
  final-unorganized/
  final/
```

Only final result files are promoted to the shared result directory.

## 10. Tests

- Hash determinism.
- Bloom no false negatives.
- `grow-as-bloom` key stream matches brute-force grown pose keys.
- `grow-filtered-by-bloom` keeps all exact overlaps.
- Estimator orientation and guardrails.
- Provenance through `organize.py --order-array`.
- Final result equals brute-force grow plus identity.
- Rotamer chunk equivalence.
- Runtime knobs do not affect sigil.
- Whole-directory checksum includes connection maps and manifest.

## 11. Handoff Readiness

The plan is implementation-ready with one deliberate staging point: implement direct single-process Bloom first, then add sharded multiprocessing. The implementer should not reintroduce atom/geometric filters, should not add `bridge` to chunk deployers for v1, and should treat runtime knobs as non-load-bearing guardrail/performance settings only.
