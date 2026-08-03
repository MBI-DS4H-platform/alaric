# Provenance propagation

## Goal

Add `alaric-propagate-provenance` to carry a grow action's provenance through
same-fragment filter and identity/merge results, writing the resulting
`prop-provenance.npy.zst` beside the representative's poses. The derived file
is intentionally not part of the result checksum.

## Command behaviour

`alaric-propagate-provenance GRAPH REPRESENTATIVE local|remote [--branch POOL]`
loads an `alaric-pool-graph` JSON file, validates the representative, and walks
its parents until it reaches grow provenance. A merge requires `--branch` to
select one graph parent. It resolves sigils under `../CACHE/results` locally or
`ALARIC_REMOTE_RESULT_DIR` remotely. Local mode runs the propagation directly;
remote mode writes a mode-2 shell script and uploads it to the remote project
deployment directory.

`alaric-propagate-provenance --pool-dirs DIR ...` executes the resolved path.
The last directory supplies grow `provenance.npy` (raw or zstd); preceding
directories supply filter provenance or merge maps. For a merge, the next
directory's sigil is matched to `pose_dir_1` / `pose_dir_2` in
`identity-filter.json`, selecting `map-1.npy` / `map-2.npy` without requiring
logical pool names at this stage. The result is compressed atomically into the
first directory.

## Supporting changes

- Exclude `prop-provenance.npy[.zst]` from pose result checksum indices.
- Add `alaric-provenance-download`, a selective variant of result download that
  transfers only grow/filter/map provenance arrays and their compressed forms.
- Let `alaric-chain` consume propagated provenance for a representative's
  cross-fragment grow route before falling back to intermediate arrays.

## Verification

Unit tests cover local and remote graph mode, representative and branch errors,
filter and merge composition, raw/zstd handling, selective download filters,
checksum exclusion, and the chain fast path/fallback.
