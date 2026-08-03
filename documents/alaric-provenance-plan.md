# Provenance propagation

## Goal

Add `alaric-propagate-provenance` to carry a grow action's provenance through
same-fragment filter and identity/merge results, writing the resulting
derived provenance beside the representative's poses. Output is keyed by the
fragment the lineage grows from: a filter-only lineage writes dense
`prop-frag<X>-provenance.npy.zst`, a lineage containing any identity/merge map
writes relational `prop-frag<X>-map.npy.zst`. Derived files are intentionally
not part of the result checksum.

## Multiple grow routes

A representative may reconnect two grow routes rather than continue one. In
`tests/1b7f-gaussian-anchor-all-unbound`, `frag9-reconnect` merges
`frag9-fwd-filter-uniq` (forward, ultimately grown from fragment 8) with
`frag9-bwd-restrict` (backward, grown from fragment 10). Both are real
provenance and both are needed: the pool graph lists that representative as the
target of two links, one per source fragment, and `alaric-chain` composes one
route per link. So every arm is propagated, into its own file, and the source
fragment in the name is what lets a link pick the route that belongs to it.

Arms that end in a source pool — an anchor has no provenance of its own —
carry nothing to propagate and are dropped, so merging a grown pool with an
anchored one still resolves to a single file. A representative whose every arm
ends that way (`frag4-merged-filter`, `frag10-anchor-filter`) writes nothing;
it only ever *sources* grow actions, and chain never asks it for a route.

## Command behaviour

`alaric-propagate-provenance GRAPH REPRESENTATIVE local|remote [--branch POOL]`
loads an `alaric-pool-graph` JSON file, validates the representative, and walks
its parents until it reaches grow provenance, forking at each merge. `--branch`
is an optional restriction to particular merge arms; it is required only when
two arms would carry the same source fragment and hence collide on one
filename. It resolves sigils under `../CACHE/results` locally or
`ALARIC_REMOTE_RESULT_DIR` remotely. Local mode runs the propagation directly,
once per arm; remote mode writes a mode-2 shell script with one command per arm
and uploads it to the remote project deployment directory.

`alaric-propagate-provenance --source-fragment X --pool-dirs DIR ...` executes
one resolved path. The fragment is passed explicitly because a result dir is
named by sigil and records no fragment of its own. The last directory supplies
grow `provenance.npy` (raw or zstd); preceding directories supply filter
provenance or merge maps. For a merge, the next directory's sigil is matched to
`pose_dir_1` / `pose_dir_2` in `identity-filter.json`, selecting `map-1.npy` /
`map-2.npy` without requiring logical pool names at this stage — which also
picks the right map per arm. The result is compressed atomically into the first
directory. If no `map-X.npy` is encountered, it is
`prop-frag<X>-provenance.npy.zst`, a dense vector indexed by representative
pose. If any map is encountered, it is `prop-frag<X>-map.npy.zst`, an `N x 2`
relation with columns `(grow_source_pose_id, representative_pose_id)`, using
the same parent/child orientation as `map-*.npy`. The two forms are mutually
exclusive *for one fragment*, so writing one removes stale copies of either
form of the other, and of the unkeyed `prop-provenance` / `prop-pair` names
this scheme replaces. Other fragments' files are left alone: they describe
other routes.

## Supporting changes

- Exclude `prop-frag*-provenance.npy[.zst]` and `prop-frag*-map.npy[.zst]`
  (and the superseded unkeyed names) from pose result checksum indices.
- Add `alaric-provenance-download`, a selective variant of result download that
  transfers only grow/filter/map provenance arrays, every propagated file of
  either form for any fragment, and their compressed forms.
- Let `alaric-chain` consume dense propagated provenance or the propagated
  sparse pair relation for a representative's cross-fragment grow route before
  falling back to intermediate arrays, selecting the file by the link's source
  fragment so a reconnected representative serves each of its links correctly.
- A file left under an unkeyed name is a pure rename to the keyed one when its
  representative has a single route, since that is the route it holds; on a
  multi-route pool it records only one arm without saying which, so it has to be
  recomputed. Falling back past such a file usually fails on an obsoleted
  intermediate — the reason the file was made — so `alaric-chain` names it in
  that error rather than reporting the missing intermediate alone.

## Verification

Unit tests cover local and remote graph mode, representative and branch errors,
filter and merge composition, dense-versus-pair output selection, raw/zstd
handling, selective download filters, checksum exclusion, and the chain fast
path/fallback for both representations. For multiple routes they additionally
cover one lineage per source fragment out of a reconnect, `--branch` as a
restriction, the same-fragment collision error, dropped anchor arms, one
representative's files not clobbering each other, and a chain whose reconnected
representative resolves both of its links from propagated files alone.
