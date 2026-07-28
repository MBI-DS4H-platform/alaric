# Anchor Action Design Notes

## Current anchor action

The anchor action creates an initial pose pool for one dinucleotide fragment by
placing a selected nucleotide base near a selected aromatic protein residue. Its
purpose is to seed fragment assembly from plausible stacking interactions rather
than from unconstrained rigid-body placement.

Inputs are a protein PDB, a protein residue id, a dinucleotide sequence, one
selected nucleotide (`first` or `second`), optional PDB-code exclusions for the
fragment library, an angle threshold, a dihedral threshold range, and a margin.
The middle layer resolves `auto` values, chooses the plain protein PDB because
anchor residue ids refer to original residue numbering, runs `anchor.py`, then
runs `organize.py` on the produced pose shards.

The physical filter has three parts:

- Ring orientation: the selected nucleotide ring plane must be close enough to
  the protein ring plane. `anchor.py` computes the sine of the angle from the
  norm of the cross product between the two ring normals and keeps rotations
  whose angle is below `--angle`.
- In-plane orientation: after projecting a nucleotide ring reference vector onto
  the protein ring plane, the code computes a dihedral-like angle against a
  protein ring reference basis. If `--dihedral MIN MAX` is supplied, rotations
  outside that allowed angular interval are removed. If `MAX < MIN`, the allowed
  region wraps around the `-pi/pi` discontinuity.
- Ring-center displacement: for rotations that pass the angular filters, the
  code enumerates integer grid translations near the displacement from the
  rotated nucleotide ring center to the protein ring center. It keeps offsets
  whose axial distance, lateral-corrected distance, and total distance satisfy
  the stacking-distance bounds, expanded by `--margin`.

Implementation outline:

1. `parse_pdb.py` loads the protein. `protein_ring_coordinates()` selects the
   aromatic ring atoms for `--resid`; `calc_plane()` gives a directed normal.
   `anchor.py` builds a protein ring reference frame from the ring center, plane
   normal, a normalized vector from the center to the first ring atom, and the
   cross product of those two axes.
2. `_select_library()` loads the dinucleotide fragment library with only the
   selected base atoms, keeps conformers that survive library exclusions, and
   builds a mask for the selected nucleotide ring atoms in the reduced atom set.
   It also returns packed rotaconformer data and conformer-to-rotamer offsets.
3. `_select_conformers()` and `_select_rotamer_positions()` apply optional
   chunking/debug restrictions. Poses store conformer and local rotamer indices
   as `uint16`, so `anchor.py` checks those ranges before writing.
4. `_process_conformer()` loops over conformers. For each conformer it computes
   the selected ring center, ring normal, and ring reference vector in the
   conformer's local coordinates. It applies every rotamer matrix for that
   conformer, filters by angle and dihedral, then processes the surviving
   rotated ring centers in batches.
5. Candidate translations are represented on the pose grid with spacing
   `sqrt(3) / 3`. `get_discrete_offsets()` receives continuous target
   displacements in grid units and returns nearby integer offsets using a
   precomputed small sphere table. `anchor.py` then evaluates the real
   distance-vector filter against the protein ring normal.
6. Surviving triples `(conformer, rotamer, translation_grid)` are handed to
   `PoseWriter.add_chunk()`. The writer sorts poses into grid buckets, flushes
   unorganized `.arc.zst` shards, and `organize.py` later creates canonical
   `poses-*.arc` output.

The key implementation point is that current anchor work is conformer-centered:
even though the stacking rules are ring-level geometric rules, every
calculation is repeated for each conformer's selected base ring after applying
that conformer's own rotamer list.

## Generic ring-level precalculation

A generic anchor precalculation is a cached enumeration of ring-frame candidate
geometry. It is not tied to one dinucleotide conformer. Instead, it assumes a
canonical reference ring frame and precomputes which sampled rotations,
translations, and rotation-translation combinations can ever satisfy a given
anchor filter.

The useful disk artifacts are:

- `rotations.npy`: sampled rotations in the canonical ring frame that survive
  rotation-only prefiltering. These are cache samples, not Alaric rotamer ids.
- `translations.npy`: sampled continuous translations in the canonical ring
  frame that survive translation-only prefiltering. These are not Alaric
  integer grid coordinates.
- `filtered-sizes.npy`: for each retained rotation, the number of retained
  translation indices associated with it.
- `filtered-concat.npy`: the concatenated zero-based indices into
  `translations.npy`; the cumulative sum of `filtered-sizes.npy` gives the
  segment boundaries in `filtered-concat.npy`.

Those names describe one filter's data, but the concept is broader: any anchor
filter that can be expressed in a reference ring frame can produce the same
kind of files. The runtime anchor step should then map each actual conformer's
selected nucleotide ring into that reference frame, consult the precomputed
rotation/translation candidates, and map the accepted candidates back into
normal Alaric pose rows.

Precalculated files are at ring level, not conformer level. Restated: the cache
knows about a canonical ring, not about the exact ring coordinates embedded in
each fragment-library conformer. Therefore every conformer still needs its own
alignment transform that carries that conformer's selected nucleotide ring onto
the canonical ring reference frame before precomputed rotations/translations can
be interpreted correctly.

## Design for `anchor.py` precalculation mode

Add a second execution path to `anchor.py`, selected explicitly by command-line
arguments, that consumes ring-frame precalculation files while preserving the
current `.arc` pose semantics. The default path should remain byte-compatible
with current behavior.

Proposed CLI:

- `--precalculated-anchor DIR`: enable precalculation mode and load
  `rotations.npy`, `translations.npy`, `filtered-sizes.npy`, and
  `filtered-concat.npy` from `DIR`.
- `--anchor-reference-ring FILE`: coordinates for the canonical selected-ring
  frame used when the precalculation was built. This can be required initially;
  later it could be folded into metadata in `DIR`.
- Optional `--precalculated-threshold NAME` or explicit file overrides can be
  added once the naming convention for multiple threshold sets is known.

Validation:

- `rotations.npy` should be either `[N, 3, 3]` rotation matrices or `[N, 3]`
  rotation vectors; internally convert to `[N, 3, 3]` with `_rotamers_to_matrices()`
  or a sibling helper.
- The rotation sampling interval used to build `rotations.npy` must be finer
  than the angular spacing between Alaric rotaconformers. Runtime snapping maps
  each real library rotamer to a nearby cache rotation; it must not quantize the
  pose orientation itself. Several real rotamers may snap to the same cache
  rotation, and one real rotamer should still be emitted as its original local
  rotamer id.
- `translations.npy` should be `[M, 3]` continuous translation vectors in the
  canonical ring frame, in Angstrom-scale coordinates. It should never be treated
  as `int16` pose-grid coordinates. Runtime conversion to the pose grid happens
  only after the translation has been rotated/mapped into the conformer/protein
  frame.
- The translation sampling interval used to build `translations.npy` must be
  finer than `GRID_SPACING`. After conformer-specific rotation and final snapping
  to the nearest Alaric grid point, several `translations.npy` rows may map to
  the same integer grid translation. This is expected and is part of why the
  finer sampling is needed: it avoids losing boundary cases through premature
  grid rounding in the canonical frame.
- `filtered-sizes.npy` should be `[N]` non-negative counts;
  `filtered-concat.npy` should contain zero-based indices `< len(translations)`,
  and `sum(filtered-sizes) == len(filtered-concat)`.
- The reference-ring coordinate file must match the selected nucleotide's ring
  atom count/order, or must carry enough atom names to reorder it into the same
  order as `residue_ring_atom_mask()`.

Reference-frame construction:

- Factor out a helper that builds a ring frame from ring coordinates: center,
  directed plane normal, normalized first-atom vector projected/defined in the
  ring plane, and the perpendicular in-plane axis. This should be used for the
  protein ring, conformer selected ring, and canonical reference ring so handedness
  and atom-order conventions remain identical.
- For each conformer, compute `T_conf_to_ref`, the rigid transform that maps the
  conformer's selected ring frame onto the canonical reference ring frame. In
  row-vector convention used by the current code, this is effectively a rotation
  matrix plus center translation such that ring coordinates from the conformer
  are expressed in the reference ring frame.

Runtime algorithm:

1. Load the protein ring and build its frame as today.
2. Load the fragment library, selected conformers, rotamer ranges, and pose
   writer as today.
3. Load and validate the precalculated arrays and reference-ring frame.
4. For each conformer, compute the conformer-to-reference-ring alignment.
5. Convert each real library rotamer into the same ring-frame representation as
   the precalculated rotations, then snap it to the corresponding precomputed
   rotation index. The snapped index is a lookup key into the ring-level cache,
   not a new pose rotation. This is what preserves pose meaning: downstream
   poses still name the original library rotamer, while the cache only says which
   precomputed translation candidates should be considered for that rotamer.
   Deduplicate at this lookup level by grouping real rotamers that snap to the
   same precomputed rotation index, so the same `filtered-sizes.npy` /
   `filtered-concat.npy` segment is fetched once per conformer/cache-rotation
   group.
6. For each matched precomputed rotation index, read the translation-index
   segment from `filtered-concat.npy` using boundaries derived from
   `cumsum(filtered-sizes.npy)`, gather those translations, transform them from
   reference-ring coordinates through the conformer's local ring alignment and
   into the current protein/world frame, then snap to the nearest pose grid
   coordinates.
7. Because the cache is generic and discretized, run the existing final
   distance-vector checks after mapping. This keeps correctness anchored to the
   same physical criteria and lets the cache serve as a candidate generator.
8. Deduplicate snapped `(conformer, rotamer, translation_grid)` rows before
   writing. Multiple continuous reference-frame translations can collapse to the
   same final grid translation for the same real rotamer; those collapsed rows
   have identical pose meaning and should not inflate the output. This second
   deduplication can be deferred to bounded batches or to the writer path if
   doing it immediately would require too much memory.
9. Emit normal pose rows using the original conformer index and original local
   rotamer index. Do not store precomputed rotation indices in the `.arc` pose;
   downstream tools expect rotamer ids to index the fragment library.

The delicate part is step 5. Current anchor does not choose arbitrary rotations;
it enumerates the library's existing rotamers for each conformer, and that must
remain true. Snapping is therefore required as an indexing operation: each real
rotamer is rounded onto the precalculated rotation grid so the code can retrieve
candidate translations from `filtered-sizes.npy` and `filtered-concat.npy`. The
snapped rotation must not replace the rotamer in the pose row and must not be
materialized downstream as the fragment orientation. If the cache is approximate,
the rotation grid must be finer than the angular spacing of the Alaric
rotaconformers so snapping acts as a stable cache lookup rather than an extra
rotamer discretization. The final angle/dihedral and distance filters should
remain authoritative, and the implementation must decide whether the snapped
rotation's translation segment is a conservative superset. Tests should compare
sorted decoded pose rows between normal mode and precalculation mode on a small
fixture.

This implies two deduplication stages in the implementation. First, group or
deduplicate real rotamers by snapped cache-rotation index so repeated cache
segments are not loaded and transformed unnecessarily. Second, deduplicate final
snapped pose rows, because finer-than-grid continuous translation samples can
collapse to the same Alaric grid translation. The second deduplication is the
one most likely to be memory-sensitive, so it can be deferred and performed in
chunks as long as the final organized pose set is not inflated by duplicates.

Testing plan:

- Unit-test reference-frame construction with synthetic rings: identity frame,
  rotated/translated copies, and reversed/degenerate cases.
- Unit-test precalculated array validation, including inconsistent
  `filtered-sizes.npy`/`filtered-concat.npy` lengths and out-of-range
  translation indices.
- Unit-test translation snapping with a fixture where several continuous
  `translations.npy` samples intentionally map to the same pose-grid coordinate
  after rotation; precalculation mode should write one pose row for that
  conformer/rotamer/grid coordinate.
- Unit-test rotation snapping with nearby library rotamers and a finer
  `rotations.npy` sampling grid; output rows should keep the original rotamer
  ids even when multiple rotamers use the same cache rotation segment.
- Unit-test both deduplication stages: repeated snapped rotation lookups should
  fetch one cache segment per cache rotation group, and repeated snapped
  translations should not produce duplicate final pose rows.
- Add a small generated precalculation fixture whose rotations/translations are
  broad enough to reproduce the current brute-force path for a restricted
  conformer/rotamer range.
- Regression-test by decoding both output directories and comparing sorted rows
  `(x, y, z, conformer, rotamer)`, so `.arc` shard boundaries do not matter.
- Add an invariant test that precalculation mode writes original library rotamer
  indices, never snapped/precalculated rotation-table indices.
