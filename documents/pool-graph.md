# The pool graph

The last step of Alaric links the poses of consecutive fragments into chains of
compatible poses, one pose per fragment. Two poses of consecutive fragments are
compatible when one was grown from the other. Each chain is a model of the RNA,
or of the stretch of it that the run covers.

To find the chains, Alaric must know how the pools of a run are connected.
`alaric-pool-graph` works this out from the run's actions and writes it to a
small JSON file, the **pool graph**. `alaric-chain` reads the pool graph to
count the chains or to build them. The README's section
[The whole picture](../README.md#the-whole-picture) shows where this step fits
in the workflow.

## How pools are connected

Every pool records where each of its poses came from:

- **A grow** records, for each new pose, the pose of the neighbouring fragment
  that it was grown from.
- **An energy filter** records which pose of its input each kept pose is.
- **A merge**, and the removal of duplicates, record the matching pose in each
  input.

These records are small files next to the poses, in the pool's `results/`:
`provenance.npy.zst` for grows and filters, and `map-1.npy.zst` and
`map-2.npy.zst` for merges.

A fragment usually has several pools. In the forward anchor gen `gen-frag2-3`,
fragment 2 has four:

| Pool | Made from |
|---|---|
| `frag2-anchor-raw` | the anchor search |
| `frag2-anchor-scored` | `frag2-anchor-raw`, filtered by energy |
| `frag2-anchor-grown` | `frag3`, grown back into fragment 2 |
| `frag2` | the poses that are in both `frag2-anchor-grown` and `frag2-anchor-scored` |

The pool graph lists every pool of the run, with its fragment, the pools it
was made from and the files that connect their poses. It holds no poses
itself.

**Representatives.** A chain takes the poses of a fragment from one pool, the
fragment's **representative**. That is the pool that was made last, with every
filter applied: normally the final pool of a gen, `fragX` or `fragX-merged`.

## Making the pool graph

Work in the run directory, after `alaric-sigil`:

```bash
alaric-pool-graph -o pool-graph.json
```

It prints the representatives. This run has the anchor gens for fragments 2–3
and 5–6, grows to fragments 1 and 4, and the merge at fragment 5:

```
Representatives:
frag1
frag2
frag3
frag4
frag5-merged
frag6
```

Check that each representative is the final pool of its fragment.

`alaric-pool-graph` reads only the `alaric.yaml` and `sigil.txt` files. None
of the actions has to have run, so you can make the graph as soon as
`alaric-sigil` has run. As results come in, `alaric-chain` adds them to the
graph file itself, so there is no need to make the graph again for that. Do
make a new graph whenever you add, remove or edit actions. If an action's
sigil has changed since the graph was made, `alaric-chain` refuses the graph,
so that it cannot use the result of the old version of the action.

The options are:

- `-o FILE` writes the graph to `FILE` instead of the screen.
- `-f dot` writes a drawing of the graph in Graphviz format instead of JSON:

  ```bash
  alaric-pool-graph -f dot -o pool-graph.dot
  dot -Tsvg pool-graph.dot -o pool-graph.svg
  ```

  Each box is a pool, labelled with its fragment and the kind of step that
  made it. The representatives are blue. Black arrows point from a pool to
  the pools it was made from, and red arrows join the representatives.
- `--target POOL` limits the graph to `POOL` and the pools it is made from.
  Repeat it to name several pools. See the next section.
- `--select POOL` makes `POOL` the representative of its fragment. See
  [A fragment reached twice](#a-fragment-reached-twice).

## One connected stretch per graph

A chain runs from the first fragment of the graph to the last, so every two
consecutive representatives must be joined by a grow. If they are not,
`alaric-pool-graph` stops with an error. This happens in two common
situations, and in both you make one graph per stretch with `--target`.

**Two stretches that are not joined yet.** When you build a run step by step,
you may at some point have two anchor gens, but not yet the merge gen that
joins their growth. For example, a run with the anchor gens for fragments 2–3
and 5–6 and a grow to fragment 4 gives:

```
error: no grow joins frag4 (frag4) to frag5 (frag5), so the chain has a gap. The run has stretches that are not joined yet, such as two anchor gens without the merge gen between them. Use --target to make a separate graph for each stretch, or add the actions that join them
```

Make a graph for each stretch:

```bash
alaric-pool-graph --target frag2 --target frag3 --target frag4 -o pool-graph-2-4.json
alaric-pool-graph --target frag5 --target frag6 -o pool-graph-5-6.json
```

Name the final pool of every fragment in the stretch. If you name too few, a
fragment can be represented by an intermediate pool. With only
`--target frag4`, fragment 2 is represented by `frag2-anchor-scored`, because
`frag4` is grown from `frag3`, which is made from `frag2-anchor-scored`, not
from `frag2`. The printed representatives show this.

**Unfixed mode.** In unfixed mode each protein range is a separate molecule,
with chains of its own. A fragment that is reached from two ranges has a pool
for each. Suppose fragments 2–3 stack on range 3, fragments 5–6 stack on
range 2, and both are grown into fragment 4. A graph of the whole run then
stops with:

```
error: fragment 4 has multiple chain lineages (frag4-range2, frag4-range3) in parts of the run that are not connected to each other, such as two protein ranges in unfixed mode, or two stretches that are not merged yet. Use --target to make a separate graph for each part
```

Make one graph per range:

```bash
alaric-pool-graph --target frag2 --target frag3 --target frag4-range3 -o pool-graph-range3.json
alaric-pool-graph --target frag4-range2 --target frag5-range2 --target frag6 -o pool-graph-range2.json
```

`--target` does not follow a common-conformer gen into the other range: a
graph with `--target frag5-range2-common-conf` contains range 2 only.

### A fragment reached twice

If the actions of a run grow into a fragment a second time, for example from
fragment 2 back into fragment 1 and then forward into fragment 2 again, that
fragment has two pools that are not made from each other.
`alaric-pool-graph` then stops with the "multiple chain lineages" error, and
this time the error suggests `--select`.
`--select POOL` makes `POOL` the representative of its fragment, and leaves
the fragment's other pools out of the graph. Pools that were grown from those
are left without a connection and drop out as well. If that leaves a gap in
the chain, you get the gap error, and `--target` can leave the disconnected
part out.

## Counting chains

`alaric-chain` reads the pool graph. Count the chains with `--count`:

```bash
alaric-chain pool-graph.json --count
```

Before it starts, `alaric-chain` adds to the graph file the results that have
become available since the graph was made. It needs, on your machine:

- the poses of each representative;
- the provenance files of the pools that connect consecutive
  representatives. The `links` section of the pool graph lists them. The
  pools between the representatives need only these files, not their poses.

For a remote run, those pools are on the cluster. **Provenance copying** then
comes between `alaric-pool-graph`, which determines the representatives, and
`alaric-chain`: see
[Provenance copying](deployment.md#provenance-copying) in the deployment
document.

`--count` prints, for each representative, how many of its poses lie on at
least one complete chain, and how many of those are distinct poses. Then it
prints the total number of chains.

**Preliminary chains.** `--target` and `--exclude` choose which
representatives `alaric-chain` uses. They must be consecutive fragments. With
one graph for the whole run, you can therefore count the chains of the
fragments that are done, while the others are still running:

```bash
alaric-chain pool-graph.json --count --exclude frag5-merged frag6
```

The fragments that are left out constrain nothing, so many of these chains
will not survive once those fragments are added.

Building the chains with `-o`, and the tools that analyse them, are described
in step 5 of the README, [Build chains](../README.md#5-build-chains).
