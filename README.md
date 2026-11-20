# Alaric

Alaric models a single-stranded RNA bound to a protein by assembling it from
overlapping dinucleotide fragments. Fragment N is the dinucleotide of
nucleotides N and N+1, so consecutive fragments share one nucleotide. Each
fragment is placed by searching a precomputed library of dinucleotide
conformations, each in many orientations, against the protein structure.

A run starts from **anchors**, nucleotides that stack on an aromatic residue
of the protein. From the anchor fragments, Alaric grows the RNA one fragment at
a time, in the 5′→3′ direction, the 3′→5′ direction or both. A pose of the new
fragment is kept only if its shared nucleotide superimposes well on a pose of
the neighbouring fragment. The candidate poses of a fragment form a **pool**,
and every pool is filtered by ATTRACT interaction energy with the protein.
Where growth in the two directions meets, the pools are merged. Finally, the
surviving poses are linked into chains of compatible poses, one pose per
fragment, and each chain is a full model of the RNA.

**Status: alpha.** The workflow below is complete and has been used on real
cases. Interfaces and file formats may still change. For now, Alaric is most
convenient when a reference structure of the complex is available (see
step 4).

Alaric comes with **Preparic**, a web interface that prepares the inputs of a
run. You start in Preparic and then continue with Alaric on the command line.

The workflow has five steps, each described in a section below:

1. Prepare the run in Preparic and download the export ZIP.
2. Unzip it and build an Alaric run à la carte from its contents.
3. Deploy and run the actions.
4. Calculate fragment RMSDs against the reference structure.
5. Build chains: make the pool graph, then build the chains and analyse them.

[The whole picture](#the-whole-picture), between steps 4 and 5, shows how the
steps fit together.

## Installation

Alaric and Preparic are installed separately, each in its own conda
environment, on Linux.

1. **Alaric.** Follow [INSTALLATION.md](INSTALLATION.md). In short:
   - Create a Python 3.12 conda environment named `alaric`. Install the
     `attract-jax` submodule into it first, then Alaric.
   - Set `ALARIC_DIR` to the repository's `alaric/` package directory.
   - Point `~/.alaric/fraglib.yaml` at the fragment library.
   - Build the Gaussian stacking-filter caches once.
2. **Preparic.** Follow Preparic's `INSTALLATION.md`. Preparic creates its own
   `preparic` environment from `environment.yml` and runs Alaric's steps in the
   `alaric` environment.

`ALARIC_DIR` is the only environment variable you need for local runs. The
`ALARIC_REMOTE_*` variables in [INSTALLATION.md](INSTALLATION.md) are needed
only to run Alaric on a remote cluster.

## 1. Prepare the run with Preparic

Preparic is a collaborative web interface. Everyone who opens it in a browser
works on the same project, and every decision is saved as a parameter file on
disk, so you can stop and continue later.

Start it in the `preparic` environment and open the address it prints
(<http://localhost:5173/>):

```bash
conda activate preparic
preparic-start
```

By default, only a browser on the same machine can open Preparic. To work on
the project together, start it with `preparic-start --collaborative` (or
`--collab`, `--co`). Others then open `http://<this machine's IP or
hostname>:5173/`. There is no authentication: anyone who can reach the port
can use and change the project, so do this only on a network you trust.

Then work through the six stages:

1. **Structure.** Enter the PDB code and choose the biological assembly.
2. **Ranges.** Define residue ranges: range 1 is the RNA, the other ranges are
   protein.
3. **Stackings.** Select the RNA–protein stackings that will anchor the RNA.
4. **Best fits.** Review the fragment RMSD thresholds, and clip fragments at
   the 5′ and 3′ ends.
5. **Energetics.** Review the interaction matrix (which fragment interacts with
   which protein range) and the energy thresholds.
6. **Gaussian filters.** Choose a stacking-filter class for each stacking.

Finally, press **Download export ZIP**. Each stage computes defaults from the
structure, and you review and edit them. Preparic's README describes the
stages, their buttons and the export in more detail.

## 2. Build a run à la carte

**Actions and gens.** Alaric works in **actions**. An action is a directory
with an `alaric.yaml` file that describes one step, such as an anchor search,
a grow or an energy filter. Running the action builds a pool of fragment poses
in that directory, and the pool takes the name of the directory.

Preparic groups the actions into **gens**. A gen is the set of actions that
generates the final pool of one fragment, usually called `fragX` for fragment
X, together with a few intermediate pools. Its folder is named after that
final pool: `gen-frag4` generates `frag4`. An anchor gen can generate several
final pools: `gen-frag2-3` generates `frag2` and `frag3`.

**The export.** Unzip the export into a directory of its own. You get one
folder, `alaric_project/`, with the input data and a catalogue of gens:

```
alaric_project/
  DATA/               input data written by Preparic
  anchor/             anchor gens: stacking on the protein
  anchor-refe/        anchor gens: from the reference structure
  grow/               grow gens
  merge/              merge gens
  common-conformer/   common-conformer gens (unfixed mode only)
```

`DATA/` also holds files for you to read, which no action uses:
`best-fit-statistics.json` has the best-fit RMSDs and energies, and
`anchor-refe-bases.pdb` has the reference nucleobases used by `anchor-refe`.

**Runs.** The export is not a ready-made run. You choose gens from it and copy
them into a run directory. One export allows many runs, and you will usually
make several. For example, you might start with a quick `anchor-refe` run as a
test and then move on to `anchor`. Keep `alaric_project/` unchanged. Give each
run its own directory, linked to the export's `DATA/` and to one `CACHE/`
shared by all runs:

```bash
mkdir -p CACHE run-refe && cd run-refe
ln -s ../alaric_project/DATA DATA
ln -s ../CACHE CACHE
```

Alaric commands find the run directory by looking upward for `DATA/`, and use
only the action directories directly inside it. So copy the action directories
inside a `gen-*` folder, not the `gen-*` folder itself. Add one gen at a time:

```bash
alaric-copy ../alaric_project/anchor-refe/fwd/gen-frag2-3/*
```

`alaric-copy` copies only the `alaric.yaml` of an action. So you can also use
it to copy an action from another run, without the results and scripts of
that run. If one of the actions is already in the run, it copies nothing.

Never run Alaric inside `alaric_project/` itself. It stays the unchanged
source for all your runs.

**The cache.** Strictly speaking, the poses are not stored in the action
directory. They are stored in `CACHE/`, and the action's `results` is a
symlink to them. Every result is identified by its **sigil**, a checksum of
its parameters and its inputs, so runs that share a `CACHE/` reuse each
other's results. Suppose run 1 builds fragments 1 → 2 → 3 → 4 and run 2
differs only in the step 3 → 4. Run 2 then computes only that step.
[documents/sigils.md](documents/sigils.md) explains how this works.

**The kinds of gen**

- **Anchor gens** start a run, one for each stretch of stacked nucleotides.
  - They come in two alternatives. `anchor/` places a fragment by stacking it
    on the selected protein residue, using the Gaussian filter class chosen in
    Preparic. `anchor-refe/` instead keeps the poses that lie close to the
    nucleobases of the reference structure, `DATA/reference.pdb`. It is a
    positive control.
  - An anchor gen that covers two or more fragments comes as a `fwd/` and a
    `bwd/` variant. Choose one of them.

  > **The two alternatives look alike but cost very different amounts.**
  > Their action files are almost identical. An `anchor-refe` run usually
  > finishes within half an hour on a laptop. An `anchor` run takes about a
  > day on 10 nodes of a compute cluster. Try the workflow with `anchor-refe`
  > first, and plan to run `anchor` on a cluster.

- **Grow gens** extend a pool by one fragment.
  - `grow/fwd/gen-frag4` makes `frag4` from `frag3` (5′→3′), and
    `grow/bwd/gen-frag4` makes `frag4` from `frag5` (3′→5′).
  - A grow keeps only the poses whose shared nucleotide superimposes well on a
    pose of the parent pool. It then removes duplicates and filters by energy.
- **Merge gens** join 5′→3′ and 3′→5′ growth where they meet.
  - `merge/fwd/gen-frag8-merged` grows `frag7` in the 5′→3′ direction. It
    keeps only the poses that are also in `frag8` from the other side, and
    writes `frag8-merged`.
  - A merge that ends on an anchor pool has an extra `anchor/` or
    `anchor-refe/` level. Choose the one that matches your anchor gens.
- **Common-conformer gens** exist only in unfixed mode, described next. They
  filter the two pools of a fragment that is reached from two protein ranges
  against each other.

**Fixed and unfixed mode.** When the protein has more than one range, the
export offers two modes. They differ in whether the relative orientation of
the protein ranges is assumed to be known. Which fragment interacts with which
range comes from the interaction matrix of Preparic stage 5.

- **Fixed:** all protein ranges share one trusted frame, that of the
  structure.
  - A fragment that interacts with several ranges must pass the energy filter
    of each of them, one after the other. The energies are not summed.
  - The range does not matter for anchoring or growth. Consecutive anchored
    nucleotides form one anchor gen, whatever ranges they stack on. Growth
    continues into any fragment that interacts with at least one range, so
    growth from two anchors always meets, and a merge is always possible.
- **Unfixed:** each protein range is an independent, unrelated molecule.
  - A pose belongs to the range its anchor stacks on. It is meaningful only in
    that range's frame, and it is scored against that range only.
  - Growth from an anchor continues as long as the fragments interact with the
    anchor's range, and stops at the first fragment that does not. Two anchors
    on one range that are separated by such a fragment give two halves that
    never meet. Anchors on different ranges are never merged.
  - A fragment that is reached from two ranges gets a pool for each,
    `frag5-range2` and `frag5-range3`. A `common-conformer` gen keeps only the
    poses whose conformation occurs in both pools, and writes
    `frag5-range2-common-conf` and `frag5-range3-common-conf`. It is optional.
    The exported gens always read the plain pools. To grow from a filtered
    pool instead, edit the `input:` of the downstream gen by hand.

With a single protein range the two modes are the same, and the catalogue has
no mode folders. Otherwise, a gen that differs between the modes sits in a
`fixed/` or an `unfixed/` folder, for example `grow/fwd/fixed/gen-frag4` or
`grow/fwd/unfixed/range-2/gen-frag8`. A gen that is the same in both modes has
no such folder. The export contains both modes, and you choose one by the gens
you copy into a run.

**Choosing the next gen.** Which gens you use to cover the RNA does not change
the final models, once the dead ends have been removed in the final
chain-building step. A stretch may be covered by 5′→3′ growth, 3′→5′ growth or
both, and the merge may fall anywhere. What changes, by orders of magnitude,
is the size of the intermediate pools. So build the run step by step:

1. Start with an anchor gen and run it (step 3).
2. Count the poses of its pools with `alaric-count-poses <pool>/results`.
3. Extend from the side where the pools are smallest.

A gen refers to its input pools by name only (`input: frag3`), so it can use a
pool from any gen that produces it. `DATA/export-manifest.json` lists every
gen's final pools and its input pools.

## 3. Deploy and run the actions

To run an action, you first *deploy* it: Alaric writes the scripts that
compute it. Where and how those scripts run depends on the deployer. This
section describes the simplest one, **local deployment**: each action runs
as a single script on your own machine. That is enough for a quick
`anchor-refe` run. Alaric can also split an action into chunks that run in
parallel, and deploy actions to a remote compute cluster, which an `anchor`
run needs. [documents/deployment.md](documents/deployment.md) describes these
deployers.

Work in the run directory, in the `alaric` environment. Each action is
deployed and run on its own:

```bash
conda activate alaric
alaric-deploy local frag2-anchor-raw
frag2-anchor-raw/run.sh
```

`alaric-deploy local` writes two scripts into the action directory, `run.sh`
and `check.sh`. `run.sh` first runs `check.sh`, which checks that the input
actions are done. It then builds the pool in `CACHE/`, links it as `results`,
and writes `result.txt`. An action with a `result.txt` is done.

If the result is already in the cache, from this run or from another run that
shares `CACHE/`, `alaric-deploy` writes `result.txt` at once, and there is
nothing to run. `alaric-result-check frag2-anchor-raw` then adds the `results`
link.

**Order.** An action can run only after the actions it reads, which are named
by `input`, `input1`, `input2` or `restrict_input` in its `alaric.yaml`.
`grep input */alaric.yaml` lists them. For example, the forward anchor gen
`gen-frag2-3` runs in this order:

1. `frag2-anchor-raw` and `frag3-anchor-raw` place fragments 2 and 3.
2. `frag2-anchor-scored` and `frag3-anchor-scored` filter them by energy.
3. `frag3-anchor-grown` grows fragment 3 from fragment 2. `frag3` keeps the
   poses that are in both pools of fragment 3.
4. `frag2-anchor-grown` grows fragment 2 back from `frag3`. `frag2` keeps the
   poses that are in both pools of fragment 2.

A shell loop deploys and runs them in that order, and skips the actions that
are done:

```bash
for action in frag2-anchor-raw frag3-anchor-raw frag2-anchor-scored \
    frag3-anchor-scored frag3-anchor-grown frag3 frag2-anchor-grown frag2; do
  [ -f $action/result.txt ] && continue
  alaric-deploy local $action || break
  [ -f $action/result.txt ] || $action/run.sh || break
done
```

If `run.sh` fails, fix the cause and run it again: it starts the action from
scratch. If you change an action's `alaric.yaml` after deploying it, `run.sh`
refuses to run until you deploy it again.

**Beyond local deployment.** All of the above runs every action as one
script on your own machine. For an `anchor` run, which takes about a day on 10
cluster nodes, use the other deployers in
[documents/deployment.md](documents/deployment.md):

- `alaric-deploy local-chunk` and `remote-chunk` split an action into chunks
  that can run in parallel, for example as separate cluster jobs;
- `alaric-deploy remote` and `remote-chunk` copy the scripts to a remote
  cluster, where you submit them. The results stay on the cluster until you
  download them (see [The whole picture](#the-whole-picture)).

The deployer only changes where and how a result is computed, not the result.
A result computed with one deployer is reused by all the others.

## 4. Calculate fragment RMSDs

> **Benchmarking mode.** For now, Alaric is much more convenient to use in
> benchmarking mode: when a reference structure of the RNA bound to the
> protein is available and can be used for RMSD calculations. Preparic
> currently assumes benchmarking mode. The reference is the RNA of the PDB
> entry you load in stage 1, saved as `DATA/reference.pdb`, and the fragment
> library conformations taken from that entry are excluded from the search.

`alaric-rmsd` calculates the RMSD of every pose in a pool to the same fragment
in the reference:

```bash
alaric-rmsd frag3
```

It takes the fragment number from the action name: `frag3`, `frag3-anchor-raw`
and `frag3-merged` are all fragment 3. It writes one RMSD per pose, in Å and
in pose order, next to the poses: `frag3/results/rmsd.txt`, or
`frag3/results/rmsd.npy` for a pool of more than 100,000 poses. The lowest
RMSD, and the number of poses below a cutoff, tell you whether the pool still
holds near-native poses:

```bash
sort -n frag3/results/rmsd.txt | head -1      # lowest RMSD
awk '$1 < 1' frag3/results/rmsd.txt | wc -l   # poses below 1 Å
python -c 'import numpy as np; r = np.load("frag3/results/rmsd.npy"); print(r.min(), (r < 1).sum())'
```

Comparing these numbers along a gen shows which step loses the near-native
poses.

## The whole picture

Steps 2 to 4 are not done once. You go through them again for each gen that
you add to a run. In full, the workflow is:

1. **Choose and copy** the action directories of the next gens into the run,
   with `alaric-copy` (step 2).
2. **`alaric-sigil`** computes the sigils and checks the run
   ([documents/sigils.md](documents/sigils.md)).
3. **Deploy** the action with `alaric-deploy` (step 3). Deploying runs
   `alaric-sigil` for the action and for everything it reads.
4. **`check.sh`** checks that the action's inputs are done and that their
   pools are present.
5. **Run** the action with `run.sh`, which runs `check.sh` first.
6. **Download** the result with `alaric-result-download`, if the action ran on
   a remote cluster ([documents/deployment.md](documents/deployment.md)).
7. **Calculate RMSDs** with `alaric-rmsd` (step 4).

**Only `check.sh` requires that the input pools are physically present**, on
the machine where the action runs. Copying, `alaric-sigil` and deploying need
only the action directories and `DATA/`, so they work before any of the
inputs has been computed. This leads to two pieces of advice:

- **Choose late.** You can, and probably should, delay choosing the next gens
  until you know which pools are smallest (see "Choosing the next gen" in
  step 2).
- **Deploy early.** Once you have chosen, you can, and probably should, copy
  and deploy all the actions of the new gens at once, including those whose
  inputs have not been computed yet. Deploying checks the new actions and
  finds the results that are already in the cache. Each action is then ready
  to run as soon as its inputs are done.

**The pool graph.** This is also where `alaric-pool-graph` comes in. The last
step of Alaric links compatible poses of consecutive fragments into chains,
each a model of the RNA. `alaric-pool-graph` prepares this. It reads the
run's actions and writes a small JSON file, the pool graph, which records how
the pools are connected and which pool represents each fragment.
`alaric-chain` reads the pool graph to count or build the chains.
[documents/pool-graph.md](documents/pool-graph.md) explains the pool graph and
counting, and [step 5](#5-build-chains) the rest.

Like deploying, `alaric-pool-graph` does not need the pools themselves, so you
can run it early, right after `alaric-sigil`. You can also run it again each
time you add gens, so that you can count or build chains for the fragments
that you already have.

## 5. Build chains

The last step links the poses of consecutive fragments into chains, one pose
per fragment. Each chain is a model of the RNA, or of the stretch of it that
the run covers. This step has three parts:

1. **`alaric-pool-graph`** makes the pool graph. This also determines the
   representative pool of each fragment, the pool that the chains take their
   poses from ([documents/pool-graph.md](documents/pool-graph.md)).
2. **Provenance copying**, for a remote run only. `alaric-chain` needs to know
   which pose each representative pose was grown from. That information is
   spread over all the intermediate pools, on the cluster. Provenance copying
   collects it into one small file per representative and downloads only
   that. It must come *between* `alaric-pool-graph`, which determines the
   representatives, and `alaric-chain`.
   [Provenance copying](documents/deployment.md#provenance-copying) in the
   deployment document explains how.
3. **`alaric-chain`** builds the chains.

### The main tools: `alaric-chain` and `alaric-chain-rmsd`

Count the chains first, then build them in a directory `chains/` in the run
directory:

```bash
alaric-chain pool-graph.json --count
alaric-chain pool-graph.json -o chains
```

`chains/` holds:

- one directory per fragment, named after its representative, with the
  distinct poses that occur in the chains;
- `chains.txt`: a header line with the pool names, then one chain per line,
  as pose numbers (from 1) in those directories;
- `chains.json`: what the other chain tools need to know about the columns.

By default, `alaric-chain` refuses to write more than 100 million chains.
`--max-chains` raises the limit, in millions.

`alaric-chain-rmsd` compares the chains with the reference structure:

```bash
alaric-chain-rmsd chains
```

It reads `DATA/reference.pdb` and `DATA/pdbcode.txt`, so `chains/` must be in
the run directory. It writes:

- `rmsd.txt` in each pose directory: the RMSD of each pose to its fragment in
  the reference, like `alaric-rmsd` for a pool;
- `chains/chain_rmsd.txt`: a header line, then one line per chain, in the
  order of `chains.txt`. The first column is the RMSD of the whole chain, the
  others the RMSDs of its fragments. The whole-chain RMSD is an approximation:
  for a nucleotide that two fragments share, it takes the mean of the two
  RMSDs of that nucleotide, rather than building the averaged coordinates.

```bash
tail -n +2 chains/chain_rmsd.txt | sort -n | head -1    # lowest chain RMSD
```

With `--pairwise OUT.npy`, `alaric-chain-rmsd` instead writes the RMSD between
every two chains, as a matrix, for at most 50,000 chains. This needs no
reference structure.

### Other chain tools

**`alaric-chain-focus`** looks at a part of the chains:

```bash
alaric-chain-focus chains frag3 frag4 frag5-merged -o chains-3-5
```

The new directory keeps only the named columns, with duplicate lines removed.
Many chains share the same poses for fragments 3 to 5, so `chains-3-5` has
far fewer lines than `chains`. `chains-3-5/chain-provenance.txt` gives, for
each chain in `chains/`, the number of its part in `chains-3-5/chains.txt`,
counting from 1 after the header line. The pose directories are links to
those in `chains/`.

These partial chains differ from the preliminary chains of
[documents/pool-graph.md](documents/pool-graph.md), which are built over only
some of the fragments, with `--target` or `--exclude`. In a preliminary chain,
the fragments that were left out constrain nothing, and many preliminary
chains will not survive once those fragments are added. A focused chain is
part of at least one complete chain, so all fragments have constrained it.
Use preliminary chains to follow a run while it grows, and focused chains to
look at a stretch of the final models.

**`alaric-chain-coordinates`** and **`alaric-write-pdb`** turn chains into
structures to look at:

```bash
alaric-chain-coordinates chains --chain-range 1 100
alaric-write-pdb chains/chains.ppdb.npy chains-1-100.pdb
```

`alaric-chain-coordinates` builds the atom coordinates of each chain, and
averages the two copies of each nucleotide that consecutive fragments share.
It writes them to `chains/chains.ppdb.npy`, or to the file given with `-o`.
All chains go into one array, which can be very large, so use
`--chain-range START END` to choose chains. The fragments must be
consecutive, so a focused directory works only if its fragments are.
`alaric-write-pdb` writes the array as a PDB file with one model per chain.

**`write_pdb_haddock.py`** writes the chains as a PDB file for HADDOCK3,
together with the protein:

```bash
python $ALARIC_DIR/write_pdb_haddock.py chains/chains.ppdb.npy DATA/range-2-aa.pdb chains-haddock.pdb
```

Each model holds the protein, with chain ID A, and the RNA of one chain, with
chain ID B. The RNA residues get one-letter names, and the phosphate of the
first nucleotide is removed. Give the PDB file of the protein ranges that the chains were built
against: one range in unfixed mode, a file with all of them in fixed mode.

## Further documentation

- [INSTALLATION.md](INSTALLATION.md): installation and configuration.
- [documents/sigils.md](documents/sigils.md): sigils and the `alaric-sigil`
  command, the shared cache, and why many runs from one export are cheap.
- [documents/deployment.md](documents/deployment.md): chunked and remote
  deployment, and copying provenance from the cluster for chain building.
- [documents/pool-graph.md](documents/pool-graph.md): the pool graph, and
  counting chains.
- [documents/middle-level.txt](documents/middle-level.txt): every action and
  its `alaric.yaml` parameters, and the `alaric-sigil`, `alaric-deploy` and
  `alaric-result-*` commands.
- [plans/alaric-project-layout.md](plans/alaric-project-layout.md): the
  project layout, the cache and the downstream commands.
- [documents/gaussian-filter.md](documents/gaussian-filter.md): the Gaussian
  stacking filters.
- [plans/export-gens-plan.md](plans/export-gens-plan.md): the full rules of
  the gen catalogue.
