# Deployment

The README describes the simplest case: deploying an action on your own
machine as a single script, with `alaric-deploy local`. This document covers
the other two possibilities: splitting an action into chunks, and running it
on a remote compute cluster. Its last section, on provenance copying, is the
remote part of chain building.

## The four deployers

```bash
alaric-deploy local                [ACTION]
alaric-deploy local-chunk  NCHUNKS [ACTION]
alaric-deploy remote               [ACTION]
alaric-deploy remote-chunk NCHUNKS [ACTION]
```

`ACTION` is an action directory of the run. It defaults to the current
directory.

| Deployer | Where the scripts run | Scripts |
|---|---|---|
| `local` | your machine | `run.sh`, `check.sh` |
| `local-chunk` | your machine | `chunk1.sh` … `chunkN.sh`, `organize.sh`, `run.sh`, `check.sh` |
| `remote` | the cluster | `run.sh`, `check.sh` |
| `remote-chunk` | the cluster | `chunk1.sh` … `chunkN.sh`, `organize.sh`, `run.sh`, `check.sh` |

All four deployers have these in common:

- They compute the sigil of the action, and of everything upstream of it.
  `alaric-deploy` refuses an action that already has a `result.txt`.
- If the local cache already has the result (`CACHE/checksum/<SIGIL>`), they
  write `result.txt` and no scripts. This holds for the remote deployers too.
- The deployer and the number of chunks are not part of the sigil. They change
  how a result is computed, not what it is. A result computed with one
  deployer is reused by all the others.
- Deploying again replaces the scripts.

## Chunks

A chunked deployment splits one action into independent pieces. These actions
can be chunked:

| Action | Split by |
|---|---|
| `anchor`, `anchor-refe`, `anchor-test` | the conformations of the fragment library |
| `grow`, `grow-test` | the poses of the input pool |
| `energy-filter`, `score` | the poses of the input pool |

Any other action gets a plain `run.sh` under a chunk deployer.

```bash
alaric-deploy local-chunk 4 frag2-anchor-raw
```

This writes:

- `chunk1.sh` … `chunk4.sh`: each computes its part of the action. The chunks
  are independent. They can run in any order, and at the same time.
- `organize.sh`: run this after all chunks have finished. It combines their
  output into the final result, and records its checksum.
- `run.sh`: runs all chunks one after the other, and then `organize.sh`.
- `check.sh`: as for an unchunked action. Every chunk runs it first.

With `local-chunk`, `run.sh` is the simplest way to run the action, so it
mostly serves to try out a chunked deployment on your own machine. Chunking
pays off on a cluster, with `remote-chunk`, where each chunk can be a job of
its own.

**A chunk cannot be resumed.** A chunk that is run again after it crashed adds
its poses a second time. If a chunk fails, start the action over:

- `run.sh` does this by itself: it removes the partial result before it
  starts.
- If you run the chunks yourself, first remove the partial output:
  - locally, `CACHE/results/<SIGIL>`, plus `CACHE/results/<SIGIL>-CHUNKS` for an
    energy filter;
  - on the cluster, `$ALARIC_REMOTE_RESULT_DIR/<SIGIL>.partial`, plus
    `$ALARIC_REMOTE_RESULT_DIR/<SIGIL>-CHUNKS` for an energy filter.

  Then run all chunks again.

## Remote deployment

### Setting up

1. Install Alaric on the cluster, as described in
   [INSTALLATION.md](../INSTALLATION.md) §6. Create `~/.alaric/fraglib.yaml`
   there as well (§8), and build the Gaussian stacking-filter caches in the
   cluster's checkout (§9), because `anchor` reads them.
2. `alaric-deploy` and the `alaric-result-*` commands call `ssh`, `scp` and
   `rsync` to the cluster, several times per action. Set up a login without a
   password prompt, for example with an SSH key.
3. Set these variables on your own machine, in your shell startup file:

   | Variable | Meaning |
   |---|---|
   | `ALARIC_REMOTE_HOST` | the SSH host name of the cluster |
   | `ALARIC_REMOTE_DEPLOYMENT_DIR` | a cluster directory for the deployed scripts |
   | `ALARIC_REMOTE_ALARIC_DIR` | the `alaric/` package directory of the cluster's checkout |
   | `ALARIC_REMOTE_RESULT_DIR` | a cluster directory for the results |

   Two more are optional:

   | Variable | Meaning |
   |---|---|
   | `ALARIC_PROJECT` | the folder in which the deployed actions are listed by name: a name under the deployment directory, or an absolute path. The default is `PROJECT`. |
   | `ALARIC_REMOTE_SCRATCH_DIR` | node-local scratch space on the cluster. The default is `$TMPDIR`, else `/tmp`. |

   The remote variables are written into the scripts at deployment, so you
   don't need to set them on the cluster.

### Deploying

```bash
alaric-deploy remote frag2-anchor-raw
```

This copies to the cluster:

- **The `DATA/` files the action reads.** They go to
  `$ALARIC_REMOTE_DEPLOYMENT_DIR/DATA/`, stored by checksum. That store is
  shared by all runs and projects, so each file is uploaded only once.
- **The scripts.** They go to `$ALARIC_REMOTE_DEPLOYMENT_DIR/SIGIL/<SIGIL>/`,
  together with the `DATA/` files they need and a `results` link to
  `$ALARIC_REMOTE_RESULT_DIR/<SIGIL>`.
- **A link by action name**, `$ALARIC_REMOTE_DEPLOYMENT_DIR/$ALARIC_PROJECT/<action>`,
  that points to the script directory. A later deployment of an action with
  the same name, from any run, replaces the link. To keep the runs apart, set
  `ALARIC_PROJECT` to a name of its own for each run.

A remote result is found by its sigil, whichever run or project computed it.
Deployment only looks in the local cache, though. If the result may already
be on the cluster, run `alaric-result-check frag2-anchor-raw` before
deploying. If the result is there, this records it locally, and deployment
then finds it. If it is not, the command fails, and you deploy as usual.

### Running

On the cluster, in the `alaric` environment, go to the action's directory
under the deployment directory and submit `run.sh`. For example, with
`ALARIC_REMOTE_DEPLOYMENT_DIR=~/alaric-deployment`:

```bash
cd ~/alaric-deployment/PROJECT/frag2-anchor-raw
sbatch run.sh          # or: bash run.sh
```

The scripts call `python`, so the `alaric` environment must be active when
they run. A SLURM job normally inherits the environment of the shell that
submits it. Alaric's worker processes follow the job's CPU allocation, for
example `sbatch -c 16`.

`check.sh` checks that the results of the input actions exist in
`$ALARIC_REMOTE_RESULT_DIR`. So every input must have been computed on the
cluster, or uploaded to it (see below). The result is built under
`$ALARIC_REMOTE_RESULT_DIR/<SIGIL>.partial`, and renamed to
`$ALARIC_REMOTE_RESULT_DIR/<SIGIL>` when it is complete.

With `remote-chunk`, submit every chunk as a job of its own, and `organize.sh`
once they have all succeeded. With SLURM, for example:

```bash
jobs=$(for i in 1 2 3 4; do sbatch --parsable chunk$i.sh; done | paste -sd:)
sbatch --dependency=afterok:$jobs organize.sh
```

`alaric-deploy remote-chunk` warns that this may slow down the shared file
system. Chunks that run as separate jobs may run on different nodes, so they
must write their intermediate output to the shared result directory. `run.sh`
instead runs all chunks in one job, and keeps the intermediate output on
node-local scratch.

### Back on your own machine

| Command | What it does |
|---|---|
| `alaric-result-check ACTION` | Reads the result's checksum from the cluster and writes `result.txt` and `CACHE/checksum/<SIGIL>`. The data stays on the cluster. |
| `alaric-result-download ACTION` | Downloads the result into `CACHE/results/<SIGIL>` and links it as `results`. |
| `alaric-result-upload ACTION` | Uploads a local result to the cluster, for example an input that you computed locally. |
| `alaric-result-clean [--all] ACTION` | Removes the local result data. With `--all`, also `result.txt` and the cached checksum. |

After `alaric-result-check`, the action has a `result.txt` and counts as done
in the run. Downstream actions that run on the cluster need nothing more.
Before you run a downstream action locally, or use a command that reads the
poses, such as `alaric-count-poses` or `alaric-rmsd`, run
`alaric-result-download`.

## Provenance copying

`alaric-chain` builds chains from the representative pools that
`alaric-pool-graph` chooses, one per fragment (see
[pool-graph.md](pool-graph.md)). On your machine, it needs the poses of each
representative, and the provenance files that connect the representatives:
which pose each pose was grown or filtered from. Those files belong to all the
intermediate pools between the representatives. In a remote run, those pools
are on the cluster, and downloading them in full would cost far too much.

Provenance copying avoids this. On the cluster, the provenance of each
representative is traced back through its intermediate pools to the pools it
was grown from, and stored as one small file next to the representative's
poses. Only these files are then downloaded. Provenance copying therefore
comes **after `alaric-pool-graph`**, which determines the representatives, and
**before `alaric-chain`**:

1. **Make the pool graph**, as in [pool-graph.md](pool-graph.md):

   ```bash
   alaric-pool-graph -o pool-graph.json
   ```

2. **Write the scripts, on your machine.** In the run directory, run
   `alaric-propagate-provenance` for each representative, with the deployer
   `remote`:

   ```bash
   alaric-propagate-provenance pool-graph.json frag5-merged remote
   ```

   This writes the script `propagate-provenance-frag5-merged.sh` in the
   current directory, and also copies it to the cluster, into
   `$ALARIC_REMOTE_DEPLOYMENT_DIR/$ALARIC_PROJECT/`. Without a representative,
   the command lists the representatives, so this does them all:

   ```bash
   for rep in $(alaric-propagate-provenance pool-graph.json | tail -n +2); do
     alaric-propagate-provenance pool-graph.json $rep remote
   done
   ```

   A representative that was not grown from anything, such as a pool that is
   only anchored, has nothing to copy. The command says so, and writes no
   script.

3. **Run the scripts, on the cluster**, in the `alaric` environment, once all
   the pools of the run have been computed there:

   ```bash
   cd ~/alaric-deployment/PROJECT
   for script in propagate-provenance-*.sh; do bash $script; done
   ```

   Each script writes, into the representative's result in
   `$ALARIC_REMOTE_RESULT_DIR`, one file per fragment that the representative
   was grown from: `prop-frag<N>-provenance.npy.zst` or
   `prop-frag<N>-map.npy.zst`. `frag5-merged`, grown from both fragment 4 and
   fragment 6, gets two.

4. **Download the files, on your machine**, for each representative:

   ```bash
   alaric-provenance-download frag5-merged
   ```

   This downloads the provenance files of the result, and none of its poses.
   If the representative's poses are already on your machine, from
   `alaric-result-download`, it adds the files to them.

`alaric-chain` also needs the poses of every representative. Download them
with `alaric-result-download` *before* `alaric-provenance-download`:
`alaric-result-download` refuses a result that is already on your machine,
even one that holds only provenance files. A representative that you download
in full after its script has run has its provenance files already.

`alaric-chain` then reads the copied provenance and does not need the
intermediate pools. If it
still reports a missing provenance file of another pool, run
`alaric-provenance-download` for that pool as well.

If a representative has two routes from the same fragment,
`alaric-propagate-provenance` stops and lists them. Choose one with
`--branch POOL`, which names the merge input to follow.

A local run needs no provenance copying: the provenance files of all its pools
are already on your machine.
