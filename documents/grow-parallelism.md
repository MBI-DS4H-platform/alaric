# Grow Parallelism: Why It Runs Single-Core

Status: findings only. No benchmark has been run; the measurement plan at the end is
deliberately deferred until realistic pool sizes are available.

## Question

Is multiprocessing/forking disabled in `grow.py` in order to leave the cores available for
multi-threaded BLAS?

## Answer

No. BLAS threading is disabled too, explicitly and for its own stated reason. Forking is not
disabled at all — it is fully implemented and merely defaulted off, for a third reason
(memory). The net effect is that `grow` runs on exactly one core by default, but that is the
sum of two independent decisions rather than a trade-off between them.

## The three facts

### 1. BLAS threading is pinned to 1 before numpy is imported

`alaric/grow.py:13-17`:

```python
# Grow uses many tiny k=9 matrix products. Multi-threaded BLAS adds more
# scheduling overhead than useful parallelism here; use process-level parallelism
# via --nprocs unless the caller explicitly overrides these.
for _thread_var in ("OPENBLAS_NUM_THREADS", "OMP_NUM_THREADS", "MKL_NUM_THREADS"):
    os.environ.setdefault(_thread_var, "1")
```

Notes:

- It runs *before* `import numpy`, which is required — these variables are read by the BLAS
  library at load time, so setting them afterwards has no effect.
- It is `setdefault`, not assignment, so an explicit `OMP_NUM_THREADS=8` in the caller's
  environment still wins. That is the "unless the caller explicitly overrides" clause.
- The stated rationale is that the products are `k=9` (i.e. 3x3 rotations), where the
  threading barrier costs more than the arithmetic returns.

### 2. The fork pool exists and works

`alaric/grow.py:815-842`, in `run_pooled_trace_grow` — the kernel shared by `grow` and
`anchor_refe`:

- target conformers are sliced round-robin (`target_conformers[i::nprocs]`);
- workers are started through an `mp.get_context("fork")` pool with an initializer carrying
  a `_GrowWorkerConfig`, so the source caches and target library are inherited copy-on-write
  rather than pickled to each child;
- each worker writes its own `unorganized-<pid>/` subdir into the shared output dir;
- there is an explicit guard raising `RuntimeError` where the `fork` start method is
  unavailable (i.e. not Linux/macOS).

So this is not dead or vestigial code. Nothing is disabled — only defaulted off.

### 3. The `--nprocs` default of 1 is about memory

`alaric/grow.py:234-243` sets `--nprocs` default `1`. The reason is the pose cache:
`--cache-size` defaults to 50,000,000 weighted poses (`alaric/grow.py:190-198`) and is passed
to every worker **unchanged** — one `_GrowWorkerConfig` holds a single `cache_size`
(`alaric/grow.py:828`), and each worker forwards `cache_size=cfg.cache_size` into its own
`_pooled_trace_grow` call (`alaric/grow.py:763`). It is never divided by `nprocs`.

Therefore `--nprocs N` multiplies the peak pose-cache footprint by N. That is why it cannot
be a silent default: raising it changes the memory profile of the job, not just its core
count. Confirmed as intentional by Sjoerd on 2026-07-30.

The intended opt-in path is the commented knob already shipped in the templates
(`alaric/middle/templates/*-chunk/grow.sh`):

```bash
grow_opts=(
#  --nprocs 8
#  --cache-size 100000000
)
```

`anchor_refe.py` shares the same kernel and the same default.

## Git history

Both the BLAS pin and the `--nprocs` default arrived in the same commit, `0a91362 rename and
reorganize`. `git log -S` on either finds no earlier or later change, so there is no
incremental history explaining them beyond the inline comment quoted above.

## Consequences worth noting

1. **The header comment and the default contradict each other in effect.** The comment reads
   as though `--nprocs` is the live parallelism path ("use process-level parallelism via
   `--nprocs`"), but with the default at 1 *and* BLAS pinned at 1, a `grow` job on an
   `sbatch -c 8` allocation uses exactly one core and leaves seven idle.

2. **Chunking is currently the only way `grow` uses more than one core on a node.** Splitting
   into separate `chunkN.sh` jobs is the only multi-core path that is on by default. This
   interacts directly with the pose-pool placement question: the parallel-submission mode is
   precisely the one that forces the unorganized pool onto the shared filesystem, so `grow`
   gets the worst of both — single-core per job, shared-FS pool — unless someone uncomments
   `--nprocs`.

3. **`grow` is excluded from the node-local pool redirection** implemented for
   `anchor`/`anchor-test`/`anchor-refe` (see `middle-level-plan.md`), because it emits
   provenance sidecars next to each shard and those are written non-atomically
   (`poses.py:869`, after the shard itself has been atomically renamed into place). Its crash
   and restart semantics need their own analysis before the same treatment is applied.

## Open question, and how to settle it

The `k=9` rationale for pinning BLAS is plausible but **unverified in this repo**. Whether
multi-threaded BLAS is genuinely pure overhead depends on the shape of the calls: if the
products are issued per-conformer in small batches, the comment is certainly right; if they
are tall-skinny `(N,3) @ (3,3)` with large N, threading over rows is not automatically
useless. Nobody has measured it.

Deferred measurement plan, to run against realistic pool sizes:

- wall time and peak RSS for `--nprocs` in {1, 2, 4, 8}, at the default `--cache-size`;
- the same sweep with `--cache-size` divided by `nprocs`, to test whether a fixed total
  memory budget makes a higher default safe;
- `--nprocs 1` with `OMP_NUM_THREADS` in {1, 2, 4, 8}, to test the BLAS claim directly;
- instrumentation of the actual matmul shapes reaching BLAS, which decides the question more
  cheaply than the sweep does.

Until then, leave both defaults alone.
