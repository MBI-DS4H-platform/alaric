# Middle-Level Command Toolchain Handoff

## Summary
Implement the middle level as a strict `alaric.yaml` action graph with deterministic `auto` resolution, content-addressed sigils, plain-bash deployers, and checksum-based local/remote result lifecycle. Use Seamless only for canonical JSON and byte/deepfolder checksums. No `seamless-run`, no Slurm, no automatizer logic.

Deferred but explicitly out of scope for this milestone: sigils do **not** yet
include backend code, helper code, or rendered template checksums. This means a
code/template change can require manual cache invalidation until code-dependency
identity is added in a later milestone.

Add `alaric/middle/` plus console scripts:
- `alaric-sigil [--force|--delete] [project-root]`
- `alaric-deploy <local|local-chunk|remote|remote-chunk> [nchunks] [action-dir]`
- `alaric-result-clean [--all] [action-dir]`
- `alaric-result-check [action-dir]`
- `alaric-result-download [action-dir]`
- `alaric-result-upload [action-dir]`

Development and verification commands must run in the `alaric` conda environment
(`conda run -n alaric ...` or an activated `alaric` env). The base shell may lack
runtime dependencies such as SciPy, so direct `python ...` invocations are not a
valid handoff check.

## Schema And Auto-Resolution
Project layout:
- Hand-authored: `DATA/`, action dirs, optional `templates/<deployer>/<action>.sh`
- Generated lazily: `CACHE/parameters`, `CACHE/checksum`, `CACHE/results`

`alaric.yaml` is strict: unknown keys error. `action:` is canonical; `type:` is accepted only if `action:` is absent.

Supported action schemas:
- `anchor`: `fragment`, `sequence`, `exclude`, `protein`, `resid`, `nucleotide`, optional `dihedral`, `angle`, `margin`
- `grow`: `input`, `fragment`, `sequence`, `exclude`, `direction`, `crmsd`, `ovrmsd`
- `score`: `input`, `sequence`, `exclude`, `protein`, optional `nb_kernel`
- `rmsd`: `input`, `fragment`, `exclude`, optional `reference`
- `score_add`: `score_input1`, `score_input2`
- `mask`: `input`, `score_input`, `threshold`
- `filter`: either `input + score_input + threshold` or `input + mask_input`
- `identity`: `input1`, `input2`

Auto-resolution rules, field by field:
- `fragment:auto`:
  Parse from action-dir name `fragN-...`. Applies to `anchor`, `grow`, `rmsd`. Fail if the dir name does not encode `fragN`.
- `sequence:auto` in `anchor` and `grow`:
  1. `DATA/constraints.json["fragN"]["sequence"]`
  2. `DATA/sequence.txt` if present and indexable by fragment number
  3. `DATA/reference.pdb` by extracting residues `N,N+1`
  Fail if none works.
- `sequence:auto` in `score`:
  Resolve from the referenced input action’s final fragment sequence, not directly from `constraints.json`.
- `exclude:auto`:
  1. `DATA/pdbcode.txt`
  2. `DATA/constraints.json["pdb_code"]`
  Normalize to a one-element list of lowercase PDB codes.
- `protein:auto`:
  `auto` → `DATA/protein-aa.pdb` (spec: auto maps to the literal `protein-aa.pdb`,
  middle-level.txt:26). Only `auto` may resolve to/fall back to `protein-aa.pdb`.
  An explicit `protein: NAME` reads exactly `DATA/NAME-aa.pdb` and must fail if that
  file is absent. It must **not** fall back to `protein-aa.pdb` or `<exclude>-aa.pdb`.
  `protein` and `exclude` differ in general — e.g. `1b7f_dom2` vs `1b7f` in the test project.
- `direction:auto`:
  Parse from action-dir suffix: `-fwd` => `forward`, `-bwd` => `backward`. Fail otherwise.
  Explicit values normalize as `fwd`/`forward` => `forward`, `bwd`/`backward` => `backward`;
  any other value is a schema error.
- `crmsd:auto` and `ovrmsd:auto`:
  For `grow`, resolve source fragment from `input`, target fragment from this action, require adjacency, then read the matching `DATA/constraints.json["pairs"]` entry for the ordered pair `{down: lower fragment, up: higher fragment}`.
- `dihedral:auto`:
  `DATA/anchor.yaml["dihedral"]`
- `angle:auto`:
  `DATA/anchor.yaml["angle"]`
- `threshold:auto`:
  Allowed only for `filter` and `mask` on the score route.
  1. Resolve the referenced `score_input`
  2. If it is a `score` action, use `DATA/constraints.json["fragN"]["scores"][protein]`, where `fragN` comes from the scored input fragment and `protein` is the resolved score protein
  3. If it is a `rmsd` action, fail: no auto-threshold source is defined
  4. If it is a `score_add` action, fail by explicit policy
- `reference` in `rmsd` is not an `auto` field:
  default to `DATA/reference.pdb` when omitted; allow explicit override

Non-auto defaults:
- `margin` default `0.5`
- `nb_kernel` default `compiled`
- `rmsd.reference` default `reference.pdb`

Cross-action validation:
- Pose outputs feed only `input`, `input1`, `input2`
- Scoring outputs feed only `score_input*`
- Mask outputs feed only `mask_input`
- `score_add` inputs must score the same underlying pose input
- `mask.score_input.input == mask.input`
- `filter` route input must equal `filter.input`
- `identity` inputs must resolve to the same fragment

## Implementation Changes
Add `alaric/middle/`:
- `project.py`: discover project, action dirs, `DATA`, lazy `CACHE` accessors
- `schema.py`: strict schema, normalization, dependency-field declarations, output kinds
- `resolve.py`: implement the exact auto rules above
- `graph.py`: build DAG, resolve refs, validate dependency kinds, detect cycles
- `checksum.py`: Seamless JSON/byte/deepfolder checksums for sigils and result sidecars;
  do not use `tests/1b7f/arc-checksum.sh` for middle-level result checksums
- `backend.py`: map resolved actions to template context and load-bearing param sets
- `sigil.py`, `deploy.py`, `result.py`

Backends:
- `anchor`: `anchor.py` then `organize.py --compress`
- `grow`: `grow.py` then `organize.py --compress`
- `score`: `${ALARIC_DIR}/score.sh`
- `rmsd`: `alaric/rmsd.py` — **move `util/rmsd.py` → `alaric/rmsd.py` via `git mv`** (it is
  an action backend alongside `anchor.py`/`grow.py`). No import edits needed: every importer
  already has both `alaric/` and `util/` on `sys.path` (`alaric/write-poses-pdb.py`,
  `util/pairwise-rmsd.py`, and the tests `test_rmsd_refe_best_fit.py` /
  `test_write_poses_pdb.py` / `test_pairwise_rmsd.py`). Optional: drop `rmsd.py`'s now-
  redundant `_ALARIC_DIR` sys.path shim.
- `score_add`: new `alaric/score_add.py`
- `mask`: new `alaric/mask.py`
- `filter`: score+threshold route uses `filter-poses.py`; mask route uses `select-poses.py`
- `identity`: `identity_filter.py`

Hard repo/path constraints:
- Generated scripts must invoke `${ALARIC_DIR}/score.sh`
- `ALARIC_DIR` means the repo’s `alaric/` directory, not repo root
- This relies on the existing `alaric/util -> ../util` and `alaric/attract-jax -> ../attract-jax` symlinks locally and remotely

Template contract:
- Templates are text templates rendered by `deploy.py` into concrete shell scripts
- Templates are per deployer and per action: `templates/<deployer>/<action>.sh`
- `resolve.py` computes concrete values; `backend.py` prepares the template context; templates render the shell invocation
- Templates receive resolved action-specific business fields and helper paths, not just monolithic command strings
- `check.sh` is generated directly by Python, not templated

Template context responsibilities:
- action templates render concrete CLI invocations like `grow.py --ov-rmsd {{ ovrmsd }}`
- templates may format helper values like `exclude_args`, `input_result_path`, `output_path`, `score_output_path`, `check_sh_path`
- templates must not perform semantic resolution such as reading `constraints.json`, deriving `fragment`, or deciding whether `threshold:auto` is legal

## Chunk Deployers
Chunking rules are action-specific and must be encoded explicitly in chunk templates.

- `anchor` chunking:
  chunk by conformer range using `anchor.py --conformer-range FIRST LAST`
- `grow` chunking:
  chunk by source pose range using `grow.py --pose-range FIRST LAST`
- `score` chunking:
  chunk by source pose range using `score.sh POSE_DIR POSE_START POSE_END SEQUENCE RECEPTOR NB_KERNEL OUTPUT`
- `rmsd`, `score_add`, `mask`, `filter`, `identity`:
  no chunk deployer support in this milestone unless later added explicitly

Chunk-template behavior:
- `local-chunk/<action>.sh` and `remote-chunk/<action>.sh` are smart templates
- They must discover chunk boundaries at runtime from resolved inputs, not from hardcoded numbers
- Chunk scripts must use inclusive 1-based ranges because that matches the current CLIs

Runtime discovery rules:
- `anchor`:
  determine conformer count from the resolved fragment library after applying exclusion, without loading rotaconformers
- `grow`:
  determine pose count from the resolved input pose dir via `PoseReader.get_nposes(...)` or equivalent
- `score`:
  determine pose count from the resolved input pose dir via `PoseReader.get_nposes(...)` or equivalent

Chunk outputs:
- `anchor` and `grow` chunks all write unorganized shards into the same output dir; after all chunks complete, run `organize.py --compress`
- `score` chunks each write a chunk-local score file; after all chunks complete, concatenate the chunk score arrays in pose-range order into final `score.npy`
- The plan must treat score-chunk concatenation as part of the chunk deployer contract, not an optional convenience

Chunk template tests must verify:
- `anchor` rendered scripts compute library size dynamically and do not contain baked-in conformer counts
- `grow` rendered scripts compute pose count dynamically and chunk by pose range
- `score` rendered scripts compute pose count dynamically and chunk by pose range
- `anchor`/`grow` rendered scripts include organize step
- `score` rendered scripts include concatenation/finalization step

## Deploy/Result Transport And Invariants
Deploy/result transport:
- Use `scp` for deploy payloads: scripts and resolved `DATA` file params
- Use `rsync -a --partial` for bulky result upload/download
- Transfer results into a partial result path, write the expected checksum sidecar(s),
  then atomically rename the result path and sidecar(s) into their final names
- A remote result is complete iff its expected checksum sidecar(s) exist and match the
  canonical checksum(s); local completion additionally copies the final sidecar checksum
  into `CACHE/checksum/<SIGIL>`.

### Result checksum sidecars
Result data is always accompanied by Seamless-style checksum sidecars. The legacy
`tests/1b7f/arc-checksum.sh` tool is **not** used for middle-level result checksums.

- Array result files:
  - `score.npy` has sidecar `score.npy.CHECKSUM`
  - `score.npy.zst` has sidecar `score.npy.CHECKSUM`
  - The checksum content is identical for compressed and uncompressed forms: checksum
    the uncompressed NumPy bytes, not the compressed byte stream.
  - `mask` writes `mask.npy.zst` (a mask is one byte per pose, or an index array).
    `score` does not compress: it is consumed by memory-mapped streaming.
  - The generated scripts always reference the logical name (`mask.npy`); the reader
    (`select-poses.py`) and the sidecar step (`result_sidecar.py array`) resolve the
    compressed form, so `backend.result_file()` stays compression-agnostic.
- Pose result dirs:
  - A pose dir `posedir/` has sidecars `posedir.INDEX` and `posedir.CHECKSUM`
  - `posedir.INDEX` is Seamless JSON containing each organized `poses-*.arc` member
    and its zstd-transparent file checksum.
  - `posedir.CHECKSUM` is the checksum of `posedir.INDEX`.
  - Compression is transparent: `poses-X.arc` and `poses-X.arc.zst` contribute the
    same member checksum when their uncompressed arc content is the same.
  - The same holds for the per-pose index arrays inside a pose dir (`provenance.npy`,
    `map-1.npy`/`map-2.npy`, `order-array.npy`): they are written `.npy.zst`, the INDEX
    entry keeps the logical `.npy` name, and the member checksum is over the
    uncompressed NumPy bytes. Compressing them therefore leaves the result checksum
    (and every cached result) unchanged. Readers resolve either form
    (`alaric/npy_io.py:find_npy`).
- Local deployment:
  - After producing the result and sidecar(s), the local run script copies the final
    sidecar checksum (`score.npy.CHECKSUM` or `posedir.CHECKSUM`) into
    `CACHE/checksum/<SIGIL>`.
- Remote deployment:
  - Results are first produced under partial result names with their sidecars, then
    renamed into final result names.
  - For array outputs, the final payload lives under `<RESULT_DIR>/<SIGIL>/`; for pose-dir
    outputs, the final pose dir is `<RESULT_DIR>/<SIGIL>/` and its adjacent sidecars are
    `<RESULT_DIR>/<SIGIL>.INDEX` and `<RESULT_DIR>/<SIGIL>.CHECKSUM`.

### HPC / NFS-friendly remote execution (REQUIRED)
**Scale:** a single pose dir can hold **tens of gigaposes** (~10^10–10^11 poses). A pose is
**~1.5 bytes compressed** (`.arc.zst`; compression is always on) and **~6 bytes
uncompressed** (in memory and on disk as raw `.arc`). So a dir is **~tens to a few hundred GB
on disk compressed**, but **~4× larger when decompressed** (hundreds of GB to >1 TB), spread
over potentially **millions of shard files**. Naively writing/reading that on a shared
NFS/Lustre filesystem is catastrophic (metadata storms, partition exhaustion, slow random
I/O); and any node-local scratch sized for the data must account for the **~4× decompression
expansion**.

**Critical constraint — a pool SHARED BETWEEN JOBS must live on the shared FS.** Each
`chunkN.sh` is an independent job (e.g. one `sbatch` per chunk), so the sibling chunks and
the organize job run on **different nodes**. Node-local `ALARIC_REMOTE_SCRATCH_DIR` is
therefore **not visible** across them — it cannot hold a pool they all have to reach. Such a
pool is written to the shared `ALARIC_REMOTE_RESULT_DIR/<SIGIL>.partial/` and atomically
renamed to `<SIGIL>/` once organized.

**A script that owns the whole pool itself does NOT put it on the shared FS.** Whenever one
script produces every shard and organizes them, the unorganized pool never has to be visible
to another node, so it goes to node-local scratch and only the organized result crosses onto
the shared filesystem. That covers two cases:

* the `remote` deployer's `run.sh` (one process, one node), and
* the `remote-chunk` deployer's `run.sh`, which runs `chunk1..N` then `organize.sh` serially
  in one job — chunking there is for bounding wall time, not for parallelism.

The redirection is expressed as an env var (`ALARIC_UNORGANIZED_DIR` for pose producers,
`ALARIC_SCORE_CHUNKS_DIR` for `score`'s per-chunk `score.npy` dirs), *not* baked into the
chunk scripts, because `chunkN.sh` and `organize.sh` are the same files in both modes: they
bind `POSE_POOL="${ALARIC_UNORGANIZED_DIR:-<SIGIL>.partial}"`, so an independently submitted
chunk still defaults to the shared FS. `run.sh` sets the variable to a **`mktemp -d`**
directory under `ALARIC_REMOTE_SCRATCH_DIR` and removes it on an EXIT trap.

`mktemp` rather than a deterministic path is a correctness requirement, not tidiness. Shard
publication is atomic (`poses.py:write_arc_file` writes a `.tmp` and renames it into place,
and the `.tmp` droppings do not match `discover_unorganized`'s globs), so a crashed run
leaves behind *valid* shards. Neither `run.sh` nor `chunkN.sh` has resume logic, so a re-run
would write a second copy into a fresh `unorganized-PID/` and organize would fold both in — a
silently doubled result that still checksums self-consistently. A name that can never be
reused removes that, and also stops two concurrent submissions of the same sigil from sharing
a pool. Adding resume later means per-chunk done-markers, which is safe *because* publication
is atomic. Cleanup is best-effort: SIGKILL cannot be trapped, so a hard-killed job leaks its
pool — harmless, since the name is unique.

`grow`/`grow-test` are **excluded** from this redirection for now: they emit provenance
sidecars next to each shard, written non-atomically, so their crash semantics need their own
analysis. They keep the shared-FS pool in every mode.

Remote strategy (anchor/grow producers):
1. **Each `chunkN.sh` writes unorganized shards to `$POSE_POOL`**, which defaults to the
   shared `$ALARIC_REMOTE_RESULT_DIR/<SIGIL>.partial/` (`--output`). Different chunks run as
   different processes/nodes; `--unorganized-subdirs` gives each a `unorganized-PID/` subdir
   so there are no collisions and no single directory with millions of entries (NFS metadata
   killer). `--unorganized-subdirs` is active by default.
2. NFS-friendliness is tuned via **commented, non-load-bearing knobs** in the template
   (`--cache-size` ↑ → fewer/larger shards; `--nprocs`; `--poselock`).
3. **`organize.sh`** organizes `$POSE_POOL` in place, writes the sidecars there, and — if the
   pool was redirected — moves it into `<SIGIL>.partial/`; the result is then atomically
   renamed to `<SIGIL>/` (+ `.INDEX`/`.CHECKSUM` sidecars). Computing the sidecar in the pool
   means hashing the organized poses reads local disk instead of pulling the whole result
   back over the shared FS.
4. **The organize staging knobs depend on where the pool is**, so `organize.sh` picks them at
   runtime. Pool on the shared FS: `--local-tempdir` (copy+decompress shards into node-local
   scratch, read-once) and `--local-stagedir` (write organized output to scratch, then move
   in — non-atomic; crash loses both) are **active**, with `export
   TMPDIR="$ALARIC_REMOTE_SCRATCH_DIR"` pointing that staging at node-local scratch. Size
   scratch for the decompressed footprint (~6 B/pose, ~4× on-disk). Pool already node-local:
   both are **skipped** — they would only add a local→local copy and that same ~4× expansion.
5. `alaric-deploy remote-chunk` prints a warning that running the chunks in parallel may
   degrade shared-filesystem performance, since that is the mode which forces the pool onto
   the shared FS.

**Worker counts honor the SLURM CPU allocation, not the node.** Actions that default their
worker count (`anchor --nprocs`, `rmsd --nprocs`, `organize --nprocs`, and the pose-index
thread pool in `middle/checksum.py`) resolve it through `alaric/nprocs.py:default_nprocs()`,
which checks in order:

    ALARIC_NPROCS -> PYTHON_CPU_COUNT -> SLURM_CPUS_PER_TASK -> SLURM_CPUS_ON_NODE
                  -> len(os.sched_getaffinity(0)) -> os.cpu_count() or 1

Values that do not parse as a positive integer are skipped rather than fatal, so
`PYTHON_CPU_COUNT=default` (legal under Python >=3.13) falls through to the next source.
`ALARIC_NPROCS` is the deployment-wide manual override; an explicit `--nprocs` on the command
line still beats everything.

Rationale: `os.cpu_count()` reports the whole node and therefore oversubscribes an
`sbatch -c N` reservation. The remote prologue exports `PYTHON_CPU_COUNT` from
`SLURM_CPUS_PER_TASK`, which **only** works under Python >=3.13 — 3.12 ignores the variable
entirely, so relying on the interpreter silently gave back the full node core count. Reading
the allocation in Python makes this version-independent, and the affinity fallback covers
cgroup-confined tasks that expose no SLURM env vars at all. `grow` and `anchor-refe` are
deliberately excluded: their `--nprocs` defaults to 1 in the code, which is a real default,
not an oversight.

Worker counts are non-load-bearing, so resolving them from the environment cannot change a
checksum.

**Non-load-bearing knobs must be present in the chunk templates in commented-out form**
(spec: middle-level.txt:153), via a bash array expanded with the `set -u`-safe idiom
`${opts[@]+"${opts[@]}"}` (older HPC bash errors on empty `"${opts[@]}"`). The set
(`--unorganized-subdirs` [active], `--cache-size`, `--poselock`, `--nprocs`,
`--local-tempdir`, `--local-stagedir`, `--capacity`, `--chunk-poses`) is non-load-bearing —
organize canonicalizes the result, so none changes the checksum (bucket-size and
max-poses-per-file stay pinned, active). The local deployer keeps its pool under
`CACHE/results/<SIGIL>` and may use `$TMPDIR` staging when a workstation partition is tight.

Sigil and lifecycle invariants:
- Sigil part 1: resolved load-bearing params excluding dependency fields and `__file` metadata
- Sigil part 2: dependency-name to dependency-sigil dict, or `null`
- File params split into checksum-bearing field plus `__field` filename metadata
- `result.txt` is written only by deploy cache hit, result-check, or result-download
- Cross-location comparisons are checksum-based, never compressed-byte-based
- Pose checksums are zstd-transparent deepfolder sidecars; `.npy`/`.npy.zst` checksums
  are zstd-transparent array-byte sidecars

### run.sh / check.sh contract
- `run.sh` is the compute step. Order: (1) invoke `check.sh` first to fail fast; (2) run the
  action backend; (3) organize (`anchor`/`grow`); (4) write result checksum sidecar(s);
  (5) for local deployment only, copy the final sidecar checksum into
  `CACHE/checksum/<SIGIL>`. `run.sh` does NOT write `result.txt`.
- `check.sh` is the **input provenance + materialization guard** (generated by Python, not
  templated). For each input dependency it verifies:
  - **provenance**: the input dir's `sigil.txt` equals the dep→sigil recorded in this
    action's `CACHE/parameters/<SIGIL>`, and the input's `result.txt` is present;
  - **materialization**: the input's bulky result data is present at the location this run
    reads from — `CACHE/results/<input SIGIL>/` (local) or `<RESULT_DIR>/<input SIGIL>/`
    (remote).
  Provenance alone is insufficient: `result.txt`/checksum persist after results are deleted.

### Operational lifecycle (mixed local/remote + cleanup)
- Actions alternate local/remote execution freely, chosen per action by which deployer is
  invoked. Lightweight state (`sigil.txt`, `CACHE/parameters`, `result.txt`, checksum) is
  kept; bulky pose results are transient and movable — the checksum is the durable identity.
- `alaric-result-clean` deletes `CACHE/results/<SIGIL>/*` once an action's *immediate*
  downstream consumers have completed (their `result.txt` written); this bounds disk use and
  is safe because the kept checksum still proves identity and the data is re-materializable.
  `--all` additionally drops `result.txt` + `CACHE/checksum/<SIGIL>`.
- When an input's data lives on the other side, the remedy is `result-download` /
  `result-upload` (checksum-based, zstd-transparent) before the consumer runs — exactly what
  check.sh's materialization check enforces.

## Milestones
1. Project/schema core
2. Auto-resolution
3. Graph/sigil/checksum engine
4. `score_add.py`, `mask.py` (note: `select-poses.py` **already** supports bool masks at
   `select-poses.py:53-59` — this milestone adds tests only, no new implementation)
5. Backend/template-context builder
6. Local deploy
7. Result lifecycle
8. Remote and chunk deployers
9. CLI/docs integration

Each milestone is gated by its matching test file group below.

## Test Plan
Create `tests/middle/` with:
- `test_schema.py`: valid/invalid YAML, alias handling, normalization, illegal `auto`, route XOR
- `test_resolve.py`: one test per auto rule above, using `tests/1b7f-project`
- `test_graph.py`: topo order, missing deps, cycles, dependency-kind mismatches, identity mismatch
- `test_checksum.py`: Seamless JSON stability, byte checksum, `.npy`/`.npy.zst` transparent sidecar behavior, pose-dir `INDEX`/`CHECKSUM` deepfolder behavior
- `test_sigil.py`: full project sigils, root `part2 == null`, cache param contents, idempotent rerun, stale-result abort behavior
- `test_backends_arrays.py`: `score_add` success/failure, `mask` representation choice, shape validation
- `test_select_poses.py`: bool mask and integer index inputs, base rules, out-of-range and length errors
- `test_backend_templates.py`: expected template context for every action, score pathing, non-load-bearing exclusion from sigils
- `test_deploy_local.py`: `sigil.txt` requirement, `result.txt` refusal, cache-hit behavior, generated `run.sh` and `check.sh`
- `test_result_lifecycle.py`: clean/check/upload/download/idempotency/mismatch behavior with fake remote dirs
- `test_deploy_remote_chunk.py`: remote payload layout, `scp`/`rsync` call shapes, anchor/grow/score chunk generation, organize step, score concatenation step

Acceptance criteria:
- `pytest tests/middle` passes
- existing `pytest tests/1b7f` passes
- `alaric-sigil tests/1b7f-project` is deterministic across reruns
- no unresolved decisions remain about schema, auto rules, chunking, template inputs, transport, or checksum semantics

## Assumptions
- Use the `alaric` conda environment for implementation, CLI smoke tests, and pytest
  runs. Examples in this handoff should be read as `conda run -n alaric <command>`
  unless an activated `alaric` shell is explicitly stated.
- `seamless-core >= 0.1.4` is required (add to `pyproject.toml`). The `>= 0.1.4` floor is
  mandatory: `seamless-checksum-file` was bugged w.r.t. zstd-compression transparency and
  only fixed in main (0.1.4), and the zstd-transparent pose-checksum invariant depends on
  that fix. Available in the Alaric conda env via editable install from git.
- `rmsd.sequence` is not part of the middle-level schema in this milestone
- `threshold:auto` explicitly fails for `score_add` and `rmsd`
- Organize layout knobs stay pinned: bucket size `16`, max poses per file `100_000_000`
