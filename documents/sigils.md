# Sigils, runs and the cache

One Preparic export holds the material for many Alaric runs. Usually you make
several of them: a quick `anchor-refe` run to test the setup, then the
`anchor` run, then variants that change one step. Sigils are what make those
extra runs cheap. Alaric recomputes only the steps that actually differ, and
takes everything else from the cache.

## What a sigil is

A sigil is the identity of one action's result. `alaric-sigil` computes it for
every action directory and writes it to `sigil.txt`:

```
frag4-fwd-grown    d89807148d-e8d08e9bb5
```

It has two parts, each the first ten characters of a SHA-256 checksum:

1. **What is computed.** The checksum of the action's resolved parameters.
   `auto` values are first filled in from `DATA/`.
2. **What it is computed from.** The checksum of the input actions' sigils.
   A root action has no inputs, so its second part is `null`.

The second part contains the sigils of the inputs, and those contain the sigils
of their own inputs. A sigil therefore covers the whole history of a result,
from `DATA/` onward.

## The `alaric-sigil` command

Run `alaric-sigil` in a run directory after copying gens into it:

```bash
alaric-sigil                 # every action in the run
alaric-sigil --pool frag4    # frag4 and everything upstream of it
```

It does three things:

1. **Checks the run.** It reads every `alaric.yaml`, resolves `auto` values
   from `DATA/`, and checks that every input exists in the run. A missing
   input stops it with an error like `unknown action dependency: frag3`, and
   so does an invalid parameter.
2. **Computes the sigils** in dependency order. Each sigil goes to
   `<action>/sigil.txt`, and the resolved parameters behind it to
   `CACHE/parameters/<SIGIL>`.
3. **Prints** one `<action>  <sigil>` line per action.

A sigil depends only on the action's parameters, the values it reads from
`DATA/` and its inputs. Running `alaric-sigil` again therefore changes nothing,
unless one of those has changed.

**Stale sigils.** Suppose you edit an action's `alaric.yaml`, a `DATA/` value
it uses, or anything upstream of it. Its sigil then no longer matches
`sigil.txt`, and `alaric-sigil` stops:

```
frag4-fwd-grown: stale sigil d89807148d-e8d08e9bb5 != ff87106df7-e8d08e9bb5; use --force or --delete
```

- `--force` accepts the new sigils and overwrites `sigil.txt`. It refuses for
  an action that already has a `result.txt`, because that result belongs to
  the old sigil.
- `--delete` also removes `result.txt` and the old sigil's
  `CACHE/checksum/` and `CACHE/results/` entries. With a shared cache, that
  deletes the old result for every run.

`alaric-deploy` computes the sigils of the action it deploys, and of everything
upstream of it, by itself. Like `--force`, it accepts a changed sigil when
there is no `result.txt`. Running `alaric-sigil` first is still worthwhile: it
checks the whole run and tells you what has changed.

## What goes into a sigil, and what does not

| In the sigil | Not in the sigil |
|---|---|
| The action and its parameters, after `auto` is resolved | The name of the action directory |
| The content (SHA-256) of the `DATA/` files the action consumes whole, such as a protein PDB | The location of the run, or of `DATA/` |
| The identity of the Gaussian filter and ring reference files | The deployer: `local`, `local-chunk`, `remote` or `remote-chunk` |
| The sigils of the input actions | The number of chunks |
| | The fragment library (`~/.alaric/fraglib.yaml`) and the Alaric code version |

Some consequences:

- **The same action gives the same sigil in every run.** This holds even when
  the action has another directory name, as long as its parameters, its
  `DATA/` content and its inputs are the same.
- **A result does not depend on where it was computed.** A result from the
  cluster and one from your laptop are interchangeable.
- **`auto` parameters are resolved to their values first.** Take a grow with
  `crmsd: auto`: its sigil holds the actual number from `constraints.json`. It
  changes only if that one pair changes. An explicit value is written in the
  same form: `crmsd: '0.25'` and `crmsd: 1` become the numbers 0.25 and 1.0,
  and `sequence: aa` becomes `AA`.
- **The extracted value goes into the sigil, not the file it came from.** This
  holds whether the value was given explicitly or resolved from `auto`. An
  `energy-filter` with `threshold: auto` holds the one threshold it reads from
  `energy-thresholds.json`. The same goes for `cascade` and `min_cascade` from
  `score-filter.yaml`. `threshold: auto` and an explicit `threshold: -15.93`
  with the same value give the same sigil. Editing another entry of the file
  changes nothing. Editing the entry the action uses changes only that action.
- **A file that an action consumes whole goes in by content.** The protein PDB
  of an `energy-filter` is an example. Any edit to such a file changes the
  sigil of every action that reads it.

`CACHE/parameters/<SIGIL>` stores the resolved parameters behind a sigil. Diff
two of these files to see why two sigils differ.

## How the cache uses sigils

```
CACHE/
  parameters/<SIGIL>   the resolved parameters (written by alaric-sigil)
  checksum/<SIGIL>     the checksum of the result (written when the action finishes)
  results/<SIGIL>/     the result itself: poses, scores or masks
```

`alaric-deploy` looks for `CACHE/checksum/<SIGIL>` first. If it is there, the
result already exists. Deploy then copies the checksum to `result.txt`,
generates no scripts, and the action is done. `alaric-result-check` also does
this, and in addition links `results` to the cached result.

Results are stored by sigil, so one `CACHE/` can serve any number of runs.
Different `DATA/` content gives different sigils, so a shared cache never
returns another run's result by mistake. That holds even for runs from
different exports.

The fragment library is deliberately left out of the sigil. It is meant to
stay constant for as long as you use Alaric, so it is not part of what
distinguishes one result from another.

## Many runs from one archive

Keep the extracted archive unchanged. Give each run its own directory, linked
to the archive's `DATA/` and to one shared `CACHE/`:

```
1cvj/
  alaric_project/          the extracted archive; DATA/ plus the gen catalogue
  CACHE/                   shared by all runs
  run-refe/
    DATA  -> ../alaric_project/DATA
    CACHE -> ../CACHE
    frag2-anchor-raw/ ...  copied from the catalogue
  run-anchor/
    DATA  -> ../alaric_project/DATA
    CACHE -> ../CACHE
    ...
```

To set up a run:

```bash
mkdir -p CACHE run-refe && cd run-refe
ln -s ../alaric_project/DATA DATA
ln -s ../CACHE CACHE
alaric-copy ../alaric_project/anchor-refe/fwd/gen-frag2-3/*
```

Copy gens into a run, don't move them, so that the catalogue stays complete
for the next run. Copy with `alaric-copy`, not `cp -r`. It copies only an
action's `alaric.yaml`, so an action copied from another run does not bring
along that run's `sigil.txt`, `result.txt`, `results` link or scripts. If one
of the actions is already in the run, it copies nothing.

## Example: two runs that differ in one step

Runs A and B contain the same gens:
- the `anchor-refe` gen for fragments 2–3;
- the forward grow gen for fragment 4.

The only difference is that run B sets `crmsd: 0.4` on the grow from fragment 3
to fragment 4. Run A has `crmsd: auto`, which resolves to 0.25.

| Action | Run A | Run B | |
|---|---|---|---|
| `frag2-anchor-raw` | `868e91fd39-null` | `868e91fd39-null` | same |
| `frag3-anchor-raw` | `73be5e21c2-null` | `73be5e21c2-null` | same |
| `frag3` | `5f096c3636-7d9ec0b458` | `5f096c3636-7d9ec0b458` | same |
| `frag4-fwd-grown` | `d89807148d-e8d08e9bb5` | `01894981d6-e8d08e9bb5` | first part differs |
| `frag4-fwd-unique` | `5f096c3636-5ca51b2888` | `5f096c3636-f3d7d2043d` | second part differs |
| `frag4-e1` | `e841dece95-f67d78326d` | `e841dece95-af8dfed98e` | second part differs |
| `frag4` | `f883ea4e04-08010f31a4` | `f883ea4e04-12f6c6ea97` | second part differs |

- **The edited action** changes its first part, its parameters. Its second
  part is unchanged, because it grows from the same `frag3`.
- **Everything downstream** keeps its first part, since its own parameters did
  not change. Its second part changes, because its input did.
- **Everything upstream** is untouched.

So run B, deployed after run A, takes all of fragments 2–3 from the cache and
computes only the four actions of fragment 4. Building fragments 1 → 2 → 3 → 4
in one run and then making a second run that differs only in 3 → 4 costs only
the 3 → 4 step. The same goes for choosing a different grow, merge or filter
after a common start.

The other way round, a change early in the chain changes every sigil
downstream of it. The cost of a variant is the cost of what lies downstream of
the change.

## Practical rules

- **Don't edit a finished action, make a new run.** Editing an action that
  has a `result.txt` makes its sigil stale. Getting past that in place takes
  `alaric-sigil --delete`, which removes the old result from the shared cache.
  A new run with the changed action keeps the old result and costs only the
  changed part.
- **A run needs its upstream actions present.** Each action's `check.sh`
  checks that its input action directories are in the same run, with matching
  `sigil.txt` and `result.txt`. So a variant run includes its whole upstream
  chain. Those actions come from the cache at no cost.
- **Don't compute the same sigil in two runs at once.** A local `run.sh` starts
  by clearing `CACHE/results/<SIGIL>`. Let one run finish it. The other run's
  `alaric-deploy` then finds it in the cache.
- **Free disk space with care.** `alaric-result-clean` removes a result from
  the shared `CACHE/` for all runs. The checksum stays, so the action still
  counts as done, but its data is gone until it is recomputed or downloaded.
- **Code changes.** A sigil identifies parameters and inputs, not the Alaric
  version. When a code change alters an action's results, Alaric adds a
  runtime version to that action's parameters, for example
  `energy_filter_runtime_version`. That changes the sigil, so the old cache
  entries are no longer used.

## Remote runs

On the cluster, results are stored as `$ALARIC_REMOTE_RESULT_DIR/<SIGIL>`.
That directory is shared by every run and project that uses it, so remote
results are reused across runs in the same way. `alaric-result-check` finds a
remote result by its sigil and records its checksum in the local `CACHE/`.
`alaric-result-download` fetches the data when you need it locally.
