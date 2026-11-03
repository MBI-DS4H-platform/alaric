# W4/W5 focused acceptance set (authored, not executed)

Authority: implementation-plan §7; oracle-contract A5, G2–G4/G8,
P5.1–P5.8, P7/P10; contract-boundaries T12–T16/T27 and §4;
state-transitions §§4.2, 5, 7, 10. Final contracts override this checklist.
Existing W4 edits are retained. This file specifies integration checks where
Preparic's W5 API does not yet exist; those checks are not automated coverage
and must not be reported as passing. No test, server, or browser was run while
authoring this set.

## Executable A5 additions

From `/home/agent/alaric`:

```sh
conda run --live-stream -n alaric python -m pytest tests/test_energetics_deps.py tests/test_w45_energetics_acceptance.py -q
conda run --live-stream -n preparic python -m pytest tests/test_energetics_deps.py tests/test_w45_energetics_acceptance.py -q
```

`test_w45_energetics_acceptance.py` adds four black-box cases: failed receptor
removes an earlier success file; empty ligand fails locally while non-PDB files
and directories are ignored; equal-byte reference/best-fit names remain two
rows; relocating ligand files preserves the entire scientific JSON. The last
case is output parity, not evidence of transformation-identity parity.
Use existing `test_energetics_deps.py` for precision, bead counts, reduction
counts, unknown residues, flat import restriction, real Seamless sandbox, and
prerequisite publication. Record successful raw JSON from both environments;
compare the float64 energies to 1e-6 and counts exactly. No float32 npy threshold
source, fitted superposition, minimization, grid, or cutoff is acceptable.

## ID: actual dependency identity (G2–G4, A5)

Use a disposable copy of the complete eight-file staging tree and the actual
T13 transformation constructor, or the actual `seamless-run -I` graph when no
factory exists. Capture the real transformation checksum and all input-pin
checksums before execution. A hand-computed digest of a manifest is insufficient.
For each case restore baseline first, use identical ligand/receptor bytes, and
record `(case, baseline identity, changed identity, changed pin, execution count)`:

1. Independently change bytes in each of the eight declared files. Each changes
   the transformation identity. Append a comment to Python/data text; change a
   copied parameter value in the npz with numpy for the binary dependency. These
   identity probes need not execute modified scientific data.
2. Independently change a grid-only file and a JAX-only file outside DEPS. Neither
   changes identity; choose files absent from the exact eight-entry list.
3. Relocate the complete staging tree with byte-identical contents. Identity is
   unchanged. Renumber a displayed range while receptor bytes remain unchanged:
   identity is unchanged and no range index appears in A5's output.
4. Change ligand bytes, then receptor bytes: each changes the corresponding pin
   and identity. A supplied factory has both pins unbound on construction and
   does not initialize or execute Seamless during construction.
5. Observe eight actual spawn workers and unavailable hashserver/database with
   persistent caching disabled; successful scoring must come from a worker PID.
   A profile file alone does not establish worker count or process execution.

Do not substitute a checksum calculated by test code for the delivered graph's
identity. Identity observations belong in the eventual W5 backend integration
harness; no unimplemented factory API is presumed by these tests.

## I: per-range integration and cost (P5.1/P5.7/P10)

Use committed 1b7f preparation fixtures, two protein ranges, full T6 reference
and T5 discrete best-fit PDB folders. Capture per-file extraction/preparation
checksums, T12 folder entries, each T13 receptor/ligand/transform checksum,
execution counters and S13/S15 content echoes. Establish a fresh baseline.

| Mutation | Required evidence |
|---|---|
| Edit only protein range 2 | Unchanged range 3 extracted/prepared bytes, T13 pins/identity/counter; exactly range 2 rescored |
| Insert/remove/reorder a protein row | Instances reconcile with S5, no stale T14 column, same receptor bytes keep scoring identity despite display renumbering |
| Refit changes ligand bytes | Every affected T13 rescores; T12 contains all refe/bestfit files, including clipped rows |
| Clip edit in stage 4 | No fit, best-fit-table, T12/T13/T16 execution; S14 retained inactive until Continue |
| Matrix edit in stage 5 | Only T15 execution; no T13/T16, fit, A9 or Gaussian-score execution |
| Ordinary energy-threshold edit | One S24 write, only T27 execution; no S14 write or scoring/fit execution |
| Class edit | No T13 or A9 execution |
| Ligand error outside locks | Error remains row-local and 4→5 can proceed; growth stops before it |
| Receptor error | Invocation fails and stage-scoped diagnostic appears; no fresh S13 success |

For every successful cell, S15 equals the attached Alaric A6 result for the raw
best-fit float64 energy. Include raw exact `-16.59` → `-16.58`, `-16.5913` →
`-16.59`, `0` → `0.01`, `0.123` → `0.13`; reference energy changes alone cannot
change that cell's threshold. Unknown output schema is refused by the consumer.

## M: deterministic pure interaction vectors (P5.3–P5.6)

Use addressed fragment rows f1…f9 and protein ranges R2, R3, R4.
S14 has one column per distinct collapsed `(RNA residue, protein range)`
anchor, sorted along RNA; its `ranges` list may repeat a range ID. A range
without an anchor has no true cells. Joined anchors remain separate columns
in grid↔columns round trips, including k columns for a k-anchor joined run. Resolve locks
through the fragment map, never by subtracting residue labels. For these vectors
explicitly map anchor A to lock `{f4}`. Energies below are `(refe,bestfit)`:

| Row | R2 | R3 |
|---|---|---|
| f1 | (2,0) | (0,0) |
| f2 | (0.999,-0.999) | (0,0) |
| f3 | (1,0) | (0,0) |
| f4 | (0,0) | (0,0) |
| f5 | (0,-1) | (20,0) |
| f6 | (25,0) | (0,0) |
| f7 | error | (0,0) |
| f8 | (-4,0) | (0,0) |
| f9 | (0,0) | (0,0) |

M1: only A=(residue r,R2) selected, uncut clip. Recompute produces R2 true
exactly `{f3,f4,f5,f6}`. f2 is excluded (strictly below magnitude 1), f1 is
unreachable, f7 is excluded (error), f8 is unreachable. Positive f6 counts.
All R3/R4 cells are false despite high R3 energy.

M2: add B on R3 with lock `{f5}`. Its run is `{f5}`, overlapping R2 legally.
Add a distinct stacking/protein-residue key with the same `(r,R2)` as A:
there is still one matrix column, but both original Gaussian anchors/jobs remain.

M3: separate data with high energies throughout; two same-range locks `{f3,f4}`
and `{f5,f6}` force a join that cannot be switched off. Repeat with overlapping
locks `{f3,f4}`/`{f4,f5}`. With separated locks, joining is allowed only to the
next downstream anchor on the same range; skip no intervening same-range anchor.
Disallow cross-range joins. Growth stops at the next lock, then touching/overlap
joins; caps cannot cross locks, enter clipped/error rows, or overlap an unjoined
same-range neighbour. Cross-range cap overlap remains allowed.

M4: grid→columns→grid preserves addressed booleans exactly, including joined
multi-anchor runs, multiple separated same-range runs, different-range overlap,
and all-false unanchored ranges. Joined run with k anchors restores k columns.

M5: independently refuse each corrupted read: wrong row count; ragged grid;
non-boolean `0`, `1`, `null` or string; missing/duplicate/out-of-order row key;
wrong range order/extent; same row count but opposite-end clip; missing lock;
unanchored true run; true error cell; unknown schema where schema is required.
Editor disables and names failed check; no automatic share write; Recompute stays
enabled. An error at A's lock names A and asks for deselection in stage 3, never
silently removes A or fabricates energy. Backend T15 checks structure; full
export anchor validation remains deferred.

M6: clip adaptation uses old addressed rows f3…f7, then new f1…f9. With all new
rows high energy, an R2 outer end at f3 expands to f1; an R3 user cap at f4 stays
f4. An end at f7 grows to f9 independently per range. Repeat with f2 low energy
or error: expansion stops before f2. Preserve every old cell, inner cap, lock
and join. Narrow clip back: only covered rows disappear. Same clip restores
identical grid; changed selected anchors chooses Recompute. Selected candidate
locks cannot be clipped; 3→4 clamps a newly selected anchor before S11 write.
A racing clip write during 4→5 makes S14's row extent inapplicable and leaves
stage 4; no racing-tab atomic transaction requirement is added.

## E: energy table and hand-off (P5.7/P5.8, D13)

E1: null S24 or stale PDB/range-1 basis creates every fragment×range as empty;
S24 acknowledgement precedes S14 on Continue 4→5. Existing unchanged range rows
retain typed values. A changed range row resets only itself, even if another
range changed index; deleted rows drop and new rows start empty. Preserve
values across refit/code updates and update diff marks rather than overwriting.

E2: computed values are visible beside edits. Copy disabled on stale S15;
fresh Copy fills every computed cell, requires confirmation before overwriting
typed values, cancel writes nothing. Per-cell revert uses current computed
value and updates diff count. Accept signed finite values with ≤2 decimals;
reject third decimal (do not round), NaN/infinity and malformed numeric input.
False/clipped cells are grey but editable. Table remains editable in stage 6.

E3: keep 5→6 closed for any empty/invalid true cell, bad matrix, stale fit,
stale geometry, stale/backend-failed Gaussian scores or failed goldens, or
failed T27. Fresh typed thresholds allow advancement while S15 is stale,
provided all other prerequisites are fresh. Cap extension adding an empty true
cell closes the gate. T27 accepts null/stale portions as specified, validates
current fragment keys and null/two-decimal-string values without waiting for
scoring; T27 failure is a gate condition, not a freshness receipt.

E4: use edited values distinct from S15, e.g. `-16.60`, `0.00`, `21.00`.
For chosen true kept cells assert exact JSON:
`{"schema":1,"frag-3-GU":{"range-2":"-16.60"},"frag-4-UA":{"range-2":"0.00","range-3":"21.00"}}`.
No false/clipped cells or epsilon/method/source/rounding metadata. T15 hand-off
has current clip, actual kept residues, ordered ranges, all candidate anchors
and fragment→ranges entries. E≥edited threshold is retained and marks downstream
“Proceed anyway”; it is not silently raised to computed threshold. Export UI
implementation remains deferred.

## B: Chrome DevTools MCP browser validation protocol

Use the `chrome-devtools` skill at
`/home/agent/.codex/skills/chrome-devtools/SKILL.md`. No Playwright and no browser
CLI. Authoring this protocol does not authorize claiming it has executed.
Start the application separately when executing acceptance:

```sh
cd /home/agent/preparic
conda run --live-stream -n preparic preparic-start --log
```

Use MCP `list_pages`/`new_page`; navigate to the actual launcher URL, then
`wait_for` and `take_snapshot`. Obtain fresh element UIDs before `click`,
`fill`, or drag actions. Save snapshots/screenshots for each case and use
`list_network_requests`/`get_network_request` for share-write bodies and ordering;
`evaluate_script` may read live diagnostics but must not manufacture UI results.
Record the actual URL, page IDs, selected anchor keys, clip, share payloads,
node identities/counters and evidence paths. A screenshot alone cannot prove
execution cost or write order.

B1: complete 1b7f through stage 5; display refe/bestfit energies with capped
colour scale around ±20 and untruncated numeric values. Grey clipped rows retain
energies; error text is on the failing row. Exercise M1–M5 controls and E1–E3
through actual UI; after refused read observe no automatic S14 write before
pressing Recompute. S24 is empty at first entry and computed values visible.

B2: in two tabs edit a threshold, then a cap. Observe exactly the decision write
expected in I, synchronized values, diff marks, gate closing/reopening, and
read-only clip in stages 5–6. Back 5→4, edit clip, Continue: observe M6 and the
awaited S24 reconciliation before S14 when needed. Continue in one tab and
verify the non-editing tab follows. Race the clip only for the extent-invalid
case; racing Continue/Back admission is not an acceptance requirement.

B3: delay a scoring result, edit and revert its receptor range, and confirm
optimistic local edits immediately show pending/stale values rather than old
fresh scoring/gates. Release old/new results and verify content echoes. A
range error at an anchor lock names the anchor and retains selection. Recompute
remains enabled; resolution requires stage-3 deselection.

B4: restart backend, immediately reload every open tab, then verify durable
S14/S24/stage restoration before cold recomputation completes. Old computed
mounts do not establish freshness; fit/scoring rerun when required. Disconnect
and reconnect: reread current decisions and drop deferred disconnected writes.
Check stage 6 retains editable S24 and class edits do not execute A9/T13.

## Existing regression commands (no new API assumed)

These existing suites are supporting regressions, not substitutes for I/M/E/B:

```sh
cd /home/agent/preparic
conda run --live-stream -n preparic python -m pytest tests/backend/test_w3_preparation.py tests/backend/test_w3_stage4.py tests/backend/test_w1_spawn_runtime.py -q
conda run --live-stream -n preparic npm test -- --reporter=dot
conda run --live-stream -n preparic npm run check
```

Completion record must separate executable A5 results, automated W5 results
once implemented, and MCP observations. Missing W5 APIs/fixtures or evidence
are outstanding work, never a passing skip. No `seamless1` environment.
