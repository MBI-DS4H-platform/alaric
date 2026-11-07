# Export gens E6 result

**Scope:** Stage 6 export readiness, backend-supplied X5 warnings and digest-bound acknowledgment, manifest-checked ZIP transfer, cancellation and stale indication. Preparic now reads the `export_admission` share checksum alongside its state-graph checksum, exposes current export nodes to Stage 6, and resolves the download through the active Seamless REST/proxy URL. The download uses `fflate` to read `alaric_project/DATA/export-manifest.json`; it sends no consent data and releases no ZIP unless schema 1 and the current digest match.

**Focused E6 command:**

```text
conda run --live-stream -n preparic npx vitest run tests/frontend/export.test.ts tests/frontend/export-download.test.ts
```

**Result:** 30 passed (17 readiness/acknowledgment cases and 13 transfer/cleanup cases).

## Frontend test gate

```text
conda run --live-stream -n preparic npm test -- tests/frontend/export.test.ts tests/frontend/export-download.test.ts
```

**Result:** 277 passed across 17 frontend test files. The npm script expands to the full `tests/frontend` suite plus the two explicit E6 paths.

## Check and production build

```text
conda run --live-stream -n preparic npm run check
```

**Result:** `svelte-check` reported 0 errors and 0 warnings; `tsc --noEmit` passed.

```text
conda run --live-stream -n preparic npm run build
```

**Result:** Vite build passed. It emitted the existing notice that `/seamless-client.js` is not bundled because it is not a module, and the existing large Molstar chunk warning.

## Review handoff

No browser interaction test was performed by A2. A3 reviewed the Stage 6 gate, checksum freshness, upstream/admission/tree state distinctions, X5 message and digest-bound local acknowledgment, request cancellation, manifest validation, error mapping, stale indication, and preservation of the existing Stage 6 Gaussian editor. The review found one cleanup edge: if removing the temporary anchor throws, the object URL still must be revoked. A2 fixed this with nested `finally` blocks, added the failure-path regression through A1, and reran the focused suite: 30 passed in 2 files. A3 review is clear; no tests were run by A3.

A4 phase verification:

```text
conda run --live-stream -n preparic npx vitest run tests/frontend/export.test.ts tests/frontend/export-download.test.ts
```

**Result:** 30 passed (17 readiness/acknowledgment cases and 13 transfer/cleanup cases).

```text
conda run --live-stream -n preparic npx vitest run tests/frontend/w7-stage6-editor.test.ts tests/frontend/w45-stage5.test.ts --testNamePattern='W8b stage 6 exposes no reference-anchor editor|keeps the editor pending immediately, coalesces edits, and writes only at 500 ms|discards queued edits on a newer remote value and Back|Stage 5 keeps S24 editable and omits the S11 RMSD editor|test_stage5_displays_computed_energy_threshold_beside_edit'
```

**Result:** 5 selected regressions passed; 78 other cases in those two files were skipped by the name filter.

For the read-only browser smoke, no Preparic start process or frontend/backend listener was present before launch. Started both services with:

```text
conda run --live-stream -n preparic ./preparic-start --log /tmp/preparic-e6-a4.log
```

The A4 Chrome DevTools MCP profile was already locked, so the root agent's attached DevTools session loaded `http://localhost:5173/`. The accessibility snapshot showed **STAGE 06**, **Export project**, the checkbox **“The nearest-to-native structure doesn't pass the filters. Proceed anyway”**, one inspectable warning for `['U', 6, 2, 10]` (selected Gaussian validation does not pass because of implied results), and a disabled **Download export ZIP** button before acknowledgment. The warning disclosure was expanded read-only; no control was changed. Stopped services with `./preparic-stop`; process and port checks confirmed no `preparic-start`, backend, Vite process, or port 5173 listener remained.

Selected regression test names:

- `W8b stage 6 exposes no reference-anchor editor while keeping Gaussian validation`
- `keeps the editor pending immediately, coalesces edits, and writes only at 500 ms`
- `discards queued edits on a newer remote value and Back`
- `Stage 5 keeps S24 editable and omits the S11 RMSD editor`
- `test_stage5_displays_computed_energy_threshold_beside_edit`
