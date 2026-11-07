# E0 Seamless download probe

Implementation plan: `export-gens-plan.md` §8 E0.

- Seamless workflow revision: `9b7c5cc` at probe preparation.
- Preparic environment: `preparic`.
- Probe: `/home/agent/preparic/tests/backend/test_export_download.py`.
- Browser/Vite proxy probe: namespace forwarding verified in E0; the export URL is
  not registered until E5.

## Results

| Contract case | Result | Evidence |
|---|---|---|
| ZIP folder members and exact leaf bytes | Pass | `test_e0_folder_zip_exact_bytes_and_required_null_plain_pins` |
| 409 for waiting, failed and blocked cells | Pass | The slow producer returned 409 while `export_tree` was `waiting`/`computing`; the completed request returned 200. Separate failed and blocked cases returned 409. An unwired required pin is exposed as output state `blocked` and also returns 409. |
| Successful null on required `plain` pins | Pass | `test_e0_folder_zip_exact_bytes_and_required_null_plain_pins` |
| Changed PUT invalidates downstream before the next cut | Pass | `test_e0_changed_put_cannot_serve_old_tree_and_equal_put_keeps_tree` |
| Same-value PUT leaves the completed result current | Pass | `test_e0_changed_put_cannot_serve_old_tree_and_equal_put_keeps_tree` |
| Edit after the cut leaves the captured archive consistent | Pass | `test_e0_edit_after_cut_keeps_old_archive_consistent` |
| Unresolvable folder leaf returns 503 | Pass | `test_e0_missing_folder_leaf_answers_503` |
| Restart recomputes and serves a fresh folder | Pass | `test_e0_folder_results_resolve_after_fresh_interpreter_restart` |
| Vite proxy URL | Pass | Chrome request to `/preparic/state-graph` returned the backend's JSON `200`; `/preparic/export.zip` returned the backend's expected `404 {"error":"no such share"}` while the download is not yet registered. |

## Live Vite proxy check

Started both services with
`conda run --live-stream -n preparic /home/agent/preparic/preparic-start --log /tmp/preparic-e0-proxy.log`
and stopped them with
`conda run --live-stream -n preparic /home/agent/preparic/preparic-stop`.
Chrome DevTools loaded `http://localhost:5173/preparic/state-graph` and received
`200` JSON from the Seamless backend. A same-origin fetch of
`http://localhost:5173/preparic/export.zip` returned `404 application/json`
with `{"error":"no such share"}`, confirming the namespace route reaches the
backend before E5 registers that download. The registered ZIP response is
verified after E5 wiring.

## Command

`conda run --live-stream -n preparic pytest -q tests/backend/test_export_download.py` — initial run: **6 passed, 1 failed** in 2.40 s. The pending fixture had not scheduled a computation to hold, so its producer had completed by the GET. After making the pending run deterministic with a slow transformer and a background `compute()`, the final run was **8 passed** in 3.76 s. During fixture adjustment, the unwired required pin surfaced as output state `blocked` (rather than `unwired`); the assertion now records the actual state. The final tests cover a distinct `waiting`/`computing` cut, failed and blocked cells, and the unwired-pin case. These were test-fixture corrections, not Seamless contract violations. No Seamless code was changed.

## Regression verification

- Preparic: `conda run --live-stream -n preparic pytest -q tests/backend/test_backend_workflow.py::test_backend_graph_mounts_and_shares_stacking_cells tests/backend/test_w1_spawn_runtime.py::test_backend_runs_real_eight_worker_pool_without_tcp_services tests/backend/test_export_download.py` — **10 passed** in 27.92 s.
- Seamless workflow: `conda run --live-stream -n preparic pytest -q tests/test_share_downloads.py::TestDownloadGet tests/test_contract_pins_bound.py::test_bound_required_nullable_pin_accepts_null` — **13 passed** in 1.18 s. A best-effort `seamless.close()` atexit warning followed; the command exited successfully.
