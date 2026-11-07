# Export gens E5 result

**Scope:** Preparic export wiring: the shared `EXPORT_INPUT_CELLTYPES` registry, both explicit transformer wrappers, source/helper modules, validator-null dependency pins, readonly admission share, private folder result, and the Seamless `export.zip` download. E5 also added the contract's `export_tree.build_tree(inputs, admission)` orchestration and canonical JSON key ordering so worker and local rendering produce byte-identical manifests.

**Gated Preparic command:**

```text
conda run --live-stream -n preparic pytest -q tests/backend/test_export_backend_bindings.py tests/backend/test_export_download.py
```

**Result:** 11 passed in 33.31s.

The binding checks confirm that both wrappers have the explicit plan signature and shared registry celltypes, use the same source cells and validator result pins, attach the expected helper source, share admission read-only, leave the tree unshared, and register `export.zip`. The worker integration runs the wrappers on Preparic's configured eight-worker pool with the production Gaussian model and attached helper sources. The resulting ZIP matches the expected 46-member folder byte-for-byte; PDB pass-through and response headers are checked. A semantic refusal leaves the tree transformer failed/result cell blocked and the registered download answers 409. The suite also includes E0 null-plain, state, and share-server regressions.

## A3 contract review

Reviewed the E5 wiring and tests against §§7.1 and 9.3 and the oracle contract. The single registry matches the specified pin/celltype table; both wrapper signatures carry the same registry pins plus the three plain validator-result gates, and the binding test checks their sources and attached interpreted modules. Admission is shared read-only, the tree result is unshared, and `export.zip` is registered against `alaric_project/export_tree`. The spawn-worker integration executes both wrappers with successful null validator cells, checks exact ZIP members and PDB pass-through, then confirms a semantic admission refusal leaves the tree blocked and the GET returns 409. No E5 contract issue remains.

**A4 existing regression names:** `test_schema1_constraints_lowercase_and_uppercase_aliases`, `test_invalid_constraints_refused_by_both_consumers`, `test_missing_requested_constraints_entry_does_not_use_fallback`.

## Share server and Vite proxy probe

Started both services with the Preparic interpreter:

```text
conda run --live-stream -n preparic ./preparic-start
```

Vite served on `http://localhost:5173/`; the Seamless share server served on `http://127.0.0.1:5813/`. A GET to `http://localhost:5173/preparic/export.zip` returned `200`, `Content-Type: application/zip`, and `Content-Disposition: attachment; filename="export.zip"`. `unzip -t` verified the archive and found 267 members under `alaric_project/`. The same GET to `http://127.0.0.1:5813/preparic/export.zip` returned `200`; both responses had SHA-256 `51e9387a28e6babf0a0b33097ccf612bbd47a864b54f8f489b8718f2062ccd06`.

Stopped services with `./preparic-stop`; a subsequent process check found no `preparic-start`, backend, npm, or Vite processes. This was an HTTP proxy check, not a Chrome DevTools browser check.

**Not run in E5:** frontend implementation and end-to-end deployment/action execution. A3/A4 review and live verification are recorded below.

## A4 live regressions and browser check

The E5 Preparic suite was rerun after A3 review:

```text
conda run --live-stream -n preparic pytest -q tests/backend/test_export_backend_bindings.py tests/backend/test_export_download.py
```

**Result:** 11 passed in 32.32s.

The A3-selected Alaric regression nodes were run in the `alaric` environment:

```text
conda run --live-stream -n alaric pytest -q tests/middle/test_export_e1_contract.py::test_schema1_constraints_lowercase_and_uppercase_aliases tests/middle/test_export_e1_contract.py::test_invalid_constraints_refused_by_both_consumers tests/middle/test_export_e1_contract.py::test_missing_requested_constraints_entry_does_not_use_fallback
```

**Result:** 46 passed in 0.07s.

Started Preparic with `conda run --live-stream -n preparic ./preparic-start --log /tmp/preparic-e5-a4.log` and stopped it with `./preparic-stop`. The post-stop process check found no backend, Vite, or `preparic-start` processes. A4's Chrome DevTools MCP process could not attach because another process owned the shared browser profile; its `new_page`, `list_pages`, and `navigate_page` calls failed, including a named isolated-context retry. The coordinator's already-attached Chrome DevTools MCP session did reach the proxy URL: `GET http://localhost:5173/preparic/export.zip` was captured as status `200`, with `Content-Type: application/zip`, `Content-Disposition: attachment; filename="export.zip"`, `Cache-Control: no-cache`, and content length 160309. Chrome reported `net::ERR_ABORTED` for the navigation because the response is an attachment download; the response body was unavailable in DevTools. A2's HTTP/unzip check verified all 267 ZIP members and the direct-share/proxy SHA-256 match recorded above.
