W8 contract fixture/test matrix (oracle-contract A12, X1–X7)

`export-seam-cases.json` fixes contract values independently of the bundle builder.
The export page and directory layout are deferred. Bundle acceptance must verify:

* X1: common exclude and fraction exactly match preparation; no environment override.
* X2: preserve arbitrary RNA bytes (including CRLF/final newline), record their SHA256;
  checksum mismatch is an error.
* X3: one Gaussian action per selected anchor with edited filter name; for each
  selected RNA residue, emit every kept dinucleotide candidate containing it,
  including its neighboring fragment when kept. This fixture yields four
  anchor-refe actions, with overlapping candidates deduplicated. All ovRMSDs
  explicit, never `auto`.
* X4: export only true kept matrix cells in energy file; preserve edited values,
  epsilon 15 and auto cascade/min_cascade in each range-specific action; include
  exact defaults above, edited kept-pair grow values, handoff and fragment map.
* X5/X6: apply readiness_cases; proceed-anyway can admit soft failures but never
  bypass hard blocks. Non-emitted clipped pair values do not block.
* X7: modifying any decision after bundle creation makes it stale; changing the
  caller's dictionaries must not mutate an already-created snapshot.

Recorded-pose runtime acceptance is in `tests/test_w8_energy_filter_integration.py`.
It compares the same 1b7f organized pose source with cascade=[] and the default
cascade, local and local-chunk templates, 1 and 2 processes, and chunk sizes
that straddle min_cascade. It checks ARC bytes, selected-ID/order provenance,
and whole-action grid/gridless counts across chunk configurations. The recorded
finite-sample bound gate passes for this one full dom2 protein range: observed
maximum grid error is strictly below half its max_error before cascade use.
Other protein ranges and clipped ranges still need their own bound evidence.

Executable fixture consumer: `tests/test_w8_export_seam_contract.py`, using
`alaric.export_bundle.build_export_bundle` and `bundle_is_stale`. Its X5 cases
consume the explicit browser-computed `proceed_anyway_required` bit; they do not
establish browser numerical classification of red/amber states. Its X7 cases
establish pure bundle snapshot behavior, not an asynchronous export-page race.
The page, actual artifact layout, and asynchronous export-page race remain
outside this focused suite.
