# W8 runtime fixture

`energy-filter-integration.json` records the finite-sample grid error bound for
the W8 cascade test. `tests/test_w8_energy_filter_integration.py` compares the
same 1B7F organized pose source with `cascade: []` and the default cascade,
local and local-chunk templates, process counts, and chunk sizes that straddle
`min_cascade`. It checks ARC bytes, selected-ID/order provenance, and whole-
action grid/gridless counts across chunk configurations.

The historical synthetic export-seam fixture was removed in E7 after its
useful X1/X2/X3/X6/digest assertions moved to real Preparic input-pin fixtures
and rendered export trees. Export admission and the downloadable artifact are
tested in Preparic's `tests/backend/test_export_e7_acceptance.py`.
