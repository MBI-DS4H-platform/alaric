# Stale example YAMLs

The anchor action now accepts only precomputed Gaussian filters
(`precomputed_filter_name: gaussian` + `precomputed_filter_threshold: rrm*`,
loaded from `precomputed-filters/gaussian/<name>/`). The legacy `r90/r95/r99`
filters and brute-force `angle`/`dihedral`/`margin` anchoring were removed.

Many example projects under `tests/` (e.g. `1b7f-gaussian-anchor*/*/alaric.yaml`,
`1b7f-project*/DATA/anchor.yaml`) still use `r90/r95/r99` or `angle`/`dihedral`
keys. They are stale and will fail schema validation or path resolution until
migrated (e.g. to `precomputed_filter_threshold: rrm95`).
