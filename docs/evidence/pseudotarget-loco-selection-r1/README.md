# Nested pseudo-target model-selection evidence

`results/` contains all five country-held-out target prediction files, selection diagnostics, monthly scores, and startup/prediction hashes. `code/` preserves the runner, shared source-only inference modules, and the frozen LOCO protocols. `input/` preserves the Energy-Charts retrieval and data-quality manifests.

The 181 MB source responses remain in the local ignored `data/energy-charts-v2-panel-r1/` directory; their SHA-256 hashes are recorded in the result startup manifest. Target 2024 labels were scored only after the five forecast files were sealed. The experiment uses reconstructed lagged months and fixed-factor generation proxies, not historical as-issued CIF data.
