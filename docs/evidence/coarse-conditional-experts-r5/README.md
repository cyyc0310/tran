# Coarse-composition source expert transfer evidence

`results/` contains sealed 2024 target predictions, source-only pseudo-target temperature and risk-scale selections, all fixed and expert baselines, monthly/country/macro scores, and startup/prediction hashes. `code/` preserves the exact final runner, shared inference modules, and frozen protocol. `input/` preserves Energy-Charts acquisition and quality manifests.

The 181 MB Energy-Charts hourly responses remain in local ignored `data/energy-charts-v2-panel-r1/`; response SHA-256 values are in `results/startup_hashes.json`. Target labels were scored after prediction sealing. The risk-scaling selection is based on source pseudo-target folds; no target coverage was used to pick its multiplier. The final result remains an exploratory five-country proxy experiment, not a confirmed transfer method.
