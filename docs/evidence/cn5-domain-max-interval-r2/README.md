# CN5 domain-maximum interval diagnostic evidence

Reproduce:

```bash
python scripts/diagnose_cn5_domain_max_intervals.py --output results/cn5-domain-max-interval-rN
```

This is a post-hoc diagnostic on France 2024-03:12 (7,344 hourly labels) because that target was already scored in the frozen FR fuel-share confirmation. The runner reuses and verifies the sealed France point predictions, refits the nine country-held-out 2023H2 source pseudo-target models, computes hourly pooled-q90 and maximum-country residual radii, then writes paired interval metrics.

Key artifacts:

- [`summary.json`](../../../results/cn5-domain-max-interval-r2/summary.json): input, code, and sealed prediction hashes plus limitations.
- [`source_country_maxima.csv`](../../../results/cn5-domain-max-interval-r2/source_country_maxima.csv): per-country/per-method maximum and q90 residuals.
- [`calibration_radii.csv`](../../../results/cn5-domain-max-interval-r2/calibration_radii.csv): pooled and domain-maximum radii.
- [`fr_interval_metrics.csv`](../../../results/cn5-domain-max-interval-r2/fr_interval_metrics.csv): FR coverage, width, Winkler score, and full-domain coverage.

The maximum source residual increased FR intervals to 3.7–4.4 times the width of pooled q90 intervals; pooled intervals already covered every FR target hour. This does not establish a new-domain guarantee. The protocol and interpretation are in `docs/CN5_DOMAIN_MAX_INTERVAL_DIAGNOSTIC_PROTOCOL_20261102.md` and `docs/CN5_DOMAIN_MAX_INTERVAL_DIAGNOSTIC_RESULTS_20261102.md`.
