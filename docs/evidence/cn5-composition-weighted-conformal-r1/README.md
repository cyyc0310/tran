# CN5 composition-weighted conformal evidence

This is a development-only nine-country leave-one-country-out evaluation of an interval reweighting method. The target point model is fixed to ratio recomposition. Target labels are fixed-factor generation proxies derived from revised Energy-Charts hourly series; they are not measured CIF labels.

The frozen inner rule selected the uniform calibration distribution (`h = infinity`) in all nine outer folds. Therefore composition weighting provided no outer increment. Same-calendar-month pooled cross-country calibration reduced macro interval width and Winkler score relative to the earlier 2023H2 pooled-radius baseline, but macro coverage fell to 88.84% and Czechia coverage remained 34.72%.

Reproduction:

```bash
PYTHONPATH=src:scripts python scripts/run_cn5_composition_weighted_conformal.py \
  --output results/cn5-composition-weighted-conformal-r1
```

The full machine output (all sealed hourly predictions, monthly scores, inner bandwidth searches, hashes and input audits) is in `results/cn5-composition-weighted-conformal-r1/`. See [protocol](../../CN5_COMPOSITION_WEIGHTED_CONFORMAL_PROTOCOL_20261101.md), [results report](../../CN5_COMPOSITION_WEIGHTED_CONFORMAL_RESULTS_20261101.md), and [runner](../../../scripts/run_cn5_composition_weighted_conformal.py).

