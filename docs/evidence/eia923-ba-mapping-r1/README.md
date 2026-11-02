# EIA-923 BA alignment evidence

This package audits the join between 2024 final EIA-923 plant-month generation and the existing EIA-930 hourly feature panel for seven previously used balancing authorities.

## Contents

- `f923_2024.zip`: official EIA-923 2024 final archive.
- `results/ba_month_fuel_comparison_all.csv`: all 84 EIA-923 BA-month rows and any EIA-930 coverage available.
- `results/ba_month_fuel_comparison_complete.csv`: the 61 complete matched BA-month rows used for mix comparisons.
- `results/summary.json`: metric summary, source hashes, and mapping contract.
- `../../../scripts/audit_eia923_ba_mix_alignment.py`: reproducible parser.

## Reproduce

From repository root:

```bash
python scripts/audit_eia923_ba_mix_alignment.py \
  --eia923-zip docs/evidence/eia923-ba-mapping-r1/f923_2024.zip \
  --eia930-feature-dir data/eia930-hourly-co2-r1/feature_inputs \
  --output-dir docs/evidence/eia923-ba-mapping-r1/results
```

This is a measurement alignment audit. It does not establish that the 2024 final file was available at earlier forecast origins, and it does not score transfer models.
