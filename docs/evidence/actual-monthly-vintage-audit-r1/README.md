# Actual monthly-vintage audit evidence

This package compares April 2024 state generation observations in EIA's June 2024 and June 2025 Electric Power Monthly archives.

## Contents

- `eia/june2024.zip`, `eia/june2025.zip`: original official monthly issue workbooks.
- `state_month_fuel_vintage_comparison.csv`: 51 state/DC rows × 11 measures, preserving suppressed/non-numeric values as missing.
- `summary.json`: paired revision and partial source-share displacement summaries, plus archive/output SHA-256 digests.
- `../../../scripts/audit_actual_monthly_vintages.py`: parser and comparison code.

## Reproduce

From the repository root:

```bash
python scripts/audit_actual_monthly_vintages.py \
  --earlier docs/evidence/actual-monthly-vintage-audit-r1/eia/june2024.zip \
  --earlier-year 2024 \
  --later docs/evidence/actual-monthly-vintage-audit-r1/eia/june2025.zip \
  --later-year 2025 \
  --observation-year 2024 \
  --month April \
  --output-dir docs/evidence/actual-monthly-vintage-audit-r1
```

The result is a release-vintage revision audit. It does not score hourly CIF forecasts or prove that input revision changes model accuracy.
