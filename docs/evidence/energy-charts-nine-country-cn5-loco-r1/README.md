# Strict CN5 nine-country LOCO r1

Run from the repository root:

```bash
python scripts/run_energy_charts_nine_country_cn5_loco.py \
  --output results/energy-charts-nine-country-cn5-loco-r1
```

This is the corrected China-compatible five-group input contract. Public features contain only thermal, hydro, nuclear, wind, and solar generation shares plus calendar. Other renewables are excluded from target monthly shares and hourly-shape references, but remain in the fixed-factor proxy response denominator. The runner seals point predictions and source radii for all nine targets before scoring.

The durable evidence directory now mirrors the source results needed for the nonnegative-support diagnostic, including all nine `*_heldout_predictions.csv.gz`, all nine source-radius files, the seal, scoring tables, and audit metadata. The same artifacts remain in the local `results/energy-charts-nine-country-cn5-loco-r1/` run directory.

SHA-256:

| File | SHA-256 |
|---|---|
| `startup_hashes.json` | `1916d61383d54df849e1f11c2a0d2787e17c3ccd67ce6f1db161ed2853f1bd8b` |
| `input_audit.json` | `d9bff82e170abab186e1f6e87801c0e6f8bbf965fb03faa9588dcda2f8bd2e78` |
| `prediction_seal.json` | `13022484b244537074bcd9bafc7172fb59715b6abf0099b20b3856d4bfb31ec8` |
| `monthly_metrics.csv` | `af1086466ccb95c1852c1f99f655a876b3090a15df7286155249e3e1b0b634aa` |
| `country_metrics.csv` | `0084ba048c8c8f0ad8887e8b88970515815c16ca2120279e3d4cb14203f36f2f` |
| `macro_metrics.csv` | `920bdcc78ac8f5f3cbdcd7067de50a1d3339f0e33f627b7cb8b8f126670dc66c` |
| `status.json` | `593aa468b28b545ef39e032644af6283767274c9ea2ce08358e229463842f614` |
