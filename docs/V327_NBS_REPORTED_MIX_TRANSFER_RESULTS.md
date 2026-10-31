# V327 results: actual NBS monthly mix for provincial annual-factor transfer

**Run date:** 2026-10-30  
**Protocol:** [V327 protocol](V327_NBS_REPORTED_MIX_TRANSFER_PROTOCOL.md)  
**Reproduction:** `python scripts/v327_nbs_mix_transfer.py`  
**Outputs:** `results/v327-nbs-mix-transfer-r1/`

## Question and method

V300/V301 demonstrated annual province-level LOPO transfer using lagged Ember source mix. V327 replaces the estimated mix with NBS-derived monthly generation by thermal, hydro, nuclear, wind, and solar, while retaining the same official MEE annual provincial CO₂-factor target, one-year lag, nonnegative linear mix model, and LOPO structure. The model is trained on 2021 target rows for the 2022 test and on 2021–2022 target rows for the 2023 test. For each held-out province, every training row from that province is excluded.

The public monthly table is above-designated-size coverage. February is the cumulative Jan–Feb report. Tianjin hydro is missing; it is represented by the remainder after subtracting its observed thermal/nuclear/wind/solar from reported total, never as zero. The five groups are normalized to absorb small report rounding gaps. This tests a real public Chinese feature source; it does not convert the target into an hourly or generation-side factor.

## Results

| Test target | Method | MAE | RMSE | P90 absolute error | MAE improvement vs training mean |
|---|---|---:|---:|---:|---:|
| 2022 | Training-province mean | 0.1207 | 0.1651 | 0.2116 | — |
| 2022 | NBS lagged five-group mix | **0.0570** | **0.0738** | **0.1207** | 52.8% |
| 2022 | Ember lagged five-group mix | 0.0598 | 0.0814 | 0.1305 | 50.5% |
| 2023 | Training-province mean | 0.1121 | 0.1510 | 0.1895 | — |
| 2023 | NBS lagged five-group mix | **0.0465** | **0.0673** | **0.1177** | 58.6% |
| 2023 | Ember lagged five-group mix | 0.0493 | 0.0770 | 0.1217 | 56.1% |
| pooled 2022–2023 | Training-province mean | 0.1164 | 0.1582 | 0.2081 | — |
| pooled 2022–2023 | NBS lagged five-group mix | **0.0517** | **0.0706** | **0.1197** | 55.6% |
| pooled 2022–2023 | Ember lagged five-group mix | 0.0545 | 0.0792 | 0.1288 | 53.2% |

The NBS mix has 5.1% lower pooled MAE and 10.9% lower pooled RMSE than the same-split Ember mix, and it is better in both target years by point estimate. However, the mean province-paired absolute-error difference (NBS minus Ember) is −0.00280 kgCO₂/kWh with 95% province-bootstrap interval [−0.00965, +0.00194]; only 14/30 provinces favor NBS. Thus the predeclared paired-evidence threshold is **not met**. Both methods robustly beat the training mean in these two temporal tests, but the incremental NBS-versus-Ember advantage is not statistically resolved.

## What changed and what did not

This is the first direct cold-start transfer evidence in this sequence using the public NBS-derived monthly generation categories rather than Ember's historical estimate. It shows that the real reported mix preserves the V300/V301 annual-level signal and has favorable point estimates in both tested years. That is useful progress toward a deployable China data path.

It is not an hourly CIF breakthrough: labels are only two forward annual vintages, the MEE factor is consumption-accounting and includes cross-province effects, the feature table covers above-designated-size enterprises, and historical issue/revision vintages are not verified. The incremental advantage over Ember is modest and has an uncertainty interval spanning zero. It does not show hourly shape, prediction interval coverage, or threshold-risk skill.

## Next experiment

The next frozen model will account explicitly for the known scope mismatch. Use lagged NBS monthly category shares multiplied by the province's coverage ratio (monthly above-size generation / annual all-plant generation), plus an unobserved-remainder share equal to one minus that ratio. Fit an additional nonnegative coefficient for the remainder group and compare against V327's normalized five-group mix on identical LOPO folds. This tests whether an explicit unobserved supply pool improves annual level transfer; it does not assume that the missing plants have zero or the same technology mix.

## Limitations and reproducibility

- MEE targets are annual consumption-side grid factors, not hourly production CIF.
- Monthly NBS-derived generation is above-size coverage; annual all-plant generation is a different scope.
- V320 documents missing Tianjin hydro and small rounding discrepancies; feature handling is explicit in `nbs_feature_audit.csv`.
- Two test years from the same 30-province panel are limited temporal evidence; province bootstrap intervals do not make years independent.
- No official/NBS as-of vintages are available here, so the lag is by statistical year, not proven release time.

Per-province predictions, coefficients, input hashes, and scoring metrics are in `results/v327-nbs-mix-transfer-r1/`.
