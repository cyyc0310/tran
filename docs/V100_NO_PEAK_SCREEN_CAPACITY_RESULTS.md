# V100: coarse-capacity identification survives removal of hourly peak screening

## Why this reanalysis matters

V90's primary cohort required lagged capacity to exceed observed target-month hourly peaks. That criterion uses the same high-frequency outcomes whose risk set is being evaluated. V100 retains all candidate months with at least 80% hourly coverage, removes the hourly peak screen, and uses only pre-outcome monthly generation versus lagged annual capacity to determine whether the capacity input is even compatible with the monthly totals. Peak violations remain in the scored cohort and can cause coverage failure.

## Cohort accounting

The frozen 25-BA roster yields **589** region-month candidates with at least 80% hourly coverage in 2024–2025. For 513 months, monthly group MWh fit within the lagged annual thermal/nuclear/hydro/wind/solar hourly caps; 76 fail this monthly-only feasibility check and are not assigned a capacity-constrained interval. Treat those 76 as unavailable/abstaining capacity inputs, not as excluded successful predictions. Among the 513 feasible months, 358 also pass the retrospective hourly peak screen; **155 do not pass and are retained in the main V100 results**. Twenty-two regions contribute at least one feasible month; three do not.

## Results: equal weight per region

All rows use the same 513 feasible months for the monthly-margin and lagged-coarse-capacity comparison. Region-level metrics are averaged equally, matching V90's aggregation convention.

| τ | Method | Mean K-set width (h) | Positive certificate, ≥1% | Negative certificate | Abstention | K-set coverage | Certificate errors |
|---:|---|---:|---:|---:|---:|---:|---:|
| 100 | Monthly fuel margins | 553.5 | 4.9% | 18.2% | 76.9% | 100% | 0 |
| 100 | Lagged coarse capacity | 423.5 | 75.9% | 18.2% | 5.9% | 100% | 0 |
| 300 | Monthly fuel margins | 553.5 | 4.9% | 18.2% | 76.9% | 100% | 0 |
| 300 | Lagged coarse capacity | 439.7 | 48.7% | 18.2% | 33.1% | 100% | 0 |

The more conservative 155-month peak-failing subset is covered as well: its region-equal τ=100/300 mean widths are 502.5/493.1 h, with no certificate errors. On the original 358 peak-passing subset, V100 reproduces V90's region-equal widths (419.0 h at τ=100; 441.7 h at τ=300) and zero certificate errors. Thus the narrowing is not created by the peak screen, although the peak screen does make the interval narrower.

If the 76 monthly-energy-infeasible months are counted as capacity-input abstentions, V100's region-equal capacity-method positive certification among all 589 coverage-qualified candidates is 66.3% at τ=100 and 42.5% at τ=300 for the ≥1% target; abstention correspondingly rises to 17.9% and 41.6%. The conditional K-set coverage remains 100% on the 513 months where the capacity contract is feasible. The data-availability loss is therefore visible and material (12.9% of candidates).

## Interpretation

This is a stronger retrospective mechanism result than V90: lagged coarse annual capacity narrows the monthly-margin identified set beyond the screened subset, while maintaining observed-K containment on every energy-feasible month including those that violate an hourly peak cap. It still does **not** establish a future-available Chinese capacity table, cross-region transport, a learned CIF forecast, or real-CIF calibration. Input-hourly consistency is checked using monthly totals for eligibility; the hourly target is used only to score coverage and certificates. The EIA balancing areas were previously opened and the label is a fixed-EF generation proxy.

The research candidate has moved from “capacity helps on a hindsight-screened subset” to “coarse capacity remains useful under a pre-outcome monthly-feasibility contract, with explicit abstention for incompatibility.” That is a defensible benchmark/partial-identification contribution candidate, but still not an algorithmic breakthrough. The next gate is a genuinely untouched system or temporal block, with the information contract and all capacity-mapping failures frozen before scoring. Chinese validation still needs a matched province-year capacity source and a valid hourly reference.

## Reproducibility

- Protocol: `docs/V100_NO_PEAK_SCREEN_CAPACITY_PROTOCOL.md`
- Script: `scripts/v100_no_peak_screen_capacity.py`
- Frozen output: `results/v100-no-peak-screen-capacity-r1/`
- Protocol SHA-256: `6aa32f4e126cc5ae8d038807a1ace464c50a843764c8c2761d31adb3636b99f5`
- Script SHA-256: `1ad7fb4cff5b009091561ad178ebd97410288b59ebceda644583c88b523378d1`
- Summary SHA-256: `54388ae50ac3cb692c799a12cae725332ae6ebc373b0d53489aec660c97fd091`
