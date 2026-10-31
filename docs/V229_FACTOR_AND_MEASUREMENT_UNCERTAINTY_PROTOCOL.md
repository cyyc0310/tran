# V229 Frozen protocol: uncertainty stress for factor-span channels

Date: 2026-10-30. Status: frozen before scoring V229 uncertainty envelopes on the V226 cohort.

## Question

How quickly does V228's exact factor-span reconstruction lose interval efficiency when (a) hourly aggregate readings are only known within a bounded relative error and (b) fixed fuel emission factors are only known within a multiplicative stress box?

This is a sensitivity analysis on the already opened V226 EIA cohort. It is not a blind confirmation and the uncertainty magnitudes are stress settings, not empirical China measurement or EF uncertainty distributions.

## Cohort and selected channels

Reuse V228's V226 cohort and target/month eligibility exactly: 2026 July–September, BAs outside V224/V225 development roster, inclusion by completeness and source availability, 26 BAs/74 BA-months expected. Use the same target-month margins and V228 factor-span minimum channel set; total generation is one measured field and selected group channels use V228's saved linear coefficients.

## Frozen uncertainty envelope

For each hour, let reported group fields be `z`, V228 coefficients `a`, nominal generation `G`, and nominal center `c`. Set measurement tolerance `r_m ∈ {0, 0.005, 0.01, 0.05}` independently for total and each selected group field, interpreted as a deterministic relative error bound around the reported magnitude. Set factor stress `r_e ∈ {0, 0.10}` independently for positive-EF categories: `e_j ∈ [e_j(1-r_e), e_j(1+r_e)]`; zero-EF categories stay zero. The ±10% setting follows V103 only as a stress box, not as a claimed real-world EF uncertainty range.

The reported-field nominal numerator is `N̂=aᵀz`. Bound the numerator perturbation from group-channel error by `δ_N,m = r_m Σ_i |a_i z_i|`. Bound the EF contribution by `δ_N,e = Σ_j r_e e_j min(M_j, G(1+r_m))`, where `M_j` is the known monthly margin. Use:

`N_lo=max(0,N̂−δ_N,m−δ_N,e)`, `N_hi=N̂+δ_N,m+δ_N,e`,
`G_lo=G(1−r_m)`, `G_hi=G(1+r_m)`,
`C_lo=N_lo/G_hi`, `C_hi=N_hi/G_lo`.

These are conservative pointwise outer bounds; they do not exploit cross-hour coupling beyond the monthly per-fuel cap. At `r_m=r_e=0`, the interval must collapse to V228's exact CIF to `1e-8 gCO2/kWh`; otherwise stop.

## Scoring and comparisons

For each day, nonconformity is the maximum across its 24 hours of `max(|C_lo-c|, |C_hi-c|)`. Calibrate a 90% finite-sample radius from 12 random target days, 100 fixed-seed repetitions per BA-month, and evaluate the remaining days against the nominal fixed-factor proxy labels. Use identical sampled/test days for:

1. V228 exact factor-span point scores;
2. robust factor-span envelopes for every `(r_m,r_e)` pair;
3. legacy V226 `total+thermal+coal+gas` feasible-bound scores;
4. exact full-eight local scores;
5. source-only leave-target-BA-out scores.

Report hourly and simultaneous-day coverage, interval width, Winkler-90, BA-average tail coverage, and paired BA-bootstrap uncertainty. Coverage is scored against the nominal proxy label; the robust score additionally bounds the declared stress set, whose true EF is not observed.

## Interpretation gates

- At least 30 BA-months and 10 BAs remain eligible.
- Zero-uncertainty envelopes match exact V228 scores and outcomes within `1e-8`.
- Show the full width/coverage/Winkler sensitivity surface. Do not choose the best uncertainty level after observing outcomes.
- A stress cell is operationally useful only if macro and every BA mean simultaneous-day coverage are at least 90% and its Winkler is lower than the legacy four-field baseline; otherwise report the tradeoff/failure.
- No claim that ±10% EF or relative measurement error matches China reporting uncertainty.

## Limitations

The denominator and group error model is a deterministic relative envelope, not a probabilistic noise model. The bound is conservative and does not include factor correlation, fuel-specific EF heterogeneity within categories, imports/storage/network flows, or revision uncertainty in monthly margins. All results remain fixed-factor EIA proxy reconstruction, not measured emissions or Chinese validation.
