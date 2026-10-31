# V385: Shanghai recent primary monthly mix and exact sharp K sets

Registered 2026-10-31 before running the solver. This is a focused follow-up to V378/V379: test whether the directly exported recent Shanghai mix changes exact threshold-hour identification sets, not just the preliminary mixture certificate.

## Inputs and frozen comparison

- User-provided NBS province-month exports for Shanghai thermal, wind, and solar. Hydro and nuclear are set to zero per the user's explicit statement. Read only nonempty `当期值`; do not convert February cumulative values into monthly observations.
- Use the 11 report months with direct current values: 2025-09 through 2025-12, and 2026-03 through 2026-09.
- Convert the CSV unit `亿千瓦时` to GWh by multiplying by 100, then to MWh. Confirm parsed values against V320's corresponding Shanghai rows before scoring.
- For each month compare (A) Shanghai's five-category local mix and (B) a national pooled five-category mix reallocated to the same Shanghai five-category total, preserving the local nonthermal proportions. The national mix is computed from provinces with complete five-category V320 values for that same report month.
- Keep the 2023 CEC installed-capacity proxies, month hours, direct factors, and exact V342 sharp-group solver identical across the two arms. Use thermal EF endpoints 400 and 1000 g/kWh; all other groups have direct EF zero; evaluate strict thresholds 100 and 300 g/kWh.
- Report exact K hull width, feasibility, and any paired difference. A capacity-infeasible arm is retained as infeasible, not repaired.

## Interpretation limits

This is one province and 11 recent report months. It is a conditional identification sensitivity, not a held-out forecast or empirical coverage test. V320 is an NBS-derived transcription; the Shanghai CSV is a direct user export. The capacity proxy is from 2023 and may be stale for 2025–2026. The model fixes unobserved above-scope generation at zero and excludes imports, hourly availability, within-thermal fuel mix, and hourly dispatch. Therefore any narrowing is conditional on a restricted reporting-scope model and cannot be described as whole-province CIF performance.

## Stop criteria

Do not call the experiment a method breakthrough unless local composition produces material, repeatable sharp-set improvement and remains useful after plausible scope-residual and factor-support uncertainty. No Chinese hourly label is available in this experiment, so no coverage claim is allowed.

## Frozen follow-up amendment: capacity-free control

The first run found all 11 national-mix counterfactuals infeasible under the lagged 2023 group-capacity proxies, while Shanghai-local mixes were feasible. Before inspecting any capacity-free risk widths, add a second arm that removes installed-capacity information from **both** local and national-mix variants. Implement this by giving each group a nonbinding hourly cap equal to its monthly energy margin (MWh per hour); this is only a mathematical monthly-margin control, not a physical capacity estimate. Keep the original 2023-capacity results frozen and report them separately. No tuning of EF endpoints, thresholds, or month cohort is permitted.
