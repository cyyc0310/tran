# V23: four local hourly labels do not resolve most tail-count ambiguity

Sol, 2026-10-27. This is a source-only historical identification experiment, not a CIF forecast, probability calibration result, or new-region confirmation. It asks how much four sparse local hourly CIF measurements can resolve when the method is already granted unusually rich contemporaneous local information. That base package—six-hour H/G and all 24 hourly G—is outside the user's stated China monthly/yearly public-statistics budget.

## Protocol, amendment, and validation

The fixed contract is in [V23 protocol](LOCAL_HOURLY_QUERY_BUDGET_V23_PROTOCOL.md). It uses the first and last valid day of each inherited V20 source region: 14 days per source, seven previously used sources, 98 source-days total. Thresholds are strict `C > 300` primary and `C > 100` sensitivity. Budgets are 0/1/2/4 local CIF points, with exact or 10 g/kWh quantized labels. The same four policies were compared: fixed clock, hash-seeded random, widest currently feasible hourly-CIF range, and one-step minimax K-span.

The first attempt preserved its freeze and stopped before writing outcomes because the inherited date key included UTC midnight while the isolated reader used its date portion. This source-only row access is disclosed in [boundary amendment](LOCAL_HOURLY_QUERY_V23_BOUNDARY_AMENDMENT.md). R1 completed, then review found its prospective quantized-bin enumerator allowed impossible adjacent bins at exact 10-unit boundaries. R1 remains unchanged. [R2 amendment](LOCAL_HOURLY_QUERY_V23_QUANTIZATION_AMENDMENT.md) corrected only that minimax branch enumeration; R2 was written to a new directory. R1 and R2 aggregate results are identical.

The R2 producer uses exact rational arithmetic over the recorded fuel decimals. It retained all 14 days per source and generated 896 day×threshold×strategy×precision×budget records. A post-run table audit confirmed: (1) each realized K lies in every reported feasible K set; (2) adding queries never enlarged a feasible K set; (3) zero wrong certificates; and (4) outputs and parent hashes are bound in `results/local-hourly-query-v23r2/completed.json`. The audit explicitly is not an independent proof of formula completeness. `py_compile` and `git diff --check` passed; no test suite was run. See [post-run audit](../results/local-hourly-query-v23r2/postrun-audit.md).

## Main result: tau = 300 g/kWh

Numbers below are source-macro averages: average within each source's two fixed days, then equal weight over seven sources. “Unresolved” means the feasible K set still contains both `K=0` and `K>=1`. Certificates are exact relative to this declared finite-dimensional information class, not guarantees for measured real-world data.

| Query policy | Labels | b=0 unresolved | b=4 unresolved | b=4 unresolved for `K>=6` | b=4 mean K-span |
|---|---|---:|---:|---:|---:|
| Fixed clock | exact | 50.0% | 35.7% | 71.4% | 12.00 |
| Seeded random | exact | 50.0% | 50.0% | 71.4% | 12.00 |
| Widest feasible CIF | exact | 50.0% | 35.7% | 71.4% | 11.93 |
| One-step minimax K-span | exact | 50.0% | 35.7% | 71.4% | 12.07 |
| Fixed clock | 10 g/kWh interval | 50.0% | 50.0% | 71.4% | 12.29 |
| Seeded random | 10 g/kWh interval | 50.0% | 50.0% | 71.4% | 12.07 |
| Widest feasible CIF | 10 g/kWh interval | 50.0% | 50.0% | 71.4% | 12.36 |
| One-step minimax K-span | 10 g/kWh interval | 50.0% | 50.0% | 71.4% | 12.64 |

The best adaptive policy did not improve on the best non-adaptive policy under the registered gate: b=4 reduced the unresolved rate by 14.3 percentage points with exact labels for some fixed policies, but the adaptive gain over the best fixed schedule was **0 pp**; with 10 g/kWh intervals, the gain was also **0 pp**. The registered required gain was at least 10 pp in both precision modes. No policy incorrectly certified an event status, but under this rich oracle base and only four local points, the K set still spans about half of the possible 24-hour count range on average. For `K>=6`, even exact b=4 queries leave 71.4% of source-macro cases unresolved.

The sparse improvement is source-specific: under exact labels, fixed clock/widest/minimax reduce ambiguity on the two sampled FPL and ISNE days; random does not. With 10 g/kWh intervals, none of these policies improves the primary unresolved rate. The ten-source literature or source folds must not be inferred from these 14 days; regions and days are highly dependent.

## Threshold 100 sensitivity

At `tau=100`, the b=0 base already decides `K=0` versus `K>=1` for most cases: source-macro unresolved is 14.3%. At b=4, widest-feasible-CIF lowers this to 7.1% under both exact and 10 g/kWh interval labels; the other policies stay at 14.3%. This is a one-day-per-source-macro-sized change in the 14-day source sample, under a threshold with many forced events. It does not pass the primary tau=300 gate and does not establish a stable adaptive selection benefit.

## What this says about the minimum local information question

The result does not identify a universal minimum number of local hourly measurements. It does narrow the question for this particular risk and measurement class: with exact historical H/G and G24 already known, four hourly CIF points do not reliably decide whether a day had any `C>300` hour, and quantization to 10 g/kWh removes the small exact-label gains. The remaining ambiguity comes from the unqueried hourly carbon allocation even after giving the diagnostic full hourly generation. This is consistent with the V19/V20 counterexamples, but is not a future probability statement.

The user's Chinese deployment budget is monthly or annual public fuel statistics, so neither this local-label experiment nor its G24/HG base can be used as a deployed feature. It instead shows that “send four local hourly labels” is not an obviously sufficient minimal add-on under the tested risk function. A useful next study would need an actually obtainable, as-of local measurement or structural bound that rules out fuel substitution; any such channel must be valued with its acquisition source, delay, precision, and label derivability stated. Active query selection and minimax measurement design are established general ideas; the current small diagnostic is not an algorithmic novelty claim.

## Artifacts

- [R2 protocol](LOCAL_HOURLY_QUERY_BUDGET_V23_PROTOCOL.md)
- [R2 code](../scripts/local_hourly_query_v23r2.py)
- [R2 per-day identification sets](../results/local-hourly-query-v23r2/day_results.csv)
- [R2 summary](../results/local-hourly-query-v23r2/summary.csv)
- [R2 completion and hashes](../results/local-hourly-query-v23r2/completed.json)
- [R1 boundary amendment](LOCAL_HOURLY_QUERY_V23_BOUNDARY_AMENDMENT.md)
- [R2 quantization amendment](LOCAL_HOURLY_QUERY_V23_QUANTIZATION_AMENDMENT.md)
