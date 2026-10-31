# V299 protocol: independent LP audit of the monthly-CIF anchor set

Frozen 2026-10-30 after V297/V298. Select the first eligible V97-cohort month for each of the first 10 V90 regions in the frozen roster. Retain lagged coarse technology-group capacities, fixed EIA proxy factors, thresholds τ=600/900, and anchor errors ±0 and ±0.25 gCO₂/kWh.

Implement the two-block feasibility problem independently with `scipy.optimize.linprog`, explicitly optimizing fuel-by-fuel event and complement allocations under: (i) thermal-total and clean-source monthly margins, (ii) the monthly-CIF-derived threshold-excess budget interval, (iii) group capacity in each block, (iv) strict positive event excess, and (v) nonpositive complement excess. Enumerate every integer K from 0 through T. Compare the complete LP feasible K set exactly to V297's analytic set for every selected BA-month, threshold, and error band.

Any discrepancy is a method failure: preserve counterexamples, revise the derivation/implementation, and rerun this audit before interpreting V297/V298. Passing this audit only verifies the declared mathematical contract on the selected proxy rows; it does not establish forecasting coverage, real CIF validity, or China transfer.

## Execution amendment (2026-10-30)

The literal first 10 V90 roster entries include CHPD, which has no eligible row in the frozen V97 cohort, so no first eligible month exists for that region. To avoid substituting an unregistered month or changing the cohort, implementation selects the first 10 regions in the unchanged V90 roster order that are represented in the frozen V97 cohort. This is a cohort-availability rule, not an outcome-based selection; the selected roster is recorded in the output summary. The protocol hash therefore includes this amendment.
