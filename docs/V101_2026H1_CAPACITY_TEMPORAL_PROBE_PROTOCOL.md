# V101 protocol: frozen V100 capacity mechanism on 2026H1

Frozen 2026-10-28 before calculating capacity-set outcomes. Apply the exact V100 monthly-margin plus lagged coarse-capacity rule to 2026-01 through 2026-06, using 2025 EIA-860 capacity and the fixed 25-region V52 quality roster. Do not tune thresholds, factor values, cohorts, or constraints after scoring.

Include months with at least 80% complete hourly proxy coverage and classify monthly group-energy feasibility from the monthly fuel totals and 2025 capacities only. Never filter on hourly group peaks. Capacity-infeasible months count as unavailable/abstention in the full candidate denominator. On feasible months, report actual-K set containment, valid/invalid certificates, widths, and positive/negative/abstention rates against the same monthly-margin baseline. Use the same thresholds (100, 300 gCO2/kWh) and tail targets (1/5/10%) as V100; report region-equal and pooled summaries.

This is a method-specific temporal transport probe, not a sealed blind confirmation: the 2026H1 source hours were previously used by other prediction experiments in this repository. Regions were also opened in prior work. Label the evidence accordingly and do not select future methods on the 2026H1 score.
