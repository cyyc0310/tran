# V331 results: simple feature support does not explain when regularization helps

Run date: 2026-10-30. Frozen design: [V331 protocol](V331_COLDSTART_SUPPORT_DIAGNOSTIC_PROTOCOL.md). Outputs: `results/v331-coldstart-support-diagnostics-r1/`.

## Results

Across 60 province-year predictions in seven held-out regions, ridge reduced mean absolute error by 0.0232 kgCO₂/kWh on average and improved 53.3% of province-year cases. None of the three training-only scores cleanly identified which cases benefited:

- Nearest training-composition Hellinger distance had Spearman ρ=0.10 with ridge gain; its region-cluster bootstrap 95% interval was [−0.32, 0.35], and AUC for positive ridge gain was 0.50.
- Regularization adjustment magnitude had ρ=0.28 (interval [−0.69, 0.72]) and AUC 0.62.
- NBS-versus-Ember prediction disagreement had ρ=0.35 (interval [−0.14, 0.66]) and AUC 0.64.

The largest apparent signal is model disagreement, but its uncertainty remains broad across only seven region clusters. Mean nearest-feature distance is not aligned with ridge benefit: Southwest has the largest mean distance (0.156) while ridge worsens its MAE; Northwest has a moderately high distance (0.119) and large ridge gains. A simple distance threshold would therefore misclassify important cases.

## Decision

Do not build an adaptive province-level gate from these diagnostics. The evidence is too sparse and region-dominated. NBS–Ember disagreement may be retained as a hypothesis for a future independent dataset, but it is not a validated uncertainty score. The current findings support a narrower conclusion: weak ridge regularization repairs two regional failure modes while harming others, and ordinary composition distance does not tell us where that correction is safe.

All targets remain annual MEE consumption-side factors. These results do not establish hourly CIF accuracy or interval coverage. The next useful data increment is a paired local source/transfer descriptor or independent hourly generation-side labels; repeatedly tuning the same seven-region sample is unlikely to resolve the identification gap.

