# V127 — Independent ±20% Factor Corner Results

> V139 correction notice: the source support-calibration routine calls the formerly unguarded capacity solver. Its handling of monthly-energy-infeasible source inputs needs re-auditing before the archived mechanism results are reused. See [correction scope](CAPACITY_CONTRACT_CORRECTION.md).

## Result

All 16 independent coal/gas/oil/other-factor corners plus nominal were recomputed for τ=900. The source calibration scores, source support gaps, target anchor intervals, capacity intervals, intersections, and fallback decisions were numerically identical to the V126 common-mode run. The independent-corner frontier is therefore:

| Method | Endpoint coverage | Mean width |
|---|---:|---:|
| Capacity-only | 37/37 | 222.0 h |
| Plain anchor∩capacity | 37/37 | **120.8 h** |
| Frozen q=20% support-gap rule | 37/37 | 144.2 h |

The fallback again triggers for JEA and PACE. Since plain intersection already covers all 37 capacity-feasible target regions, the fallback adds 23.4 hours of average width without adding observed October coverage. Six rows have nonzero endpoint-hull upper K under the factor scenarios; both plain intersection and the frozen rule cover them.

## Why independent corners match the shared low/high stress

For any fixed hourly fuel allocation, increasing any fuel's EF cannot turn an above-threshold hour into a below-threshold hour. Thus the observed hourly tail count is coordinatewise nondecreasing in the four fossil EFs. Monthly mean CIF has the same monotonicity. For the physical feasible-set endpoints, every feasible schedule's K is ordered the same way, so the minimum and maximum feasible K endpoints are also bounded by the all-low and all-high vectors. V127's 16-corner enumeration verifies the finite implementation and shows that the target/calibration outputs equal the shared low/high run.

There is a remaining calibration detail: the score `|observed tail fraction − monthly-mean anchor|` can peak near the EF vector where the anchor crosses τ, even when the K endpoints themselves are monotone. V127 takes the maximum over corners, not over all interior vectors. V128 addresses this with a conservative continuous-box score upper bound; see [V128 results](V128_CONTINUOUS_EF_BOX_SCORE_BOUND_RESULTS.md).

## Interpretation boundary

This is a posthoc sensitivity analysis on the October labels already used in V122/V125. It is not an untouched spatial or temporal confirmation. It shows that independent factor corners do not change the tested endpoint frontier relative to the shared low/high stress, while identifying the anchor-crossing score as the remaining uncertainty-sensitive piece.

## Reproduction

- Protocol: [V127_INDEPENDENT_EF20_CORNER_PROTOCOL.md](V127_INDEPENDENT_EF20_CORNER_PROTOCOL.md).
- Runner: [v127_independent_ef20_corner_stress.py](../scripts/v127_independent_ef20_corner_stress.py).
- Output and hashes: `../results/v127-independent-ef20-corner-stress-r1/`.
