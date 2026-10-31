# V127 — Independent ±20% Fossil-Factor Corner Stress

## Question

Does the V122 τ=900 frozen support-gap rule retain endpoint coverage and interval efficiency when coal, gas, oil, and other-fossil emission factors can vary independently by ±20%?

## Frozen procedure

- Reuse the V122 October month-to-date candidate cohort and the V111 source calibration roster. This is explicitly posthoc because October outcomes have already been inspected.
- Evaluate the nominal EF vector and all 16 independent low/high corners for coal, gas, oil, and other fossil. Other fuel factors stay fixed. This is a finite corner-scenario stress; it does not claim to enumerate every attainable hourly tail count at interior EF vectors.
- Recompute source τ=900 monthly-anchor scores and capacity-support gaps under all 17 scenarios. For each target region, exclude that region from the calibration-radius and support-boundary pools.
- Use the same α=10% region-level rank, q=20% support-gap rank, lagged capacity, and V92 sharp set. Keep the frozen rule unchanged: plain anchor∩capacity unless the gap exceeds the source boundary, then capacity.
- Report endpoint hulls over the 17 scenarios, coverage, widths, support-gap fallbacks, and the full 44-region availability denominator.

## Interpretation boundary

This is neither a continuous-box proof nor an independent confirmation. The corner hull is a sensitivity diagnostic; the target October labels are reused, and the factors are illustrative fixed proxies rather than measured China-specific factor intervals. A broader claim about all factor vectors inside the box would require an exact or conservative interior analysis.
