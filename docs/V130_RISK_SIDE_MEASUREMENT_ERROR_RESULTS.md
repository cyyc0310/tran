# V130 — Risk-Side MWh Error under EF-Support Uncertainty

## Corrected result

The formal result is the corrected r2 run. On all 358 frozen region-months (2,148 threshold/rate rows), the risk-side interval passed zero-error equality with V129, nominal-K containment, monotone expansion as reporting error increased, and nesting within the coarse thermal-total outer set: **0 assertion failures**.

| τ | Side-field error | Risk-side + coarse capacity width | Detailed fuel + same capacity width | Negative certificate at ≥1% |
|---:|---:|---:|---:|---:|
| 600 | 0% | 232.6 h | 222.1 h | 56.0% |
| 600 | 10% | 235.2 h | 222.1 h | 56.0% |
| 900 | 0% | 162.4 h | 148.6 h | 70.3% |
| 900 | 10% | 164.1 h | 148.6 h | 70.3% |

At 10% side-field error, width grows only 2.6 h at τ=600 and 1.7 h at τ=900. The risk-side set remains about 58.2% and 66.1% narrower, respectively, than the broad thermal-total-plus-capacity set. At τ=600 the small positive-certificate rate disappears by 5% reporting error; negative decisions remain unchanged. At τ=900 there are no positive certificates at any tested error, and the negative/abstention rates remain stable.

The error-robust side interval is 13.1 h wider than detailed fuel margins at τ=600 and 15.5 h wider at τ=900 under the 10% stress. This is a modest cost for compressing multiple monthly fuel fields into one threshold-indexed MWh statistic, within this oracle coarsening experiment.

## Audit note

The first execution (r1) was not used for conclusions. Its comparison variable selected the detailed no-capacity endpoint while the risk-side arm included capacity, producing 798 false nesting failures. The failed assertion exposed this indexing error. The runner now selects the detailed robust set with the same coarse capacity, uses a new output directory (r2), and the corrected run reports zero failed assertions. Both outputs are retained so the correction is auditable.

## Interpretation

V129's candidate information compression survives this ±10% measurement-error sensitivity under the tested EF intervals: the field's error widens the interval slightly without erasing the large gain over the coarse monthly budget. This makes a **risk-threshold-indexed, factor-support-aware monthly sufficient-statistic map** a credible paper mechanism candidate. It is not yet a verified algorithmic breakthrough: the threshold-side values are oracle-aggregated from U.S. hourly fuel labels, the EIA cohort is retrospectively capacity-screened, and neither the measurement-error band nor the EF intervals are estimated from China publication data. The remaining decisive test is whether a point-in-time Chinese field can supply the same statistic, followed by a truly independent hourly-label or authorized-data validation.

## Reproduction

- Protocol: [V130_RISK_SIDE_MEASUREMENT_ERROR_PROTOCOL.md](V130_RISK_SIDE_MEASUREMENT_ERROR_PROTOCOL.md).
- Runner: [v130_risk_side_measurement_error.py](../scripts/v130_risk_side_measurement_error.py).
- Formal corrected output: `../results/v130-risk-side-measurement-error-r2/`.
- Superseded diagnostic output retained for audit: `../results/v130-risk-side-measurement-error-r1/`.
