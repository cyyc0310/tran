# V305：识别区域进口碳因子需要最少几条线路电量？

Frozen 2026-10-30 before analysis. This is an exact linear-identification calculation conditional on the V304 2023 regional flow graph and margins; it is not a claim that the required flow observations have already been acquired.

## Question

V304 constructs a transportation polytope for 13 listed regional routes. Determine the minimum number of additional exact bilateral route-flow measurements needed to identify (a) each receiving region's imported-electricity source-factor numerator, and (b) all receiving regions' import-weighted average factors simultaneously.

## Frozen calculation

For listed-route flow vector `x`, let `A x = b` encode all seven exact origin and destination margins. For destination `d`, define coefficient vector `c_d` as the official MEE/NBS source-region factor on routes entering `d` and zero elsewhere. Inbound volume is fixed by `b`, so identifying `c_dᵀx` is equivalent to identifying the destination's import-weighted source factor.

Enumerate every subset of 0, 1, or 2 of the 13 route flows as candidate observations. A subset identifies a target functional iff its coefficient vector lies in the row span of `A` plus the selected unit rows. Report the smallest subset size for each receiver and jointly across all receivers, plus all minimal subsets. Independently verify the LP intervals collapse to a point for each identified functional after augmenting the equalities.

The calculation assumes exact margins, exact source factors, exact route values, and the table's listed route set as exhaustive. It makes no measurement-noise or release-time claim. A pair with full column rank is a sufficient check for complete matrix identification, but the primary target is the imported-factor functional, which may require fewer measurements.

## Interpretation

This provides a formal lower/upper count of missing flow scalars under the declared 2023 regional information budget. It does not imply those scalars are the same minimum for the 30 provincial grid problem, for later years, for hourly carbon intensity, or under uncertain inputs. A later experiment must obtain the selected route volumes independently and test whether the resulting factor bounds are calibrated against held-out labels.

## Sources

Use the data and primary sources frozen in [V304](V304_SHARP_REGIONAL_IMPORT_FACTOR_BOUNDS_PROTOCOL.md): MEE/NBS 2023 regional factors and Quan et al. (2025) regional route support and margins.
