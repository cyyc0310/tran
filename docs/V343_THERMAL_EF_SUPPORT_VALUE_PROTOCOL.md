# V343 value of narrowing the local thermal-factor support

Registered 2026-10-30 after V342 established a sharp K hull under the declared group-capacity contract. Keep the 330 complete province-periods, same five generation margins, same capacity arms, direct nonthermal factors zero, and threshold τ=300 g/kWh. Reuse the V342 exact fixed-factor solver. No hourly labels enter.

Evaluate fixed thermal factors 400, 500, 600, 700, 800, 900 and 1,000 g/kWh. Then quantify the hull for symmetric factor-support bands centered at 700: `[400,1000]`, `[500,900]`, `[600,800]`, and the point support `[700,700]`. For each band, use monotonicity in thermal EF to get the minimum K from the low factor and maximum K from the high factor. Compare the monthly-only, lagged 2022 capacity, and retrospective 2023 capacity arms.

The bands are a sensitivity curve, not calibrated probability intervals or information literally available in the user's monthly data. The point-factor case is an oracle diagnostic to determine how much uncertainty remains after an exact local thermal factor is known. Do not translate hull narrowing into prediction coverage or claim that NBS five-category fields identify the true local thermal factor.

Decision question: if exact thermal-factor knowledge still leaves a very wide τ=300 K hull, the dominant missing information is temporal dispatch/shape; if the factor support drives most of the hull, the next data request should target a legal local thermal fuel or heat-rate split. Neither conclusion is causal without independent hourly generation/CIF labels.
