# V269 protocol: asymmetric fuel-factor support from measured CEMS

Date: 2026-10-30. This tests whether asymmetric empirical support intervals change the V265/V268 threshold gate.

## Frozen design

- Reuse measured 2025 CEMS records and the same coal/gas/oil harmonization as V267.
- Aggregate each state-quarter factor and gross-generation weight by fuel group.
- For each fuel-quarter, find the narrowest factor interval covering at least 90% of state-quarter gross generation (weighted shortest interval).
- Apply these lower/upper fuel-factor endpoints to the same 16 already-open V240 EIA BA-month fuel compositions. Divide by all-source generation, including zero-direct-EF sources.
- Report monthly mean-factor intervals and threshold crossings at 500.0001 g/kWh. Compare with V268's symmetric CEMS interval and the uniform project ±10% reference.

## Limits

The shortest interval is an empirical U.S. state-quarter support summary. It is not a confidence or prediction interval, and applying it to EIA BA mixes is a cross-system transport stress. It does not estimate Chinese fuel-factor uncertainty or validate predicted hourly CIF coverage.

