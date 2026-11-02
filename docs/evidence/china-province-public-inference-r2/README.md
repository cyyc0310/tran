# China all-province public-month inference evidence

This is a label-free application of the frozen DE/GB/US model package to the 31 provinces in the sealed NBS monthly panel.

- `predictions.csv`: 31 provinces × 8 reference methods × 24 hourly predictions.
- `public_features.csv`: the input features and support coordinates used at each forecast hour.
- `accepted_months.csv`: provincial monthly generation values admitted by the `t−2` input contract.
- `summary.json`: per-province acceptance, eligible-month counts, and coordinate-support diagnostics.
- `provenance.json`: SHA-256 hashes for model, panel, source manifest, input seal, code, and outputs.

The forecasts contain no target hourly labels and have no China accuracy or coverage validation. The input panel is a current snapshot, not a recovered historical as-issued vintage.
