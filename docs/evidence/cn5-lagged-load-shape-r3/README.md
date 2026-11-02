# CN5 lagged load-shape transfer r3

- Protocol: [CN5_LAGGED_LOAD_SHAPE_TRANSFER_PROTOCOL_20261102.md](../../CN5_LAGGED_LOAD_SHAPE_TRANSFER_PROTOCOL_20261102.md)
- Results: [CN5_LAGGED_LOAD_SHAPE_TRANSFER_RESULTS_20261102.md](../../CN5_LAGGED_LOAD_SHAPE_TRANSFER_RESULTS_20261102.md)
- Runner: `scripts/run_cn5_lagged_load_shape_transfer.py`
- Machine outputs: `results/cn5-lagged-load-shape-r3/`

This is a nine-country development LOCO run on fixed-EF generation-side proxies, not blind confirmation. It compares strict five-category lagged monthly generation features (M) with M plus the previous-calendar-year same-month load profile (M+L12). Nine target predictions were sealed before scoring. The profile uses 48 local weekday/weekend-by-hour bins, normalized to the previous-year month mean. Sampling resolution was inferred separately by country and year so 15-minute and hourly portions are handled consistently.

The candidate did not pass its frozen gain gate: macro MAE increased 0.67%, coverage was essentially unchanged and interval score worsened 0.38% versus the same-sample M factor head. Strong ratio recomposition remained best on point MAE. Full per-month results, hashes, seals and resolution audit are in the machine output directory.
