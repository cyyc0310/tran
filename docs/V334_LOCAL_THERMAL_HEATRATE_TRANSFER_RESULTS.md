# V334 results: local thermal coal-consumption rates improve coarse annual transfer

Run date: 2026-10-30. Frozen design: [V334 protocol](V334_LOCAL_THERMAL_HEATRATE_TRANSFER_PROTOCOL.md). Corrected run: `results/v334-local-thermal-heatrate-transfer-r2/`. The initial `r1` output omitted 2023 targets because the feature table stopped at 2022; it is preserved as `results/v334-local-thermal-heatrate-transfer-invalid-r1/` and must not be used. The r2 run includes both target years and the protocol's forward folds.

## Field audit

The *China Electric Power Statistical Yearbook 2024* reports province-level 6000-kW-and-above generation coal-consumption rates (table 2-41) and supply coal-consumption rates (table 2-43), in gce/kWh for 2019–2023. V334 manually transcribed 2020–2022 for the 30 MEE provinces and verified the used rows against the PDF text extraction. The downloaded source PDF SHA-256 was `6aebb42952155995cc43041de542516d8a37b9171db5d74575ea261d76bcf15c`. These values are an efficiency proxy, not a carbon factor, and their plant-size scope is not identical to NBS monthly production.

## Results

Each heat-rate model replaces the thermal mix share with `thermal_share × lagged_local_rate/300`; other four source shares remain unchanged. Under province LOPO, NBS five-group MAE fell from 0.05173 to 0.04381 using generation coal-consumption rate and 0.04393 using supply rate, with paired province bootstrap intervals including zero. The Ember five-group MAE fell from 0.05453 to 0.04732 and 0.04790 respectively. Gains appeared in both forward target years.

The harder whole-region holdout gives the more relevant result. Region-equal NBS MAE fell from 0.07567 to 0.06180 (generation rate) and 0.06178 (supply rate), a 18.3% reduction. Province-weighted NBS MAE fell 0.07461→0.06046/0.05942. Ember region-equal MAE fell 0.07148→0.06322/0.06425; province-weighted MAE fell 0.06951→0.06230/0.06255. Both years improved in aggregate.

The increment is not yet decisive across regions. NBS-generation-rate adjustment improved 4/7 regions; supply-rate adjustment improved 3/7. Paired region-bootstrap intervals crossed zero for both NBS rates (generation: [−0.0397,+0.0037]; supply: [−0.0414,+0.0042]) and for both Ember rates. Northwest improved sharply: generation-rate-adjusted NBS MAE fell from 0.195/0.161 to 0.101/0.084 in 2022/2023; Northeast improved from 0.113/0.116 to 0.096/0.085. The same adjustment slightly worsened East and Central in both years and had mixed effects in Southwest. It reduces, but does not eliminate, region-specific failures.

## Interpretation

This is the strongest China-specific information increment in the current annual cold-start track. It supports a two-part structure: generation mix describes source composition, while a local annual coal-consumption rate adjusts the thermal conversion layer. The result appears under strict whole-region label exclusion and survives either generation-rate or supply-rate definition. The two rate definitions perform similarly; neither is a universal winner.

It is not yet a novel prediction algorithm or an operationally legal forecast. The yearbook vintage postdates the target factors and historical as-of availability is not verified. Labels are annual MEE consumption-side factors; they are not hourly CIF, and the coal rate covers only larger plants. Cluster evidence is low-power with seven regions. A next experiment should use training-only nested blending to avoid applying the local rate adjustment at full strength where it hurts, then test on an independent hourly target set if the adjusted annual level still adds information there.

