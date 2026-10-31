# V332 results: detailed fuel groups worsen whole-region transfer on average

Run date: 2026-10-30. Frozen design: [V332 protocol](V332_FUEL_GRANULARITY_REGION_COLDSTART_PROTOCOL.md). Outputs: `results/v332-fuel-granularity-region-coldstart-r1/`.

## Results

Under the same Ember mix snapshot and seven whole-region holdouts, full eight-source NNLS had pooled province-weighted MAE 0.0801 kgCO₂/kWh, compared with 0.0695 for the corresponding five-group coarse Ember model and 0.0746 for NBS five-group NNLS. Equal-region MAE was 0.0792 for full8, 0.0715 for Ember coarse5, and 0.0757 for NBS5. The full8 model's error was higher in both target years: 0.0866 vs 0.0754 in 2022 and 0.0737 vs 0.0636 in 2023 (province-weighted).

Full8 improved the region-averaged MAE in only 3/7 areas. Its paired region-equal MAE difference from coarse5 was +0.00768 kgCO₂/kWh (full8 worse), with a seven-region bootstrap interval [−0.01676,+0.03993]. Fuel detail helped Northwest (difference −0.0350) and Northeast (−0.0252), but worsened North sharply (+0.0949) and had small negative effects in Central, East, South, and Southwest. P90 full8 was slightly lower in the pooled province-weighted score (0.1318 vs 0.1546), while its MAE and RMSE were worse; the tail/center tradeoff does not overturn the mean-transfer result.

## Interpretation

Splitting thermal into coal, gas, other fossil, and bioenergy does not by itself solve region cold start. It reduces error in Northwest and Northeast but destabilizes another region. This mirrors V330: correcting selected failures is possible, yet global improvements are not broad enough. More input categories increase the number of group factors to estimate while the training set remains 23–27 provinces; with only two target years, transfer coefficients are weakly identified.

V333 will test nested train-only regularization for both five- and eight-group Ember schemas. If detailed groups still fail after that, stop adding fuel categories as a standalone fix. These remain retrospective estimated generation mixes predicting annual consumption-side MEE factors, not user-table validation or hourly CIF evidence.

