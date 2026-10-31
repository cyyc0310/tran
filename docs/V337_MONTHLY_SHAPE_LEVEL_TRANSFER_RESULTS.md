# V337 results: monthly seasonal shape does not transfer beyond annual mix reliably

Run: `results/v337-monthly-shape-level-transfer-r1/`, under the frozen [protocol](V337_MONTHLY_SHAPE_LEVEL_TRANSFER_PROTOCOL.md). The test adds first-harmonic monthly source-share coefficients to the existing annual mean composition and reselects ridge regularization inside each whole-region outer fold.

## Results

For NBS, annual mean plus seasonal shape lowers pooled region-equal MAE from 0.05450 to 0.05294, but this combines a worse 2022 result (0.05920→0.06103) with a better 2023 result (0.04980→0.04486). Only 3/7 regions improve; the paired region bootstrap for the pooled MAE difference is [−0.00836,+0.00302]. For Ember, shape slightly worsens pooled MAE 0.05112→0.05161; 2022 worsens 0.05994→0.06111, while 2023 is nearly unchanged. Only 3/7 regions improve, and its interval [−0.00094,+0.00233] crosses zero.

The NBS seasonal features exclude February because the released value is Jan–Feb cumulative; the annual baseline retains the already-audited V327 construction. This means the test is a conservative, low-dimensional shape check rather than a reconstructed twelve-month profile.

## Interpretation

The month-to-month generation mix signature does not yield a stable cross-region increment for the annual MEE consumption-side target. Stop expanding seasonal bases against these two annual forward folds. Shape/level separation may still be valuable for hourly risk outcomes, but that requires independent hourly labels and a matching monthly table. This experiment is not evidence for hourly CIF and does not provide interval coverage.
