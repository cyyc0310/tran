# V148 — Quantile baseline numerical diagnosis

## Scope and environment

This is a read-only diagnosis of the PACE, threshold 900, seed 41, alpha 0.10 `direct_cqr_all` result in V142 versus `direct_common_all` in V145. It does not refit the full experiment or change either archived result. The diagnostic interpreter was `/opt/homebrew/bin/python3.12`: Python 3.12.13, NumPy 2.4.5, pandas 3.0.3, scikit-learn 1.8.0. All fits below used the frozen `GradientBoostingRegressor` parameters: 150 estimators, learning rate 0.05, depth 2, minimum leaf size 15, random state 20261028, and quantile loss.

## Input equivalence and observed discrepancy

- Reconstructing V142's panel with `original_budget_frame(v140 monthly_feature_panel)` followed by eligible V141 additions yields 1,872 `(region, month, tau)` rows. These keys match all 1,872 V144 canonical rows one to one. V142 and V145 use the same 30 fit and 12 calibration regions for this fold; the threshold-900 fit has 636 rows.
- At threshold 900, `hours`, both `factor_box_K` endpoints, both physical endpoints, and the anchor fractions agree row by row. The 16 common features agree exactly except `physical_upper_fraction` and `physical_width_fraction`, whose maximum absolute difference is `1.11e-16`. Across the entire panel, only six threshold-600 upper labels change, by one count each. Thus the canonical EF change cannot explain this threshold-900 discrepancy.
- The archived V142 calibration radius is 0.0229005151168004, yielding 17.5 mean target hours of width and 17/18 coverage. The archived V145 radius is 0.8208333333333333, yielding 599.9444 mean hours and 18/18 coverage. Independent single-fold fits reproduce both.
- V142 places the original V140 rows before eligible added rows. V144 stores rows in canonical region/month order, interleaving the two sources. With the *same* V140 fit function, V142 values in V144 row order give radius 0.8208333333333333; V144 values in V142 row order give radius 0.022900515116800493. This isolates training row order as the cause. The V142 and V145 prediction/scoring units are algebraically equivalent: the former works in counts and divides scores by hours, while the latter works in count fractions.

## Mechanism, including exact weight masses

Each fit region has total sample weight one, assigned as `1/n_r` per month. The high-end label is positive for 30 of 636 fit rows. Its total positive weight is exactly `211/168 = 1.25595238095` out of 30, or `211/5040 = 4.1865%`. A positive 95th-percentile intercept requires positive weight greater than `3/2 = 5% of 30`, so both models initialize at zero. The low-end trees are identical across the two orders; all 150 high-end trees differ, starting with the first.

At this zero intercept, scikit-learn's `PinballLoss` gives every fit row the same initial negative gradient, exactly 0.95, including rows with a zero label. A squared-error regression tree therefore has no genuine first-stage split signal. Nevertheless, the first tree's computed root impurity is about `1.15e-14` in V142 order and `1.82e-14` in canonical order; some child impurities are negative at the same scale. With the default zero minimum impurity decrease, floating accumulation admits different spurious splits. The V142 root splits on `fuelshare_wind` (feature index 5); the canonical root splits on `fuelshare_solar` (index 6).

The post-split leaf updates themselves agree with exact rational weighted 95th percentiles. In V142 order, the positive-update leaf has exact positive mass `29/24` within total mass `385/24`, a positive share `29/385 = 7.532%`, and exact update `317/360 = 0.8805555556`. In canonical order, its positive mass is again `29/24`, within total mass `7343/456`, a positive share `551/7343 = 7.504%`, and the same exact update `317/360`. The split assigns calibration month SC 2024-06 to the positive leaf under V142 order but to a zero-update leaf under canonical order. Its high-end label is `591/720`: the old model's first and final predictions are 0.0440278 and 0.999872 (719.91 projected hours); the canonical model remains zero through all 150 stages. The latter gives that month's score `591/720 = 0.8208333333`, which sets the 12-region calibration radius. The former's maximum SC score is 0.0229005151, attained in a different month.

This is a combination of a non-smooth quantile-loss tie at zero and floating tree-split calculations. The row-dependent split is numerical, while the large change in leaf quantiles is a consequence of sending different samples to each leaf. The evidence does not identify a new quantile method or make a claim about scikit-learn outside this fold and version.

## Minimal single-fold reproduction

Run from the repository root. This reads archives and fits only this fold in memory; it writes no files.

```bash
PYTHONDONTWRITEBYTECODE=1 /opt/homebrew/bin/python3.12 - <<'PY'
import json
import sys
from pathlib import Path
import numpy as np
import pandas as pd

sys.path.insert(0, "scripts")
import v140_monthly_conditional_risk as v140
from v141_source_diversity_ablation import original_budget_frame

root = "results/"
old = original_budget_frame(pd.read_csv(root + "v140-monthly-conditional-risk-r1/monthly_feature_panel.csv"))
add = pd.read_csv(root + "v141-source-expansion-input-r1/additional_monthly_panel.csv")
old = pd.concat([old, add.loc[add.eligible_donor]], ignore_index=True, sort=False)
new = pd.read_csv(root + "v144-canonical-profile-cache-r1/canonical_monthly_panel.csv")
roles = pd.read_csv(root + "v145-recomposed-risk-baselines-r1/region_roles.csv")
fold = roles.query('target_region == "PACE" and seed == 41')
fit_regions = set(fold.loc[fold.role.eq("fit"), "region"])
cal_regions = set(fold.loc[fold.role.eq("calibration"), "region"])
v140.FEATURES = json.loads(Path(root + "v140-monthly-conditional-risk-r1/summary.json").read_text())["common_features"]
keys = ["region", "month", "tau"]
old_at_new = old.set_index(keys).loc[pd.MultiIndex.from_frame(new[keys])].reset_index()
new_at_old = new.set_index(keys).loc[pd.MultiIndex.from_frame(old[keys])].reset_index()

for name, panel in (("V142 order", old), ("V142 values/V144 order", old_at_new),
                    ("V144 order", new), ("V144 values/V142 order", new_at_old)):
    frame = panel.loc[panel.tau.eq(900)]
    fit = frame.loc[frame.region.isin(fit_regions)]
    cal = frame.loc[frame.region.isin(cal_regions)]
    target = frame.loc[frame.region.eq("PACE")]
    models = v140.fitted_models(fit, "direct_cqr_all")
    ca, cb = v140.projected_predictions(cal, "direct_cqr_all", models)
    scores = pd.DataFrame({"region": cal.region.to_numpy(),
                           "score": v140.score(cal, ca, cb, False)}).groupby("region").score.max()
    radius = float(scores.sort_values().iloc[11])  # ceil((12 + 1) * 0.9) = 12
    a, b = v140.projected_predictions(target, "direct_cqr_all", models)
    lo = np.maximum(target.physical_lower, np.floor(a - radius * target.hours + 1e-9))
    hi = np.minimum(target.physical_upper, np.ceil(b + radius * target.hours - 1e-9))
    covered = (lo <= target.factor_box_K_lower) & (hi >= target.factor_box_K_upper)
    print(name, "radius", radius, "mean_width", np.mean(hi - lo), "coverage", np.mean(covered))
PY
```

Expected radii, in order: approximately `0.0229005151`, `0.8208333333`, `0.8208333333`, `0.0229005151`.

## Interpretation and controls

The V142 narrow result is obtainable under some row orders, including its frozen concatenation order; it is not a stable estimate of this baseline's performance. Ten additional fixed random permutations of the same single fold all returned radius 0.8208333333 and mean width 599.9444 hours. This small sensitivity probe establishes that V142's narrow result is unusual among those tested orders, not a distributional claim about all permutations. Both archived baselines remain immutable; their EF contracts must be labelled, as must V146's different joint calibration scope.

In a single-fold diagnostic, setting `min_impurity_decrease=1e-10` gave a one-node first tree under both examined orders, radius 0.8494623656, and mean width 620.8333 hours. This is a sensitivity control, not a replacement for the frozen baselines. Replacing `1/n_r` weights with proportional integer weights changed the chosen split but did not remove spurious first-tree splitting; higher precision in the terminal leaf quantile cannot repair an earlier split. A strict-positive-residual subgradient convention, if evaluated, changes the optimizer's tie handling and needs separate pre-registration and fair application across baselines. Neither that convention nor suppressing numerical splits is claimed as original research.

The V146 horizontal envelope cannot be credited with beating a strong ordinary baseline merely by comparing it with V145 `direct_common_all` in this cell. V145 also contains narrower ordinary references here, including `residual_response_all` at 23.0 hours and `recompose_conditional` at 28.39 hours, both with 18/18 coverage; their calibration is per threshold, whereas V146 uses a joint two-threshold score. The numerical diagnosis concerns one fold and one threshold; it does not establish behavior in other cells or exchangeability of the region-level calibration.
