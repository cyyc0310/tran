# V329 protocol: whole-region cold-start transfer of provincial annual EF

Registered 2026-10-30 before the V329 run. V327 found a modest NBS-versus-Ember improvement under province-wise LOPO, but V328 did not improve it by adding a common unobserved-scope coefficient. V329 tests the harder user-relevant condition: predict provinces in a whole unseen Chinese statistical region.

## Frozen geographic blocks

Use the standard seven NBS statistical regions, excluding Tibet because the MEE target table has no Tibet row:

- North: Beijing, Tianjin, Hebei, Shanxi, Inner Mongolia
- Northeast: Liaoning, Jilin, Heilongjiang
- East: Shanghai, Jiangsu, Zhejiang, Anhui, Fujian, Jiangxi, Shandong
- Central: Henan, Hubei, Hunan
- South: Guangdong, Guangxi, Hainan
- Southwest: Chongqing, Sichuan, Guizhou, Yunnan
- Northwest: Shaanxi, Gansu, Qinghai, Ningxia, Xinjiang

No region assignment may be changed after seeing scores.

## Inputs, splits, and models

Use the same 2021–2023 official MEE annual provincial factor targets and lagged NBS/Ember five-group feature definitions as V327. Hold out one whole geographic region at a time. For 2022 targets, train only on 2021 target rows from provinces outside the held region; for 2023 targets, train on 2021–2022 target rows from provinces outside the region. The target province and every province in its macroregion are excluded from all fitting rows. Fit same-input-budget nonnegative least squares without an intercept for NBS and Ember; include the outside-region training-target mean as the no-feature baseline. Do not use the held province's prior MEE factor.

## Outcomes

Report province-weighted and equal-region-weighted MAE/RMSE/P90 by target year and pooled; each region-year fold; province counts improved over the outside-region mean; and a region-cluster bootstrap for paired NBS-minus-Ember absolute error. Report support/missingness and all selected folds. Compare against V327 LOPO only as context, not as a paired comparison. The target remains annual consumption-side MEE EF, not hourly generation CIF. This is retrospective spatial block transfer, not a real-time forecast, because target release vintages are delayed and feature vintages are unverified.

## Decision rule

The cold-start claim advances only if NBS mix beats the outside-region mean in most held-out regions and both target years, and if its edge over Ember is not driven by one region. A severe failure in any region blocks a universal transfer claim even if the pooled average improves.
