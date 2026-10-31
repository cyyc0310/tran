# Conditional risk calibration: novelty boundary for a coarse-mix method

2026-10-27. Focused literature check after the V50 source-held-out mix probes. This is a novelty-boundary note, not a systematic review.

## Closest existing methods

- [Guan, “Localized conformal prediction: a generalized inference framework for conformal prediction,” Biometrika (2023)](https://academic.oup.com/biomet/article/110/1/33/6647831) localizes the calibration distribution around a test covariate and combines that localization with different conformity scores. Local weighting or choosing nearby source regions by monthly mix is therefore not, by itself, a new calibration principle.
- [Gibbs, Cherian & Candès, “Conformal Prediction With Conditional Guarantees” (2023)](https://arxiv.org/abs/2305.12616) formalizes a spectrum of conditional targets as coverage over classes of covariate shifts, including finite subgroup guarantees, and explicitly notes that unrestricted exact conditional coverage is impossible distribution-free. A ten-region dataset cannot support an unrestricted region-wise guarantee without stronger assumptions or substantially more source information.
- [Jiang & Xie, “Spatial Conformal Inference through Localized Quantile Regression,” ICML 2026, PMLR 306](https://proceedings.mlr.press/v306/jiang26at.html) couples local quantile regression and conformal calibration for unobserved spatial locations. It reports finite-sample marginal validity under spatial exchangeability and asymptotic conditional coverage under stationarity and spatial mixing. This is close to the *shape* of a spatially conditioned risk rule, although the intended CIF transfer setting has not established those assumptions.
- [Tibshirani et al., “Conformal Prediction Under Covariate Shift,” NeurIPS 2019](https://papers.neurips.cc/paper/2019/hash/8fb21ee7a2207526da55a679f0332de2-Abstract.html) reweights calibration scores for a specified covariate shift when the density ratio is known or accurately estimated. That assumption cannot be inferred from monthly mix similarity alone; cross-region changes in the conditional hourly dispatch response can violate it.

## Implication for transfCIF

The full monthly fuel-mix probe is an interesting predictor of source-level normalized risk, but the direct next step—fit a local/conditional score quantile from that covariate—is already a familiar conformal construction. It should be treated as a strong baseline or an application of existing local calibration, not labeled a new algorithm.

A defensible original contribution would need to change the scientific object or provide evidence unavailable in the generic literature, for example: (1) a fuel-accounting-aware, time-availability benchmark that separates what monthly generation mix identifies from hourly dispatch risk; (2) a source-held-out study of whether a physically defined mix proxy predicts the *residual hourly tail* beyond standard localized conformal/heteroscedastic baselines; and (3) robust uncertainty or sensitivity bounds that explicitly expose the few-source and label-vintage limitations instead of claiming distribution-free region-conditional guarantees.

The current ten-region probe is far short of this bar. It motivates a conditional-calibration comparator experiment; it does not establish novelty, coverage, or China transfer.
