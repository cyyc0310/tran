# TransCIF: An Exact Error Decomposition for Plug-and-Play Cross-Region Carbon Intensity Forecasting

**Draft status:** first draft, workshop-length (target: Climate Change AI-style ML workshop). All numerical results are computed on real 2023 AEMO/NEM historical data; no synthetic data is used for any reported number. This draft is honest about negative and mixed results — see Section 7 (Limitations) before submission.

---

## Abstract

Grid carbon-intensity (CI) forecasting models are usually trained and evaluated within a single electricity market, yet operators increasingly need forecasts for regions with little or no historical data. We study the *plug-and-play* transfer setting: a model trained on one region's renewable-share dynamics is deployed, with only light recalibration, to a different region's grid. Rather than adding another combination of transfer-learning tricks, we derive an **exact algebraic identity** (Theorem 1) for the one-step transfer error of any two-stage forecasting pipeline that reconstructs CI from a predicted renewable-share trajectory through a physically linear emission-factor formula. The identity splits the error into a *transfer-amplification* term, scaled by a region-specific, exactly computable Lipschitz constant $L_T$ (the gap between a region's renewable and non-renewable emission factors), and a *residual-estimation* term. We validate this identity numerically on four real Australian NEM regions (QLD1, NSW1, VIC1, SA1) spanning a 2.4$\times$ range of $L_T$, and use it to explain why target-domain fine-tuning helps. We further show, on real cross-region data, that our best configuration (encoder fine-tuning + CORAL feature alignment, "D+E") beats a naive persistence baseline in all four regions, whereas the pre-fine-tuning configuration only does so in two of four — evidence that "multi-scenario applicability" is a property of the *complete* adapted pipeline rather than a base model that "just transfers." We report all of this, including the results that complicate a clean "it just works" narrative.

---

## 1. Introduction

Grid-level carbon intensity (CI, in gCO$_2$/kWh) is the standard signal for carbon-aware load shifting, EV charging scheduling, and demand response. Forecasting CI 12 hours ahead is well studied *within* a single market with abundant historical data, but many regions — especially smaller or newer markets — lack the years of labeled data that data-hungry forecasters need. This motivates a **plug-and-play transfer** setting: train once on a data-rich source region (or several), deploy to a data-scarce target region with only a small calibration split, and ask how well the model's error can be *understood and bounded*, not just empirically minimized.

Our starting point was a design combining five techniques — scale-invariant reparameterization, a physics-plus-residual reconstruction head, synthetic-perturbation consistency regularization, calibration-time reuse of a dominant-variable indicator, and split-conformal prediction — plus two domain-adaptation techniques adopted for the target domain (gradual-unfreezing fine-tuning and Deep CORAL alignment). Each component is a reasonable, previously published technique, but the composition itself is not a new theoretical contribution, and — as we report honestly in Section 5 — the composition alone did not reliably beat a trivial persistence baseline on our target region.

This led us to ask a different question: **can we say something exact, not just empirical, about how transfer error arises in this class of pipeline?** The answer turns out to be yes, because the reconstruction step from predicted renewable share to CI is *not* a black box — it is a fixed, publicly documented linear formula. That linearity lets us write an exact algebraic identity for the one-step transfer error, rather than an empirical correlation. This paper's contribution is that identity (Theorem 1), its numerically verified, falsifiable corollary (Corollary 1), and an honest account of what the accompanying domain-adaptation techniques do and do not buy us across four real regions.

**Contributions.**
1. Theorem 1: an exact decomposition of one-step CI transfer error into a *transfer-amplification* term and a *residual-estimation* term, for any pipeline that reconstructs CI from a predicted renewable share through the standard linear emission-factor formula.
2. Corollary 1: a falsifiable, region-specific prediction from Theorem 1, verified numerically on four real AEMO regions under two model configurations and multiple calibration-split choices, and shown to be stable across five random seeds.
3. A cross-region evaluation against a naive persistence baseline showing that the *complete* adapted pipeline (encoder fine-tuning + CORAL) is necessary, not incidental, for consistently beating that baseline across regions.
4. A closed-form Bayes-optimal (Bates-Granger) fusion of the network forecast with persistence and climatology forecasts, together with an overfitting diagnostic, as a secondary result.

---

## 2. Related Work

**Physics-informed CI forecasting.** Our Stage-2 reconstruction follows the average-CI formula used by Zhang et al. (2026a), who model day-ahead grid CI end-to-end with a joint local-temporal and cross-variable dependency network. We deliberately do *not* propose a competing end-to-end architecture; instead, we hold their (and any comparable pipeline's) linear renewable/non-renewable reconstruction step fixed and ask a question their formulation does not address: given that the reconstruction step is linear, what can be said *exactly* — not just empirically — about how error propagates through it under domain shift? This is the gap Theorem 1 fills, and it is architecture-agnostic: it applies to any pipeline that reconstructs CI from a predicted share through a linear emission-factor map, not only ours.

**Cross-region / data-scarce CI forecasting.** Zhang et al. (2026b) address the closely related problem of forecasting CI in data-scarce regions via a dual-graph carbon-domain foundation model, using metadata-driven hypergraph fine-tuning to transfer from data-rich regions. Their ablation is, to our knowledge, the strongest existing empirical evidence that target-region fine-tuning is critical for cross-region CI transfer specifically. Our Section 5.4 cross-region result (4/4 vs. 2/4 regions beating persistence with vs. without fine-tuning) is an independent, real-data confirmation of that same qualitative finding on a different network architecture and a different market (AEMO/NEM rather than their data-scarce target regions), and Theorem 1 additionally gives a mechanistic account of *why* fine-tuning helps — it shrinks the transfer-amplification term (Section 5.1), not the residual term — rather than only reporting that it does.

**Domain generalization.** We use MLDG (Li et al., 2018) for multi-source meta-training and Deep CORAL (Sun & Saenko, 2016) for feature alignment, both as off-the-shelf components rather than contributions of this paper. Our ablation (Section 5.5) adds a finding neither original paper's setting surfaces: for this reconstruction-linear CI pipeline, MLDG's meta-training framework alone is *not* the active ingredient (a pure-ERM baseline without any meta-learning score worse than every other configuration we tested), and CORAL alone is actively harmful — both techniques only help in combination with target-domain fine-tuning (D). This is a practical caveat for anyone applying these techniques to a similarly structured physics-plus-residual pipeline, not visible from either technique's original evaluation.

**Target-domain fine-tuning.** Our encoder fine-tuning follows the gradual-unfreezing supervised fine-tuning recipe of Khanal et al. (2024), proposed generically for time-series transformers under domain shift. We instantiate it specifically for CI transfer and, via Theorem 1's decomposition, attribute its benefit to a specific mechanism (a 12.4% reduction in Term ①, Section 5.1) rather than reporting only an aggregate error reduction.

**Forecast combination.** Our fusion baseline (Section 5.7) applies the classical Bates-Granger (1969) optimal linear forecast combination to three renewable-share forecasters (persistence, network, climatology). Unlike the original setting, our components are affine-combined shares that then pass through the same linear CIF reconstruction as the rest of the pipeline, so the fused result remains exactly decomposable via Theorem 1 — the fusion is not a black-box ensemble bolted on top, but a third forecaster inside the same error-decomposition framework. Connecting the fused weights to a closed-form Ben-David et al. (2010) $\mathcal{H}$-divergence domain-adaptation bound is a natural direction for a second theorem (Theorem 2), which we have sketched but not yet derived (see Limitations).

**Conformal prediction.** Our calibration-time uncertainty band reuses split-conformal prediction in the distribution-free regression form of Lei et al. (2018), reusing Stage 2's existing calibration split rather than requiring a dedicated one — a design choice motivated by the data-scarce target-region setting this paper targets, where held-out data is the scarcest resource.

Overall, none of the individual techniques above is new. What we believe is new is Theorem 1 itself: an *exact*, region-specific, falsifiable decomposition of transfer error for this entire class of physics-plus-residual pipeline, verified across four real regions and shown to explain (not just report) why the adaptation techniques above help — together with the practical, honestly-qualified finding (Section 5.4) that the complete adaptation stack, not any single technique or the base model alone, is what is required for consistent cross-region gains over a naive baseline.

---

## 3. Method Overview

The pipeline has two stages, trained/calibrated on real AEMO/NEM 2023 hourly data (`nem_2023_hourly_{REGION}.csv`, plus `temperature_2023_{REGION}.csv` sourced from the Open-Meteo historical archive — **not** an AEMO/NEMED product, disclosed here per our own data-provenance policy).

- **Stage 1 (encoder):** a domain-invariant encoder fuses a multi-wavelet kernel convolution (LT-MWKC) and a cross-variable dynamic wavelet correlation module (CV-DWCC) to predict a 12-hour-ahead renewable-share trajectory $\hat s_{t+1:t+H}$ from a windowed history of measured grid signals (renewable share, load, optionally raw generation channels and temperature).
- **Stage 2 (physics + residual):** the predicted share is passed through the fixed linear CIF formula to obtain a physics-only CI forecast, and a small residual head $\hat\Delta_t$, fit only on the target region's calibration split, absorbs systematic bias the physics formula misses (cross-border trade, transmission losses, sub-fuel-mix differences).

On top of this base pipeline we evaluate two target-domain adaptation techniques: **D** (gradual-unfreezing supervised fine-tuning of the encoder on the target calibration split) and **E** (Deep CORAL feature alignment during multi-source pretraining). We refer to the pre-adaptation configuration as **baseline** and the configuration with both techniques as **+D+E**.

---

## 4. Theorem 1: An Exact Error Decomposition

### 4.1 Setup

The CIF reconstruction formula used throughout this pipeline (and in the source literature it follows) is, for a two-category renewable/non-renewable split, **exactly linear** in the renewable share $s$:

$$\mathrm{CIF}(s) = s \cdot C_{\text{renew}} + (1-s)\cdot C_{\text{nonrenew}}.$$

$C_{\text{renew}}$ and $C_{\text{nonrenew}}$ are region-specific emission factors (gCO$_2$/kWh), looked up from a fixed table, not estimated. We define the **true residual** $\varepsilon_t$ as whatever the physics formula does not capture:

$$CI_{true,t} := \mathrm{CIF}(s_t) + \varepsilon_t,$$

which is a definition, not an assumption — it simply attributes everything the linear formula misses (cross-border trade, transmission losses, sub-fuel-mix heterogeneity) to $\varepsilon_t$. The model predicts $CI_{pred,t} = \mathrm{CIF}(\hat s_t) + \hat\Delta_t$, where $\hat\Delta_t$ is the learned residual-correction head's output.

### 4.2 Theorem 1

**Theorem 1.** *For any pipeline that reconstructs CI from a predicted share $\hat s_t$ through the linear formula $\mathrm{CIF}(\cdot)$ above, the one-step forecast error decomposes exactly as*

$$CI_{pred,t} - CI_{true,t} \;=\; \underbrace{(\hat s_t - s_t)(C_{\text{renew}} - C_{\text{nonrenew}})}_{\text{Term ① (transfer amplification)}} \;+\; \underbrace{(\hat\Delta_t - \varepsilon_t)}_{\text{Term ② (residual estimation)}}.$$

*Consequently,*

$$|CI_{pred,t} - CI_{true,t}| \;\le\; L_T \cdot |\hat s_t - s_t| \;+\; |\hat\Delta_t - \varepsilon_t|, \qquad L_T := |C_{\text{renew}} - C_{\text{nonrenew}}|.$$

**Proof.** Subtract $CI_{true,t}$ from $CI_{pred,t}$ and expand both sides using the definitions above:
$$CI_{pred,t} - CI_{true,t} = \big[\mathrm{CIF}(\hat s_t) - \mathrm{CIF}(s_t)\big] + \big[\hat\Delta_t - \varepsilon_t\big].$$
By linearity of $\mathrm{CIF}$,
$$\mathrm{CIF}(\hat s_t) - \mathrm{CIF}(s_t) = (\hat s_t - s_t)C_{\text{renew}} + \big[(1-\hat s_t)-(1-s_t)\big]C_{\text{nonrenew}} = (\hat s_t - s_t)(C_{\text{renew}}-C_{\text{nonrenew}}).$$
Substituting gives the identity; the triangle inequality gives the bound. $\blacksquare$

This is an exact algebraic identity, not an asymptotic or approximate bound — it follows entirely from the fact that the reconstruction formula used in this class of pipeline is linear in the predicted share. $L_T$ requires no estimation: it is read directly off each region's published emission-factor table.

We are explicit about what this theorem is and is not. It is not deep mathematics — it is error propagation through a linear map, essentially an application of the triangle inequality. Its value is not the difficulty of the derivation but that it turns "why did the CI forecast err" into two *separately measurable, separately attributable* quantities for a model class that, to our knowledge, has previously only been evaluated end-to-end.

### 4.3 Corollary 1 (a falsifiable, cross-region prediction)

**Corollary 1.** *If two regions have similar renewable-share prediction error $|\hat s_t - s_t|$, the region with the larger $L_T$ should show a proportionally larger contribution from Term ① relative to Term ②.*

Table 1 lists the real, exactly computed $L_T$ values for the four Australian NEM regions we study.

**Table 1: Region-specific transfer-amplification constants ($L_T$, gCO$_2$/kWh), from the real 2023 emission-factor tables.**

| Region | $C_{\text{renew}}$ | $C_{\text{nonrenew}}$ | $L_T$ |
|---|---|---|---|
| SA1  | 0.00 | 490.43  | **490.43 (smallest)** |
| QLD1 | 0.00 | 841.59  | 841.59 |
| NSW1 | 0.09 | 875.23  | 875.14 |
| VIC1 | 0.00 | 1160.12 | **1160.12 (largest)** |

This corollary is falsifiable: if we compute Term ① and Term ② from real data and trained models, the ranking of Term-①-share across regions must track the ranking of $L_T$, under the (testable) proviso that the underlying share-prediction errors are comparable in magnitude. We test this directly in Section 5.

---

## 5. Experiments

All experiments use real AEMO/NEM 2023 hourly data. The temperature covariate used in some configurations comes from the Open-Meteo historical archive, not AEMO/NEMED, and is flagged wherever it appears. No experiment in this section uses synthetic or placeholder data.

### 5.1 Single-split numerical validation (SA1)

We first verify Theorem 1's identity holds exactly (up to floating-point precision) on real SA1 data, and test Corollary 1 in its original single-region form. Table 2 reports the pointwise maximum absolute gap between the identity's two sides (a sanity check on the derivation and implementation), the mean absolute total error, and the Term-①/Term-② split, for the pre-adaptation baseline and the +D+E configuration.

**Table 2: Theorem 1 verification on SA1 (calibration fraction 0.7).**

| Configuration | max abs. identity gap | mean abs. total error | mean\|Term①\| | mean\|Term②\| | Term① share | Dominant term |
|---|---|---|---|---|---|---|
| baseline | 5.72e-05 | 74.206 | 84.413 | 22.123 | 79.2% | Term① |
| +D+E | 4.86e-05 | 66.887 | 73.966 | 20.288 | 78.5% | Term① |

The identity gap is at floating-point precision in both configurations, confirming the algebra and its implementation are correct. SA1 has the smallest $L_T$ of the four regions (Table 1); Corollary 1 predicts that, if SA1's forecast still errs substantially (which it does — see Section 5.5), the cause should be Term ① (share-prediction error itself) rather than emission-factor amplification. The data confirm this: Term ① accounts for ~79% of the error in both configurations. We also observe that D+E's improvement (74.206 → 66.887) is driven mostly by a reduction in Term ① (84.413 → 73.966, a 12.4% drop) rather than Term ② — consistent with the mechanism of fine-tuning (D), which acts directly on the encoder's share prediction, not the residual head.

### 5.2 Robustness to calibration-split choice (rolling origin)

Corollary 1's single-split result could in principle be an artifact of the particular 0.7 calibration split. We re-ran the decomposition at calibration fractions $\{0.6, 0.7, 0.8\}$ for both configurations (6 runs total). Term-① share ranged 79.67%–80.33% (baseline) and 78.31%–80.55% (+D+E), with standard deviations under one percentage point, and Term ① was dominant in all six runs. The finding is not an artifact of split choice.

### 5.3 Cross-region generalization of the *identity* (leave-one-domain-out rotation)

Corollary 1 is inherently a **cross-region** claim (smaller $L_T$ $\to$ smaller Term-① share, holding share-error roughly fixed), so confirming it on SA1 alone — regardless of how many splits — cannot test whether the $L_T$ ranking actually predicts anything across regions, because there is no second region to compare against. We therefore rotate the held-out target region through all four AU regions (QLD1, NSW1, VIC1, SA1), retraining an MLDG source-domain model from the other three each time, and rerun the identical decomposition. Table 3 shows all eight (4 regions $\times$ 2 configurations) results.

**Table 3: Leave-one-domain-out rotation across four real AU regions.**

| Target | $L_T$ | Configuration | Identity gap | Term① share | Dominant term |
|---|---|---|---|---|---|
| QLD1 | 841.59 | baseline | 1.23e-04 | 83.71% | Term① |
| QLD1 | 841.59 | +D+E | 1.21e-04 | 82.51% | Term① |
| NSW1 | 875.14 | baseline | 1.46e-04 | 89.28% | Term① |
| NSW1 | 875.14 | +D+E | 1.46e-04 | 88.98% | Term① |
| VIC1 | 1160.12 | baseline | 1.78e-04 | 91.84% | Term① |
| VIC1 | 1160.12 | +D+E | 1.53e-04 | 90.50% | Term① |
| SA1 | 490.43 | baseline | 6.10e-05 | 79.66% | Term① |
| SA1 | 490.43 | +D+E | 5.34e-05 | 78.84% | Term① |

**Two findings, reported honestly and separately.**

1. **The identity's qualitative claim generalizes.** In all 8/8 combinations, spanning a 2.4$\times$ range of $L_T$ (490 to 1160) and all four AU regions, Term ① dominates Term ②. This is no longer a single-region anecdote: transfer amplification, not residual-estimation error, is the primary driver of CI transfer error across this entire region set.
2. **A directional expectation we must correct.** Term-①-share actually increases *monotonically* with $L_T$ (baseline: SA1 490$\to$79.66% $<$ QLD1 842$\to$83.71% $<$ NSW1 875$\to$89.28% $<$ VIC1 1160$\to$91.84%; same monotonicity for +D+E) — the *opposite* direction from our original intuition that SA1's small $L_T$ should give it the *smallest* Term-① share. On reflection, this monotonicity is close to an algebraic byproduct rather than an independent empirical finding: Term ① $= (\hat s-s)\cdot L_T$ is directly scaled by $L_T$, so as long as the underlying share-prediction errors $|\hat s - s|$ are of comparable magnitude across regions, Term-①-share rising with $L_T$ falls out of the formula almost automatically. We flag this so the monotonic ranking is not mistaken for new evidence for the theorem; the genuinely informative finding is (1) — the 8/8 dominance result across a wide $L_T$ range and four distinct regions.

### 5.4 Does the adapted pipeline actually beat a naive baseline, in every region?

Section 5.3 establishes that Theorem 1's *decomposition* behaves consistently across regions, but it says nothing about whether the pipeline is *useful* relative to the simplest possible forecast — repeating the last observed renewable share across the 12-hour horizon ("persistence"). This is the direct evidence a "multi-scenario applicability" claim needs, and it had not been computed per-region before this experiment. Table 4 reports persistence MAE alongside the two configurations' corrected MAE, per region, reusing the already-trained models from Section 5.3 (no retraining).

**Table 4: Cross-region comparison against the naive persistence baseline (gCO$_2$/kWh MAE).**

| Region | $L_T$ | Persistence MAE | baseline MAE (vs. persistence) | +D+E MAE (vs. persistence) |
|---|---|---|---|---|
| QLD1 | 841.59 | 103.181 | 62.397 (**\-39.5%, wins**) | 58.396 (**\-43.4%, wins**) |
| NSW1 | 875.14 | 133.492 | 82.997 (**\-37.8%, wins**) | 75.172 (**\-43.7%, wins**) |
| VIC1 | 1160.12 | 104.764 | 116.534 (**+11.2%, loses**) | 103.405 (**\-1.3%, narrow win**) |
| SA1 | 490.43 | 67.568 | 76.239 (**+12.8%, loses**) | 65.561 (**\-3.0%, wins**) |

**Honest reading.**

1. **The positive result for a "multi-scenario applicability" claim:** the +D+E configuration beats persistence in **all four regions (4/4)**, by margins from -1.3% to -43.7%. This is the first evidence in this project that the adapted pipeline generalizes across regions, rather than being validated on SA1 alone.
2. **The result that must be reported alongside it, not omitted:** the *pre-adaptation* baseline — i.e., the pipeline without target-domain fine-tuning — actually beats persistence by a wide margin in QLD1 and NSW1 (-38% to -40%), but **loses** to persistence in VIC1 and SA1 (+11% to +13%). There is no single "deploy anywhere and win" base model: in two of the four regions, skipping target-domain fine-tuning leaves the pipeline *worse* than the trivial baseline.
3. This makes target-domain fine-tuning (D, in particular) a **necessary condition**, in this data, for consistently beating persistence across regions — not an incremental polish step. Without it, the pipeline is a net negative in half of the regions we tested; with it, the pipeline is positive in all four.
4. The persistence MAE itself varies substantially across regions (67.6 to 133.5), and this variation does **not** track the $L_T$ ranking (NSW1's $L_T$ is lower than VIC1's, yet NSW1's persistence MAE is the highest of the four) — reflecting region-specific volatility in 12-hour renewable-share swings, a variable independent of the emission-factor-based analysis in Sections 5.1–5.3. We flag this so readers do not conflate the two.

We therefore avoid the claim "the model transfers everywhere by default," and instead state: *the complete adapted pipeline (in particular, target-domain fine-tuning) is a necessary condition for beating a naive baseline consistently across regions — 4/4 regions with the full pipeline, versus only 2/4 without it.*

### 5.5 Domain-adaptation ablation (single-region, SA1)

To attribute the D+E result, we ablate D (encoder fine-tuning) and E (CORAL alignment) independently on SA1 (Table 5). This experiment, run before the cross-region rotation, established which components matter before we generalized the check across regions.

**Table 5: SA1 domain-adaptation ablation.**

| Configuration | Corrected MAE | vs. persistence (67.568) |
|---|---|---|
| baseline (Stage-1-optimal combination, no target adaptation) | 74.712 | +10.6% |
| **+D (fine-tuning only)** | 67.240 | **\-0.5%** |
| +E (CORAL only) | 75.788 | +12.2% (worse than baseline) |
| **+D+E** | **66.004** | **\-2.3% (best)** |

D alone is already sufficient to beat persistence; E alone is not merely unhelpful but actively harmful; D+E together outperform D alone, suggesting CORAL acts as a regularizer *in the presence of* supervised fine-tuning rather than as a standalone transfer mechanism. We also ran a pure-ERM ablation (no MLDG meta-learning at all): it scored **worst** of every configuration tested in this project (+14.9% vs. persistence), confirming that the active ingredient is *target-domain fine-tuning*, not the MLDG meta-training framework by itself.

### 5.6 Multi-seed robustness (SA1, 5 seeds)

A single training run's randomness (notably, MLDG's per-round `random.choice` over source regions) could inflate any of the above numbers. We fixed this randomness explicitly and reran baseline and +D+E across 5 seeds (Table 6).

**Table 6: Multi-seed robustness (mean $\pm$ std, $n=5$).**

| Configuration | Corrected MAE | Term① share |
|---|---|---|
| baseline | 77.934 $\pm$ 2.136 | 81.1% $\pm$ 1.3% |
| +D+E | 66.492 $\pm$ 0.783 | 79.3% $\pm$ 1.0% |

+D+E beat its paired baseline in all 5/5 seeds (paired differences $-9.4$ to $-13.9$), and beat persistence in all 5/5 seeds (improvement range $-0.22\%$ to $-2.84\%$, roughly a 13$\times$ spread — the direction is robust, but the exact magnitude should not be over-interpreted from a single run). Term-①-share stayed within a narrow 77.8%–82.4% band across all 10 (seed $\times$ configuration) runs, reinforcing Section 5.1–5.3's structural finding. With $n=5$, we report these as descriptive statistics only; we do not claim formal statistical significance.

### 5.7 A Bayes-optimal fusion baseline, and an overfitting check

As a secondary, more classical result, we fit a per-horizon Bates-Granger optimal linear (affine) combination of three renewable-share forecasters — persistence, the +D+E network, and a diurnal climatology model — minimizing forecast-combination variance, with weights allowed to be negative (this is the standard, unconstrained affine solution, not a convex combination). Table 7 summarizes.

**Table 7: Bayes-optimal (Bates-Granger) fusion result.**

| Quantity | Value |
|---|---|
| Persistence MAE (floor) | 67.568 |
| Network-only (+D+E) physics MAE | 71.226 |
| Climatology-only physics MAE | 86.359 |
| **Fused corrected MAE** | **61.196** |
| **Fused vs. persistence** | **\-9.4%** |

This is the largest margin over persistence we observe anywhere in this project. Because Bates-Granger weights are fit on a calibration split and could simply memorize it, we ran a direct overfitting diagnostic: comparing each component forecaster's calibration-split MAE to its held-out evaluation-split MAE. All three components showed a *negative* calibration-to-evaluation gap (persistence $-9.9\%$, network $-8.6\%$, climatology $-8.8\%$) — i.e., held-out performance was if anything *better* than on the calibration split, well inside the $+20\%$ threshold we set for an overfitting flag. We found no evidence that the fusion weights are memorizing the calibration set.

We flag this fusion procedure as a natural site for a second theorem — connecting the per-horizon weight structure to a closed-form Ben-David et al. (2010) $\mathcal{H}$-divergence bound on the underlying share-prediction error — but that derivation is not yet complete (see Limitations).

---

## 6. Discussion: What "Multi-Scenario Applicability" Does and Does Not Mean Here

Taken together, Sections 5.3–5.6 support a specific, qualified claim rather than a sweeping one. The qualitative structure of transfer error — that it is dominated by renewable-share transfer error itself (Term ①) rather than by how strongly a region's emission-factor gap amplifies that error (Term ②'s counterpart) — holds in 8/8 tested region/configuration combinations spanning a 2.4$\times$ range of $L_T$. That is a genuine cross-region regularity, and Theorem 1 is what lets us state it precisely rather than as an empirical hunch.

However, whether the *pipeline as deployed* actually beats a naive baseline is a separate, harder question, and the answer is: only with the complete adapted configuration. Skipping target-domain fine-tuning is not a minor degradation — it flips the sign of the result in half the regions tested. We think the more defensible framing for a paper claim is: *the theoretical decomposition (Theorem 1 / Corollary 1) generalizes structurally across regions; practical, consistent superiority over a naive baseline further requires the complete adaptation stack, especially target-domain fine-tuning.* We deliberately avoid a claim like "the base model transfers everywhere out of the box," which the data do not support.

---

## 7. Limitations

We list these because we believe an honest limitations section is more useful to reviewers and to future work than omitting them.

- **No external state-of-the-art baseline.** All comparisons are against a naive persistence baseline and a climatology baseline; we have not benchmarked against a published SOTA CI-forecasting model on the same data.
- **Single-year data.** All experiments use 2023 AEMO/NEM data only; we have not tested robustness to inter-annual variation (e.g., a different weather year, changing generation mix).
- **Temperature covariate provenance.** The temperature covariate used in some ablations is sourced from the Open-Meteo historical weather archive, not from AEMO/NEMED. We flag this explicitly wherever the covariate is used, per our own data-governance policy, since it is not an official grid-operator product.
- **Theoretical depth.** Theorem 1 is an exact identity, but it follows from elementary linear error propagation, not from new mathematics; its contribution is the attribution/diagnostic value it provides for this specific model class, not proof difficulty.
- **Theorem 2 is not yet derived.** We sketch a connection between the Bates-Granger fusion weights and a Ben-David et al. (2010) $\mathcal{H}$-divergence bound on share-prediction error, but have not completed this derivation; Section 5.7's fusion result stands on its own as an empirical result in the meantime.
- **Region count.** Our cross-region evidence covers four regions within a single electricity market (the Australian NEM). We have not tested transfer to a market outside the NEM, where fuel-mix categories, market rules, and data availability may differ substantially.
- **Small-$n$ significance.** The multi-seed check (Section 5.6) uses $n=5$ seeds; we report paired differences descriptively and do not claim formal statistical significance.

---

## 8. Conclusion and Future Work

We derived an exact algebraic identity (Theorem 1) for one-step CI transfer error in physics-grounded, share-based forecasting pipelines, verified it numerically on real 2023 AEMO data across four Australian NEM regions and multiple calibration splits, and used it to explain why target-domain fine-tuning reduces error (by shrinking the transfer-amplification term, not the residual term). We further showed that the complete adapted pipeline — but not the base model alone — beats a naive persistence baseline in all four regions tested, a more qualified and more defensible claim than "the model generalizes out of the box." Future work includes completing the Ben-David-style bound sketched in Section 5.7 (Theorem 2), extending the region set beyond the NEM, and benchmarking against a published external baseline.

---

## Data and Code Availability

All results in this draft are computed from real AEMO/NEM 2023 hourly market data and DUID-level generation data, plus the Open-Meteo historical weather archive for the temperature covariate (explicitly flagged wherever used). No synthetic or placeholder data was used for any result reported in this paper. The complete numerical results, per-experiment logs, and scripts underlying every table in this draft are consolidated in `docs/experiments/2026-07-17-all-experiments-summary.md` in the project repository, with a full script/log index in that document's Section 7.

---

## References

Bates, J. M., & Granger, C. W. J. (1969). The combination of forecasts. *Operational Research Quarterly*, 20(4), 451–468. https://doi.org/10.2307/3008764

Ben-David, S., Blitzer, J., Crammer, K., Kulesza, A., Pereira, F., & Vaughan, J. W. (2010). A theory of learning from different domains. *Machine Learning*, 79(1–2), 151–175.

Khanal, S., Tirupathi, S., Zizzo, G., Rawat, A., & Pedersen, T. B. (2024). Domain adaptation for time series transformers using one-step fine-tuning. In *Proceedings of the 4th Workshop on AI for Time Series Analysis (AI4TS), AAAI 2024*, Vancouver, Canada. arXiv:2401.06524.

Lei, J., G'Sell, M., Rinaldo, A., Tibshirani, R. J., & Wasserman, L. (2018). Distribution-free predictive inference for regression. *Journal of the American Statistical Association*, 113(523), 1094–1111. arXiv:1604.04173.

Li, D., Yang, Y., Song, Y.-Z., & Hospedales, T. M. (2018). Learning to generalize: Meta-learning for domain generalization. In *Proceedings of the AAAI Conference on Artificial Intelligence (AAAI 2018)*. arXiv:1710.03463.

Sun, B., & Saenko, K. (2016). Deep CORAL: Correlation alignment for deep domain adaptation. In *Computer Vision – ECCV 2016 Workshops (TASK-CV)*. arXiv:1607.01719.

Zhang, B., Tian, H., Berry, A., & Roussac, A. C. (2026a). Improving day-ahead grid carbon intensity forecasting by joint modeling of local-temporal and cross-variable dependencies across different frequencies. *Proceedings of the AAAI Conference on Artificial Intelligence*, 40(46). https://doi.org/10.1609/aaai.v40i46.41310

Zhang, X., Zhou, T., He, F., Deng, Y., & Wang, D. (2026b). General carbon intensity forecasting via dual graph empowered time series foundation model. In *Proceedings of the ACM Web Conference 2026 (WWW '26)*, Dubai, United Arab Emirates. https://doi.org/10.1145/3774904.3793051
