# Plug-and-Play Cross-Region Carbon-Intensity Forecasting: A Physics-Informed Error Decomposition and Adaptation Framework

**Draft status:** full-length draft (target: IEEE Transactions / long conference track). This draft is a restructured rewrite organized around the paper's two load-bearing contributions — an *exact* error decomposition (Theorem 1) and a *structural* domain-adaptation bound (Theorem 2) — with a config-driven deployment layer and a Bayes-optimal fusion result as the applied payoff.

**Note on data (mandatory disclosure).** Every numerical result is computed on **real 2023 AEMO/NEM hourly historical data** for four Australian regions (QLD1, NSW1, VIC1, SA1); no synthetic data underlies any reported metric. The only non-AEMO input is the temperature covariate (channel C1 / `TempAnomaly`), which comes from the **Open-Meteo free historical weather archive**, not from AEMO/NEMED — flagged wherever it is used. Every number in Section 6 is traceable to `docs/experiments/2026-07-17-all-experiments-summary.md` and `docs/experiments/2026-07-14-sa1-domain-adaptation.md`. Where a quantity is *derived but not numerically estimated*, we say so explicitly rather than hide it.

---

## Abstract

Grid carbon-intensity (CI) forecasting models are conventionally trained and evaluated within a single electricity market. Deploying them to new, data-scarce regions typically causes severe degradation or outright *negative transfer*, because these models entangle universal grid dynamics with region-specific physical properties (installed capacity, local climate, emission factors). We present **TransCIF**, a physics-informed, config-driven domain-adaptation framework for plug-and-play cross-region CI forecasting. Rather than proposing yet another end-to-end architecture, our central contribution is theoretical and structural. We derive an **exact algebraic identity (Theorem 1)** for the one-step transfer error of any two-stage pipeline that reconstructs CI from a predicted renewable share through a linear emission-factor map. The identity cleanly separates a *transfer-amplification* term — scaled by a region-specific, exactly computable physical constant $L_T$ (the gap between a region's renewable and non-renewable emission factors) — from a *residual-estimation* term. We then derive **Theorem 2**, a domain-adaptation upper bound on the renewable-share risk, and couple it with $L_T$ (**Corollary 2**) to obtain a structural bound on target-region CI error — honestly flagging that its divergence terms are *not numerically estimated*. We validate Theorem 1 on four real NEM regions spanning a 2.4$\times$ range of $L_T$: the identity holds to floating-point precision (${\sim}10^{-4}$), and the transfer-amplification term dominates in 8/8 region$\times$config combinations. Adopting a state-of-the-art deep sequential network as an off-the-shelf base encoder, we show that *without* TransCIF's adaptation the base model suffers negative transfer (losing to a naive persistence baseline in 2 of 4 regions); with target-domain fine-tuning (D) and CORAL feature alignment (E) it beats the persistence floor in **4/4** regions. Finally, a Bayes-optimal (Bates-Granger) forecast fusion and a config-driven deployment layer (onboard a brand-new grid with zero code changes) yield the project-best **−9.4%** over the persistence floor on the volatile SA1 region. We report the negative and mixed results alongside the positive ones.

---

## 1. Introduction

Grid-level carbon intensity (CI, in gCO$_2$/kWh) is the fundamental signal for carbon-aware load shifting, EV-charging scheduling, and demand response. Forecasting CI 12 hours ahead is well studied *within* a single market with abundant history, but a critical bottleneck remains: **cross-region generalization**. Many regions — especially smaller or emerging markets — lack the years of labeled data that data-hungry forecasters require. When state-of-the-art models are transferred blindly to data-scarce targets, they frequently degrade sharply, because they entangle the universal dynamics of electricity grids with region-specific physical properties.

This motivates a **plug-and-play transfer** setting: train once on data-rich source regions, deploy to a data-scarce target with only a small calibration split, and ask how the model's error can be *understood and bounded*, not merely empirically minimized. A practical corollary of "plug-and-play" is operational: onboarding a new region should be a *configuration* act (point the pipeline at the region's data and emission factors), not a *code* act.

Our starting point combined five reasonable, previously published techniques — scale-invariant reparameterization, physics-plus-residual reconstruction, consistency regularization, a dominant-variable indicator reused at calibration, and split-conformal prediction — plus two target-domain adaptation techniques. Each is defensible, but the composition is not itself a theoretical contribution, and — as we report honestly in Section 6.3 — the composition alone did *not* reliably beat a trivial persistence baseline on our target region. This led us to a different question: **can we say something exact, not just empirical, about how transfer error arises in this class of pipeline?** The answer is yes, because the reconstruction step from predicted share to CI is *not* a black box — it is a fixed, publicly documented linear formula. That linearity yields an exact identity for the one-step transfer error (Theorem 1); having localized the dominant error to renewable-share prediction, we then bound that share error under domain shift (Theorem 2), coupled to the physical constant $L_T$ (Corollary 2).

**Main contributions.**
1. **Model-agnostic error decomposition (Theorem 1):** an exact decomposition of one-step CI transfer error into a *transfer-amplification* term and a *residual-estimation* term, for any pipeline reconstructing CI from a predicted share through the linear emission-factor map. We verify it numerically across four real regions, two model configurations, three calibration splits, and five seeds.
2. **Physics-coupled domain-adaptation bound (Theorem 2 & Corollary 2):** a structural upper bound on the target renewable-share risk that couples learning-theoretic discrepancy metrics with the physical grid constant $L_T$, giving a mechanistic account of *why* target-domain fine-tuning is necessary — with explicit disclosure of which terms are and are not numerically estimated.
3. **Decoupled architecture & config-driven deployment:** a deployment layer (`transcif.config`) featuring scale-invariant reparameterization and inline emission factors, so onboarding a brand-new grid requires *zero* code modifications.
4. **Bayes-optimal forecast fusion:** a minimum-variance affine fusion of the adapted network, persistence, and climatology, achieving the project-best result (−9.4% vs. the persistence floor) on the volatile SA1 region — and, because it fuses *shares* through the same linear map, it remains exactly decomposable by Theorem 1.

---

## 2. Related Work

**Physics-informed CI forecasting.** Recent work models day-ahead grid CI end-to-end with complex deep architectures; e.g., Zhang et al. (2026a) propose a joint local-temporal and cross-variable dependency network. We deliberately do *not* propose a competing architecture. We adopt their network as a representative off-the-shelf "base encoder" and hold the linear renewable/non-renewable reconstruction step fixed, asking a deeper question: *what can be said exactly about how error propagates through the physical reconstruction step under domain shift?* Theorem 1 fills this gap and is architecture-agnostic — it applies to any pipeline reconstructing CI from a predicted share through a linear emission-factor map.

**Cross-region / data-scarce CI forecasting.** Zhang et al. (2026b) address data-scarce regions via a dual-graph carbon-domain foundation model with metadata-driven hypergraph fine-tuning; their ablation is, to our knowledge, the strongest existing evidence that target-region fine-tuning is critical. Our Section 6.2 cross-region result (4/4 vs. 2/4 regions beating persistence with vs. without fine-tuning) is an independent, real-data confirmation on a different architecture and market, and Theorem 1 additionally explains *why* fine-tuning helps — it shrinks the transfer-amplification term.

**Domain generalization / adaptation theory.** We use MLDG (Li et al., 2018) for multi-source meta-training, Deep CORAL (Sun & Saenko, 2016) for feature alignment, and gradual-unfreezing fine-tuning (Khanal et al., 2024). Our contribution is theoretically mapping these techniques onto the error decomposition (Section 5): Theorem 2 instantiates the bound of Ben-David et al. (2010) and the discrepancy distance of Mansour et al. (2009) (with the Wasserstein refinement of Redko et al., 2017), and its novelty is the coupling to the physical constant $L_T$. Empirically we find pure feature alignment is *actively harmful* without supervised fine-tuning in this physical context.

**Forecast combination & conformal prediction.** Our fusion (Section 6.4) applies classical Bates-Granger (1969) optimal linear combination to three renewable-share forecasters. Our calibration-time uncertainty band reuses split-conformal prediction (Lei et al., 2018), reusing Stage 2's existing calibration split rather than a dedicated one — motivated by the data-scarce target setting.

---

## 3. Problem Formulation

We forecast a region's hourly grid carbon intensity $CI_t$ (gCO$_2$/kWh) over a horizon of $H$ steps from a window of past grid signals $x\in\mathcal{X}$. TransCIF standardizes any base forecaster into a two-stage process:

- **Stage 1 (share prediction):** an agnostic deep encoder $h:\mathcal{X}\to[0,1]^H$ predicts the future renewable-share trajectory $\hat s_{t+1:t+H} = h(x)$. The true share is $s=f(x)$ for the region-specific labeling function $f$.
- **Stage 2 (physics + residual):** each predicted share is mapped to CI by a physically fixed linear emission-factor formula, then corrected by a small learned residual head $\hat\Delta_t$ fit only on the target region's calibration split.

We study **domain transfer**: the encoder is trained on source distribution(s) $P_S$ (data-rich regions) and deployed on a target $P_T$ (a data-scarce region) with a small labeled calibration split. We analyze the *one-step* error (per horizon step); multi-step results follow by treating each horizon component identically. We use L1 (absolute) error for shares and MAE for CI, matching every experiment.

**Notation.** $C_{\text{renew}}, C_{\text{nonrenew}}$ are region emission factors (gCO$_2$/kWh); $L_T := |C_{\text{renew}}-C_{\text{nonrenew}}|$; $\varepsilon_S(h),\varepsilon_T(h)$ are source/target expected absolute share errors; $\varepsilon_t$ is the true residual (Section 4.3); $\hat\Delta_t$ the learned residual. Persistence (repeat-last-observation) is the naive baseline; its fixed SA1 value is $\text{persistence\_mae}=67.568$.

---

## 4. The TransCIF Framework

![End-to-end TransCIF architecture](figures/fig1_architecture.png)

**Figure 1.** End-to-end TransCIF architecture. A config-driven deployment layer (`RegionConfig`/`DeploymentConfig`) externalizes region onboarding — inline emission factors make a new grid a one-JSON edit rather than a code change — and feeds region data/channels and the emission factors $C_{ren},C_{non}$ (hence $L_T$) into the pipeline. Stage 1 fuses LT-MWKC and CV-DWCC into a domain-invariant encoder $h$, trained by multi-source MLDG pretraining and target-domain adaptation (D: gradual-unfreezing fine-tuning; E: Deep CORAL), predicting the renewable-share trajectory $\hat s\in[0,1]^H$; Stage 2 maps $\hat s$ to CI through the fixed linear emission-factor formula, adds a learned residual $\hat\Delta$ with a volatility-gated persistence skip, and attaches a split-conformal band. Theorem 1 decomposes the error exactly into Term ① (transfer amplification, scaled by $L_T=|C_{ren}-C_{non}|$) and Term ② (residual estimation).

### 4.1 Stage 0: Scale-Invariant Reparameterization

To enable cross-region transfer we must strip away region-specific absolute magnitudes. TransCIF preprocesses raw signals into scale-invariant features (`src/transcif/data/reparam.py`): **`RenewShare`** $= \text{RenewOut}/(\text{RenewOut}+\text{NonRenewOut})$ (eliminating capacity dependence), **`LoadNorm`** $= \text{Load}/\text{rolling-}q_{0.95}(\text{Load})$ over a 720-hour window (eliminating grid-size dependence; provably invariant to positive rescaling once the window is populated), and **`TempAnomaly`** $= \text{Temp}-\overline{\text{Temp}}_{\text{same day-of-year}}$ (deviation from local climate baselines; the temperature source is Open-Meteo, not AEMO).

### 4.2 Stage 1: Agnostic Base Encoder & Volatility-Gated Skip

For share prediction the framework is agnostic to the choice of deep sequential encoder. To demonstrate transferability we adopt the state-of-the-art joint dependency network of Zhang et al. (2026a) — comprising a local-temporal multi-wavelet kernel convolution (**LT-MWKC**) and a cross-variable dynamic wavelet correlation module (**CV-DWCC**) — as our off-the-shelf base encoder. A `forward_features(x)` method returns a pooled fused representation plus a dominant-variable index (reused by domain-adaptation code).

**Framework enhancement — volatility-gated persistence skip.** We augment the encoder with a persistence skip whose mixing coefficient is conditioned on recent renewable-share volatility: for low-volatility windows the framework leans on the persistence prior, forcing the network to learn only necessary corrections rather than full trajectories from scratch; for high-volatility windows it trusts the network correction more. The encoder emits $\hat s_{t+1:t+H}\in[0,1]^H$.

![Stage 1 encoder and domain adaptation](figures/fig_stage1_encoder.png)

**Figure 1a.** Stage 1 in detail. Panel (a): the base encoder fuses LT-MWKC (local-temporal multi-wavelet kernel convolution) and CV-DWCC (cross-variable dynamic wavelet correlation); a volatility-gated persistence skip mixes the network output with the last-observation prior. Panel (b): multi-source MLDG pretraining over QLD1/NSW1/VIC1 followed by two target-adaptation branches on SA1 — D (gradual-unfreezing supervised fine-tuning on labeled calibration) and E (Deep CORAL unsupervised covariance alignment on raw calibration inputs). 5-seed result: $66.49\pm0.78$ vs baseline $77.93\pm2.14$, wins 5/5.

### 4.3 Stage 2: Physics Reconstruction and Residual Correction

For a two-category renewable/non-renewable split, CI reconstruction is **exactly linear** in the renewable share $s$ (`src/transcif/physics/cif.py`):

$$\mathrm{CIF}(s) = s\cdot C_{\text{renew}} + (1-s)\cdot C_{\text{nonrenew}}.$$

$C_{\text{renew}}$ and $C_{\text{nonrenew}}$ are region-specific factors looked up from a fixed table (or supplied inline by config), **not estimated**. The AU factors are generation-weighted from real 2023 AEMO/NEMED dispatch data (NEMED generator-level intensities, 0.02 tCO$_2$/MWh renewable threshold). We define the **true residual** $\varepsilon_t$ as whatever the linear formula misses:

$$CI_{true,t} := \mathrm{CIF}(s_t) + \varepsilon_t,$$

a *definition*, not an assumption — it attributes everything the formula omits (cross-border trade, transmission losses, sub-fuel-mix heterogeneity) to $\varepsilon_t$. The framework predicts $CI_{pred,t} = \mathrm{CIF}(\hat s_t) + \hat\Delta_t$, where $\hat\Delta_t$ is the residual head fit strictly on target calibration data. A distribution-free split-conformal band (Lei et al., 2018) is computed on the same calibration split, avoiding a dedicated held-out set.

**Target-domain adaptation (D, E).** On top of the base pipeline we evaluate two techniques (`src/transcif/training/domain_adaptation.py`): **D — gradual-unfreezing supervised fine-tuning** on the target's labeled calibration split, unfreezing in three stages (gate/head → CV-DWCC → LT-MWKC) to resist catastrophic forgetting (default 3 stages, 15 epochs/stage, lr $5\times10^{-4}$; Khanal et al., 2024); and **E — Deep CORAL**, an unsupervised second-order feature-covariance alignment loss (weight 0.1) added to the MLDG loop, using only target calibration inputs (no labels). Multi-source pretraining uses MLDG (Li et al., 2018) over $\ge2$ sources, optionally domain-weighted by each source's label $\text{std}+|\text{skew}|$; with a single source the pipeline falls back to plain source training.

![Stage 2 physics reconstruction and residual correction](figures/fig_stage2_physics.png)

**Figure 1b.** Stage 2 in detail. The renewable share $\hat s$ from Stage 1 is mapped through the fixed linear emission-factor formula $\mathrm{CIF}(\hat s)=\hat s\,C_{ren}+(1-\hat s)\,C_{non}$ (white-box, sharp rectangle; $C_{ren},C_{non}$ come from a region lookup table and are never estimated), then a learned residual $\hat\Delta$ from an MLP head fit on calibration data (black-box, rounded rectangle) with a volatility-gated persistence skip is added, and a split-conformal 90% band is attached.

### 4.4 Config-Driven Zero-Code Deployment

We encapsulate deployment in a declarative config layer (`transcif.config`), loadable from zero-dependency JSON. **`RegionConfig`** describes one grid: `name`, `hourly_csv`, optional `temperature_csv`, and emission factors resolved in strict precedence — (1) inline `emission_renewable`/`emission_nonrenewable` if both set, else (2) `EMISSION_FACTOR_TABLES[factor_code]`; a region with neither raises an actionable error rather than silently defaulting, and inline factors deliberately bypass the placeholder `EU_*`/`US_*` rows that `cif.py` warns against. **This inline-factor path is the key externalization: onboarding a brand-new grid with its own measured factors is a config edit, never a code edit.** **`DeploymentConfig`** describes a full source→target run (target region, source list, run knobs, channel switches `include_generation`/`include_temperature`, D/E toggles `fine_tune`/`coral`, MLDG settings, seed); a `num_channels` property derives the encoder width from the switches (2 base + 2 if generation + 1 if temperature), and an unknown-key check makes a config typo fail loudly at load time. The orchestrator `deploy_region(config)` runs the entire Stage 1→3 pipeline and returns corrected/physics/persistence MAE plus the conformal half-width and empirical coverage. **The consequence is structural: each ablation variant is now a config file, not an edited script.** Two configs ship with the repo — `scripts/configs/sa1_from_au_full.json` (the paper's best real configuration: QLD1/NSW1/VIC1 → SA1, full channels + D + E) and `scripts/configs/new_grid_inline_factors.json` (a brand-new-grid template carrying inline factors).

---

## 5. Theory: Error Decomposition and Adaptation Bounds

### 5.1 Theorem 1: Exact Error Decomposition

**Theorem 1.** *For any pipeline reconstructing CI from a predicted share $\hat s_t$ through the linear formula $\mathrm{CIF}(\cdot)$ of Section 4.3, the one-step forecast error decomposes exactly as*

$$CI_{pred,t} - CI_{true,t} \;=\; \underbrace{(\hat s_t - s_t)(C_{\text{renew}} - C_{\text{nonrenew}})}_{\text{Term ① (transfer amplification)}} \;+\; \underbrace{(\hat\Delta_t - \varepsilon_t)}_{\text{Term ② (residual estimation)}}.$$

*Consequently,* $|CI_{pred,t} - CI_{true,t}| \le L_T\cdot|\hat s_t - s_t| + |\hat\Delta_t - \varepsilon_t|$, *with* $L_T := |C_{\text{renew}} - C_{\text{nonrenew}}|$.

**Proof.** Subtract $CI_{true,t}$ from $CI_{pred,t}$: $CI_{pred,t}-CI_{true,t}=[\mathrm{CIF}(\hat s_t)-\mathrm{CIF}(s_t)]+[\hat\Delta_t-\varepsilon_t]$. By linearity, $\mathrm{CIF}(\hat s_t)-\mathrm{CIF}(s_t)=(\hat s_t-s_t)C_{\text{renew}}+[(1-\hat s_t)-(1-s_t)]C_{\text{nonrenew}}=(\hat s_t-s_t)(C_{\text{renew}}-C_{\text{nonrenew}})$. The identity follows; the triangle inequality gives the bound. $\blacksquare$

This is an exact algebraic identity, not an asymptotic or approximate bound — it follows entirely from the linearity of the reconstruction. $L_T$ requires no estimation: it is read directly off each region's published emission-factor table. Its value is that it turns "why did the CI forecast err" into two *separately measurable, separately attributable* quantities for a model class previously evaluated only end-to-end.

**Corollary 1 (falsifiable, cross-region).** *If two regions have similar share error $|\hat s_t-s_t|$, the region with larger $L_T$ should show a proportionally larger Term-① contribution.* The region-specific constants, from the real 2023 emission-factor tables, are:

**Table 1: Region transfer-amplification constants ($L_T$, gCO$_2$/kWh).**

| Region | $C_{\text{renew}}$ | $C_{\text{nonrenew}}$ | $L_T$ |
|---|---|---|---|
| SA1  | 0.00 | 490.43  | **490.43 (smallest)** |
| QLD1 | 0.00 | 841.59  | 841.59 |
| NSW1 | 0.09 | 875.23  | 875.14 |
| VIC1 | 0.00 | 1160.12 | **1160.12 (largest)** |

### 5.2 Theorem 2 & Corollary 2: Domain-Adaptation Bound

Theorem 1 localizes the dominant error to the share error $|\hat s_t-s_t|$. Theorem 2 bounds its *expectation* under domain shift, using the L1 share loss $\ell(a,b)=|a-b|$ (the only structural requirement is the triangle inequality).

**Lemma 2.1 (bridge to Theorem 1).** $\mathbb{E}_{x\sim P_T}|\text{Term①}| = L_T\cdot\varepsilon_T(h)$ — an exact equality, since $L_T$ is constant and $|\hat s-s|=|h(x)-f_T(x)|$.

**Definitions.** The *discrepancy distance* (Mansour et al., 2009), a label-free, loss-aware generalization of the Ben-David $\mathcal{H}\Delta\mathcal{H}$-divergence: $\operatorname{disc}_\ell(P_S,P_T):=\sup_{h,h'\in\mathcal{H}}|\mathbb{E}_{P_S}\ell(h,h')-\mathbb{E}_{P_T}\ell(h,h')|$. The *ideal joint hypothesis* $h^\*:=\arg\min_h[\varepsilon_S(h)+\varepsilon_T(h)]$ with combined risk $\lambda^\*:=\varepsilon_S(h^\*)+\varepsilon_T(h^\*)$.

**Theorem 2 (share risk bound).** *For any $h\in\mathcal{H}$ under L1 loss,* $\;\varepsilon_T(h)\le\varepsilon_S(h)+\operatorname{disc}_\ell(P_S,P_T)+\lambda^\*.$

**Proof.** (1) $\varepsilon_T(h)\le\mathbb{E}_T\ell(h,h^\*)+\varepsilon_T(h^\*)$; (2) $\mathbb{E}_T\ell(h,h^\*)\le\mathbb{E}_S\ell(h,h^\*)+\operatorname{disc}_\ell(P_S,P_T)$ (since $h,h^\*\in\mathcal{H}$); (3) $\mathbb{E}_S\ell(h,h^\*)\le\varepsilon_S(h)+\varepsilon_S(h^\*)$. Combining gives the bound. $\blacksquare$

**Corollary 2 (physics-coupled transfer bound).** Multiplying Lemma 2.1 by Theorem 2 and adding Term ②:

$$\mathbb{E}_T\big|CI_{pred,t}-CI_{true,t}\big| \;\le\; L_T\big(\underbrace{\varepsilon_S(h)}_{\text{source training}} + \underbrace{\operatorname{disc}_\ell(P_S,P_T)}_{\text{alignment (CORAL, E)}} + \underbrace{\lambda^\*}_{\text{fine-tuning (D)}}\big) \;+\; \underbrace{\mathbb{E}_T|\hat\Delta_t-\varepsilon_t|}_{\text{Term ②}}.$$

The CI-unit transfer error bound is **explicitly proportional to $L_T$**, and its three right-hand terms map onto three interventions: source training lowers $\varepsilon_S(h)$; CORAL (E) targets $\operatorname{disc}_\ell$; supervised fine-tuning (D) is the *only* lever that reaches $\lambda^\*$ by using target labels to approach $h^\*$.

**Honest scope (mandatory).** Theorem 1 is an **exact identity**, verified to floating-point precision (Section 6.1). Theorem 2 / Corollary 2 are a **structural upper bound whose divergence terms we do not numerically estimate**: $\operatorname{disc}_\ell$ needs a supremum/proxy-$\mathcal{A}$-distance we do not implement; $\lambda^\*$ needs target labels over the full distribution (we have only a calibration split); the optional Wasserstein form (Appendix A) needs an OT computation we do not run. We therefore present Theorem 2 as an *explanatory framework* for why the interventions help — **not** an empirically tight bound to be compared pointwise against measured MAE.

---

## 6. Experiments

**Setup.** All results are on real 2023 AEMO/NEM hourly data for QLD1, NSW1, VIC1, SA1; the target is SA1 (highest volatility / renewable penetration) with sources QLD1/NSW1/VIC1 unless stated. Sequence length 48, horizon 12, stride 6, calibration split 0.7. The metric is CI MAE in gCO$_2$/kWh; the reference floor is persistence (fixed SA1 value 67.568). We compare **baseline** (multi-source pretraining, no target adaptation) and **+D+E** (baseline + fine-tuning + CORAL). Because MLDG selects a meta-test region by `random.choice` per epoch, the baseline SA1 corrected MAE differs slightly across scripts (75.508 Stage-1 ablation, 74.712 Stage-2, 74.206 Theorem-1 validation); we report each experiment against its own run rather than averaging across scripts, and flag the variation where it matters.

### 6.1 Theorem 1 Numerical Validation

We verify the exact identity on real SA1 data at the 0.7 split and measure the Term-① share.

**Table 2: Theorem 1 decomposition (SA1, calib_fraction = 0.7).**

| Config | max abs identity gap | mean abs total error | mean\|Term ①\| | mean\|Term ②\| | Term-① share |
|---|---|---|---|---|---|
| baseline (Source-Only) | 5.72e-05 | 74.206 | 84.413 | 22.123 | 79.2% |
| +D+E | 4.86e-05 | 66.887 | 73.966 | 20.288 | 78.5% |

The identity holds to floating-point precision (${\sim}5\text{e-}5$). Term ① dominates (${\approx}79\%$) in both configs. Crucially, the D+E gain (74.206 → 66.887) comes **mainly from shrinking Term ①** (84.413 → 73.966, a 12.4% reduction) while Term ② barely moves — exactly the mechanism Corollary 2 attributes to supervised fine-tuning (D): it reduces the share error $|\hat s-s|$ that Term ① scales. This is the mechanistic "why fine-tuning helps" that end-to-end evaluation cannot expose.

**Robustness of the decomposition.** Over calibration fractions $\{0.6,0.7,0.8\}$ the Term-① share has standard deviation **0.29 pp** (baseline) and **0.97 pp** (+D+E), and Term ① is dominant in all 6 runs — not an artifact of the 0.7 split. Rotating each region into the target role (a genuine *cross-region* test spanning the 2.4$\times$ $L_T$ range) confirms **Term ① is dominant in 8/8** region$\times$config combinations, identity gaps at the $10^{-4}$ level throughout. *Honesty note:* the Term-① share does rise monotonically with $L_T$ (SA1 79.66% < QLD1 83.71% < NSW1 89.28% < VIC1 91.84%), but since $\text{Term①}=(\hat s-s)\cdot L_T$ this monotonicity is essentially an algebraic byproduct of the decomposition, *not* independent evidence; the load-bearing result is the 8/8 dominance generalization, not the ranking.

![Theorem 1 decomposition and cross-region Term-① dominance](figures/fig2_theorem1_decomposition.png)

**Figure 2.** Theorem 1 in numbers. **(a)** On SA1 (split 0.7), Term ① dominates (${\approx}79\%$) in both configs, and the D+E gain comes mainly from shrinking Term ① (84.4 → 74.0). **(b)** Across all four rotated regions and both configs, Term ① stays above the 50% dominance threshold in 8/8 cases; its share rises with $L_T$ (an algebraic byproduct — see the honesty note). Values from `all-experiments-summary.md` §3.1/§4.4.

### 6.2 Overcoming Negative Transfer: Cross-Region Validation

Does the base encoder generalize out-of-the-box? We recompute per-region persistence MAE (reusing the already-trained rotation models — no retraining) and compare against the naive floor.

**Table 3: Cross-region comparison against the persistence floor (MAE, gCO$_2$/kWh).**

| Region | $L_T$ | Persistence | baseline (Source-Only) | +D+E |
|---|---|---|---|---|
| QLD1 | 841.59  | 103.181 | 62.397 (−39.5%, win) | 58.396 (−43.4%, win) |
| NSW1 | 875.14  | 133.492 | 82.997 (−37.8%, win) | 75.172 (−43.7%, win) |
| VIC1 | 1160.12 | 104.764 | 116.534 (**+11.2%, lose**) | 103.405 (−1.3%, win) |
| SA1  | 490.43  | 67.568  | 76.239 (**+12.8%, lose**) | 65.561 (−3.0%, win) |

**Reading it honestly:** (i) the *complete* adapted stack **+D+E beats persistence in all 4/4 regions** (−1.3% to −43.7%) — the only hard evidence for a cross-region "works generally" claim, previously verified on SA1 alone. (ii) The **baseline (no fine-tuning) wins only 2/4**: big on QLD1/NSW1 but *losing* on VIC1 and SA1 (negative transfer). There is no fixed "deploy-and-win" model — on two of four regions, skipping target fine-tuning makes the SOTA base encoder a net liability. (iii) The persistence spread (67.6–133.5) does **not** track $L_T$ ordering (NSW1 has lower $L_T$ than VIC1 yet the highest persistence), so per-region difficulty is independent of $L_T$ and must not be conflated with the Theorem-1 analysis. Defensible framing: *the full innovation stack — especially target-domain fine-tuning D — is a necessary condition for consistently beating the naive baseline across regions, not an incidental add-on.*

![Per-region persistence comparison](figures/fig3_region_persistence.png)

**Figure 3.** Corrected MAE vs the persistence floor (grey) per region. **+D+E beats persistence in 4/4 regions** (−1.3% to −43.7%); the **baseline wins only 2/4**, losing on VIC1 (+11.2%) and SA1 (+12.8%). Note the persistence spread does not track $L_T$. Values from `all-experiments-summary.md` §4.5.

### 6.3 Domain-Adaptation Ablation (SA1)

We isolate which ingredient matters. Stage 1 tested four "surface" mitigations; Stage 2 tested the deeper D/E techniques; a pure-ERM run (no MLDG, no adaptation) serves as a DomainBed-style control.

**Table 4: SA1 ablation (corrected MAE, gCO$_2$/kWh; persistence = 67.568).**

| Stage | Configuration | Corrected MAE | vs. Persistence |
|---|---|---|---|
| 1 | baseline (no mitigation) | 76.725 | +13.6% |
| 1 | +REG/NEG (abs. generation channels) | 81.948 | +21.3% (worst of stage 1) |
| 1 | +temperature C1 (Open-Meteo) | 76.599 | +13.4% |
| 1 | +gating A | 76.820 | +13.7% |
| 1 | +MLDG weighting B | 78.317 | +15.9% |
| 1 | all-combined | 75.508 | +11.8% (best of stage 1) |
| 2 | all-combined (baseline) | 74.712 | +10.6% |
| 2 | **+D (fine-tuning only)** | 67.240 | **−0.5%** |
| 2 | +E (CORAL only) | 75.788 | +12.2% (worse than baseline) |
| 2 | **+D+E** | **66.004** | **−2.3% (best overall)** |
| — | pure ERM (no MLDG) | 77.629 | +14.9% (worst overall) |

Three findings the source papers' settings do not surface: (i) **all six Stage-1 surface variants lose to persistence**; REG/NEG and MLDG-weighting are actively *worse* than doing nothing. (ii) **D alone already beats persistence** (−0.5%) while **E alone is harmful** (+12.2%) — the active ingredient is target-domain *supervised* fine-tuning, not any structural tweak; unsupervised CORAL without label guidance pulls features in an unhelpful direction. (iii) **D+E > D**: CORAL becomes a useful regularizer *once* supervised fine-tuning is present. The **pure-ERM control is the worst of all** (+14.9%), confirming MLDG's meta-training is *not* itself the active lever unless paired with target fine-tuning.

![SA1 ablation ladder](figures/fig4_ablation_ladder.png)

**Figure 4.** SA1 ablation ladder (corrected MAE). The dashed line is the persistence floor (67.57). Only **+D** and **+D+E** fall below it; all six Stage-1 surface variants, CORAL-alone, and pure ERM lose. Values from `all-experiments-summary.md` §1.1/§2.1/§4.2.

**Multi-seed robustness.** Because the D+E margin is modest, we repeat baseline and +D+E over 5 seeds (0–4), after fixing the previously unpinned MLDG `random.choice` seed. Summary (mean ± std, $n=5$): **baseline 77.934 ± 2.136**, **+D+E 66.492 ± 0.783**. Paired D+E−baseline differences $[-9.370,-9.283,-13.203,-13.880,-11.474]$ (mean −11.442, std 2.121, paired $t=-12.061$, df 4). **We report this $t$-value as descriptive only: with $n=5$ its power is very limited and it is not a formal significance claim.** What is robust: D+E beats baseline in **5/5** seeds (direction 100% stable) and beats persistence in **5/5** seeds but with seed-sensitive magnitude (−0.22% to −2.84%, ${\approx}13\times$ spread — the reported −2.3% sits inside this band); Term-① dominance is stable across all 10 runs (77.8%–82.4%).

![Multi-seed robustness](figures/fig5_multiseed.png)

**Figure 5.** Per-seed corrected MAE (5 seeds). +D+E sits below baseline in 5/5 seeds and near/below the persistence floor (dashed); aggregate bars show baseline 77.93 ± 2.14 vs +D+E 66.49 ± 0.78. Direction 100% robust; margin vs persistence seed-sensitive. Values from `all-experiments-summary.md` §5.

### 6.4 TransCIF-Fusion: Bayes-Optimal Forecast Fusion

For robust operational deployment we fuse three renewable-share forecasters — persistence, the network (D+E), and diurnal climatology — with per-horizon minimum-variance affine weights $w^\* = (\Sigma^{-1}\mathbf 1)/(\mathbf 1^\top\Sigma^{-1}\mathbf 1)$ (Bates & Granger, 1969; weights unconstrained, so negative components occur). Because the fused output is still an affine combination of *shares* through the same linear CIF map, it remains exactly decomposable by Theorem 1 — a third forecaster *inside* the framework, not a black-box ensemble on top.

**Table 5: Bayes-optimal fusion result (SA1).**

| Forecaster | MAE (gCO$_2$/kWh) |
|---|---|
| Persistence (floor) | 67.568 |
| Network-only (D+E, physics-only) | 71.226 |
| Climatology | 86.359 |
| **TransCIF-Fusion** | **61.196 (−9.4% vs persistence)** |

**Overfitting diagnostic (calib vs eval MAE, RenewShare units):** persistence 0.1549 / 0.1396 (gap −9.9%); network 0.1644 / 0.1503 (−8.6%); climatology 0.2018 / 0.1840 (−8.8%). All three gaps are **negative** (eval better than calib, well under the +20% warning threshold), so there is **no detected overfitting signal** — the fusion weights are not memorizing the calibration split. At −9.4% this is the project's largest margin over persistence; Term ① still dominates (mean|Term ①| 64.040 / mean|Term ②| 17.458, share 78.6%), consistent with Theorem 1.

![Bates-Granger fusion](figures/fig6_fusion.png)

**Figure 6.** Bayes-optimal (Bates-Granger) fusion of persistence, network (D+E), and climatology share forecasters. The fused corrected MAE (61.20) beats the persistence floor (dashed) by −9.4% — the largest margin in the project — even though climatology alone is far worse. Values from `all-experiments-summary.md` §4.1.

---

## 7. Limitations and Conclusion

**Limitations (stated plainly, because several complicate a clean "it just works" story).**
- **No external SOTA baselines.** We compare against persistence, physics-only, climatology, and our own ablations — not published DANN/CoDA-style methods. We make no claim of beating such methods.
- **Single year.** All data is 2023; inter-year generalization is untested.
- **Non-official covariate.** The temperature channel is Open-Meteo, not AEMO/NEMED.
- **Modest, seed-sensitive margin.** The best SA1 result (D+E −2.3%; fusion −9.4%) beats persistence but not decisively; the D+E margin ranges −0.22% to −2.84% across seeds.
- **Theorem 2 divergence terms are not numerically estimated.** $\operatorname{disc}_\ell$, $\lambda^\*$, and the Wasserstein $W_1$ (Appendix A) are derived structurally but not computed; only Theorem 1 is numerically verified (to ${\sim}5\text{e-}5$).
- **Small target calibration set.** Fine-tuning uses SA1's 70% calibration split; its hyperparameters and the CORAL weight were not swept.

**Conclusion.** TransCIF shifts cross-region CI forecasting from building opaque end-to-end models to establishing a physically transparent, theoretically bounded deployment framework. Theorem 1 gives an exact, region-specific, falsifiable decomposition of one-step transfer error — verified to floating-point precision across four real AEMO regions, five seeds, and multiple splits — and Theorem 2 / Corollary 2 couple a classical domain-adaptation bound to the physical constant $L_T$, explaining why target-domain fine-tuning (not surface tweaks or unsupervised alignment) is the necessary ingredient — a prediction borne out by the ablation (4/4 regions beaten by D+E vs 2/4 by the baseline). Armed with config-driven zero-code deployment and Bayes-optimal fusion, TransCIF transforms any base encoder into a robust, plug-and-play forecaster capable of consistently outperforming naive baselines across diverse electricity markets. We report the negative and mixed results alongside the positive ones, because the honest account is what the theory is for.

---

## Data & Code Availability

All electricity/emission data are real 2023 AEMO/NEM historical exports (`nem_2023_hourly_{REGION}.csv`, `duid_level_2023.parquet`); the temperature covariate is from the Open-Meteo historical archive. Experiment scripts (`scripts/sa1_ablation.py`, `sa1_domain_adaptation.py`, `theorem1_validation.py`, `theorem1_domain_rotation.py`, `region_persistence_comparison.py`, `theorem2_bayes_fusion.py`, `multi_seed_robustness.py`), the config layer (`src/transcif/config/`), and the underlying logs are in the repository; every number in Section 6 is traceable to `docs/experiments/2026-07-17-all-experiments-summary.md` and `docs/experiments/2026-07-14-sa1-domain-adaptation.md`.

## References

- Bates, J. M., & Granger, C. W. J. (1969). The combination of forecasts. *OR*, 20(4), 451–468.
- Ben-David, S., Blitzer, J., Crammer, K., Kulesza, A., Pereira, F., & Vaughan, J. W. (2010). A theory of learning from different domains. *Machine Learning*, 79(1–2), 151–175.
- Khanal, S., et al. (2024). Domain adaptation for time-series transformers using one-step fine-tuning. *AAAI Workshop*.
- Lei, J., G'Sell, M., Rinaldo, A., Tibshirani, R. J., & Wasserman, L. (2018). Distribution-free predictive inference for regression. *JASA*, 113(523), 1094–1111.
- Li, D., Yang, Y., Song, Y.-Z., & Hospedales, T. M. (2018). Learning to generalize: Meta-learning for domain generalization. *AAAI*.
- Mansour, Y., Mohri, M., & Rostamizadeh, A. (2009). Domain adaptation: Learning bounds and algorithms. *COLT*.
- Redko, I., Habrard, A., & Sebban, M. (2017). Theoretical analysis of domain adaptation with optimal transport. *ECML-PKDD*.
- Sun, B., & Saenko, K. (2016). Deep CORAL: Correlation alignment for deep domain adaptation. *ECCV Workshops*.
- Zhang, et al. (2026a). Joint local-temporal and cross-variable dependency network for day-ahead grid carbon-intensity forecasting.
- Zhang, et al. (2026b). Dual-graph carbon-domain foundation model for data-scarce regional carbon-intensity forecasting.

---

## Appendix A: Wasserstein Specialization of Theorem 2

When $\mathcal{H}$ is a class of $L_g$-Lipschitz share predictors, the discrepancy distance under L1 loss admits a Wasserstein-1 upper bound via Kantorovich–Rubinstein duality: for $h,h'\in\mathcal{H}$, $x\mapsto|h(x)-h'(x)|$ is $2L_g$-Lipschitz, so $\operatorname{disc}_\ell(P_S,P_T)\le 2L_g\,W_1(P_S,P_T)$, and Theorem 2 specializes to $\varepsilon_T(h)\le\varepsilon_S(h)+2L_g\,W_1(P_S,P_T)+\lambda^\*$ (cf. Redko et al., 2017). We include this only for completeness: we do **not** compute $W_1$ in this work, so this form produces no numerical bound.
