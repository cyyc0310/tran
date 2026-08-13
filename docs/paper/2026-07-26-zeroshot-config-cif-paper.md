# Zero-Shot Cross-Region Carbon Intensity Forecasting via Physics-Informed Adaptive Decomposition with Configuration-Driven Domain Transfer

**Draft status:** full-length draft (target: *Applied Energy*; alternates: IEEE TSTE, *Energy*). This draft supersedes `2026-07-22-transcif-full-paper.md` (the fine-tuning-era draft) and is organized around the **config-only zero-shot** paradigm validated in Phases 1–3 of the Q1 submission plan.

**Data disclosure.** All numerical results are computed on real hourly data from 29 electricity regions across three jurisdictions: 4 Australian NEM regions (AEMO 2023), 17 UK DNO regions (National Grid ESO Carbon Intensity API), and 8 US balancing authorities (EIA-930). No synthetic data underlies any reported metric. Every number is traceable to `results/*.json` produced by the scripts listed in the Reproducibility section.

---

## Abstract

Grid carbon intensity forecasting (CIF) is the fundamental signal for carbon-aware computing, EV-charging scheduling, and demand response, yet state-of-the-art forecasters follow a one-region-one-model paradigm: each deployment requires months of local history and dedicated training. We formalize a new problem — **config-only zero-shot cross-region CIF prediction** — in which a model trained on data-rich source regions is deployed to a never-seen target region given only two public configuration scalars (mean renewable share and non-renewable emission factor) and the real-time renewable-share input stream, with *no* target-domain training data and *no* fine-tuning. We propose **TransCIF**, which couples (i) a physics decomposition layer that predicts the renewable share and reconstructs carbon intensity through the region's linear emission-factor map, (ii) a low-capacity, config-conditioned linear decomposition backbone, and (iii) an adaptive persistence gate that arbitrates between model and persistence based on input volatility and configuration. The physics layer admits an exact, architecture-agnostic error identity (Theorem 1): CIF error factorizes into a share-error term amplified by a region-specific, exactly computable constant $L_T$, verified to floating-point precision ($1.3\times10^{-4}$) on all 29 regions, where the amplification term explains 71% of total error on average and $L_T\cdot\varepsilon_{rs}$ predicts realized CIF error with $R^2=0.84$. A transfer-difficulty analysis (Theorem 2) shows zero-shot degradation is governed by the target's position in configuration space: a model-weighted effective source distance correlates with the transfer ratio at $r=0.58$ ($p=0.001$), and transfer difficulty is U-shaped in mean renewable share ($R^2=0.66$, easiest near $\bar{rs}=0.58$). Across a 29-region leave-one-region-out benchmark (5 seeds), zero-shot TransCIF attains a median transfer-efficiency ratio of 1.24 versus a fully supervised PatchTST upper bound trained on each target's own history — i.e., zero-deployment-cost forecasting at ~80% of supervised accuracy — and outright beats the persistence floor in 12/29 regions (median ratio 1.04). A training-free test-time calibration variant, **TransCIF-ZS+** — three closed-form corrections (level anchoring, physics-residual removal, and a self-validated multi-branch fusion whose configuration is re-selected at every forecast origin from replayed past-only errors) using only the target's observable streams — lowers the median ratio to 1.08, beats persistence in **29/29 regions** (median ratio 0.94, worst case 0.995), outperforms even the supervised upper bound outright in 6/29 regions, bounds the worst region at 1.29× supervised, and attains the lowest MAE among all cross-domain methods (persistence, zero-shot CarbonCast, config-only ZS) in 27/29 regions. Against CarbonCast, the domain's leading supervised method, TransCIF wins 5/9 representative regions in the zero-shot setting (ZS+: 8/9 at mean ratio 0.73) while CarbonCast degrades by up to 2.4× when transferred. Split-conformal calibration on the ZS+ forecasts yields valid 90% coverage in 25/29 regions (mean 0.952) without target labels beyond a short calibration stream; under temporal-OOD splits the ZS+ ratio-vs-persistence stays flat at 0.82 across all three split points; and a deployment case study shows supervised models need 30 to >270 days of accumulated target data to match what TransCIF delivers on day one.

**Keywords:** carbon intensity forecasting; zero-shot transfer; domain generalization; physics-informed learning; conformal prediction; carbon-aware computing

---

## 1. Introduction

Decarbonizing electricity consumption requires knowing *when* the grid is clean. Hourly grid carbon intensity (CIF, gCO$_2$/kWh) forecasts drive carbon-aware load shifting in data centers, EV-charging schedulers, and industrial demand response. A decade of work has produced accurate *within-region* forecasters — but the operational reality is that every one of them is trained, tuned, and validated on the history of a single region. Onboarding a new region means collecting months-to-years of local generation and emissions data, retraining, and revalidating: a **one-region-one-model** cost structure that scales linearly with the number of grids and excludes precisely the data-scarce, newly instrumented regions where carbon-aware control could matter most.

This paper asks a deliberately extreme question: *how well can a new region be forecast with zero local training data?* We formalize **config-only zero-shot cross-region CIF prediction**: a model is trained once on a pool of source regions; at deployment the target region supplies only (a) two public configuration scalars — its mean renewable share $\bar{rs}$ and its non-renewable emission factor $ef_{nr}$, both obtainable from public capacity registries and grid operator reports at ±10–20% accuracy — and (b) its real-time renewable-share input stream. No target CIF history, no fine-tuning, no per-region hyperparameters.

Three observations make this tractable. First, for a two-category renewable/non-renewable split, the map from renewable share to carbon intensity is *exactly linear* with region-specific coefficients: $\mathrm{CIF}(rs) = rs \cdot ef_r + (1-rs)\cdot ef_{nr}$. Predicting the **share** instead of the CIF removes the dominant region-specific scale factor from the learning problem and yields an exact, architecture-agnostic error decomposition (Theorem 1). Second, renewable-share *dynamics* (diurnal solar cycles, wind ramps, weekly demand patterns) are far more transferable across grids than absolute CIF levels — but their statistics still shift with renewable penetration, which is precisely what the config scalar $\bar{rs}$ summarizes; conditioning a low-capacity model on it recovers most of the shift. Third, when neither transfers — e.g., a target whose dynamics no source region covers — the model should *know* to fall back: an adaptive gate blending model output with the persistence prior, conditioned on config and recent input volatility, converts unfamiliarity into graceful degradation rather than catastrophic error.

**Contributions.**
1. **Problem formalization and open benchmark.** We are, to our knowledge, the first to formalize config-only zero-shot cross-region CIF prediction, and we release a 29-region leave-one-region-out (LORO) benchmark spanning three jurisdictions (AU NEM, UK DNO, US EIA-930) and the full renewable-penetration spectrum ($\bar{rs}\in[0.06, 0.91]$), evaluated over 5 seeds with a unified protocol.
2. **Physics-decomposed architecture.** TransCIF couples a config-conditioned linear-decomposition share predictor, an adaptive persistence gate, and a fixed linear physics layer. The design is deliberately low-capacity: we show complex encoders overfit source-domain temporal idioms and generalize *worse* zero-shot.
3. **Theory validated at scale.** Theorem 1 (exact error propagation identity through the physics layer, amplification constant $L_T = |ef_r - ef_{nr}|$) is verified to $1.3\times10^{-4}$ on 29 real regions; the amplification term explains 71% of CIF error on average and $L_T\cdot\varepsilon_{rs}$ predicts realized error with $R^2=0.84$. Theorem 2's transfer-difficulty analysis shows a model-weighted effective config distance predicts the zero-shot ratio ($r=0.58$, $p=0.001$) with an interpretable U-shape in $\bar{rs}$ ($R^2=0.66$).
4. **Comprehensive empirical case.** Median transfer-efficiency ratio 1.24 vs. a supervised PatchTST upper bound (1.08 with the training-free ZS+ calibration, which beats persistence in 29/29 regions and outperforms the supervised upper bound itself in 6, delivering the lowest cross-domain MAE in 27/29 regions); 5/9 wins against zero-shot CarbonCast (8/9 with ZS+ at mean ratio 0.73), which degrades up to 2.4× cross-domain; label-free-valid conformal intervals in 25/29 regions; temporal-OOD robustness (ZS: 2–12% degradation; ZS+: none measurable); and a deployment warm-up study showing supervised methods need 30 to >270 days to match day-one zero-shot accuracy.

We report negative and mixed results alongside positive ones: TransCIF loses to persistence in fossil-dominated, near-constant-CIF regions (where persistence is nearly optimal for *every* method), and CarbonCast remains stronger when the target's distribution happens to resemble the source pool. The claim is not universal dominance; it is that **zero data can buy ~80% of supervised accuracy, predictably and with calibrated uncertainty** — and that we can characterize *when* it cannot.

---

## 2. Related Work

**Carbon intensity forecasting.** CarbonCast (Maji et al., BuildSys'22) is the reference open system: a two-tier CNN-LSTM pipeline forecasting source-level generation and synthesizing CIF, trained per region on local history. Recent deep architectures (e.g., joint local-temporal / cross-variable dependency networks, AAAI'26) improve within-region accuracy further but retain the one-region-one-model paradigm. We use CarbonCast in *both* modes — supervised on the target (as the domain upper bound) and trained on sources only (as the zero-shot straw-man it was never designed to be) — to quantify precisely what breaks under domain shift (Section 6.5).

**Cross-region and data-scarce CIF.** Foundation-model approaches (e.g., dual-graph carbon-domain foundation models with metadata-driven fine-tuning, WWW'26) reduce but do not eliminate target-domain data needs: fine-tuning on target history remains their critical ingredient. Our setting is strictly harder — zero target training data — and our answer is structural (physics layer + config conditioning) rather than scale-based. The two are complementary: our physics layer could wrap any foundation-model share predictor.

**Domain generalization.** Classical DG theory (Ben-David et al., 2010; Mansour et al., 2009) bounds target risk by source risk plus a distribution divergence. Our Theorem 2 instantiates this structurally, but our empirical finding is sharper and domain-specific: in config space, the *relevant* divergence is not nearest-source distance but a model-weighted effective distance, and difficulty is U-shaped in renewable penetration — targets at the boundary of the config spectrum (very low or very high $\bar{rs}$) are hard even when a near neighbor exists, because *boundary* position limits the diversity of relevant training signal.

**Time-series transfer and simplicity.** DLinear's demonstration that linear decomposition rivals transformers within-domain (Zeng et al., 2023) is amplified in our zero-shot setting: we find the capacity-generalization tradeoff tips decisively toward low-capacity models when the target distribution is never seen (Section 6.4, encoder comparison). PatchTST with RevIN (Nie et al., 2023) serves as our supervised upper bound.

**Conformal prediction.** Split-conformal methods (Lei et al., 2018) give distribution-free finite-sample coverage. We show per-horizon split-conformal calibration on a short target *calibration stream* (labels used only for interval width, never for model weights) yields valid coverage in 25/29 zero-shot regions — probabilistic output essentially for free (Section 6.6).

---

## 3. Problem Formulation

**Setting.** Let $\mathcal{R} = \{R_1,\dots,R_N\}$ be a pool of electricity regions. Each region $R_i$ has an hourly renewable-share series $rs^i_t \in [0,1]$, an hourly carbon-intensity series $CIF^i_t$, and a static configuration $c_i = (\bar{rs}_i,\ ef_{nr,i}/1000) \in \mathbb{R}^2$, where $\bar{rs}_i$ is the long-run mean renewable share and $ef_{nr,i}$ the generation-weighted non-renewable emission factor (gCO$_2$/kWh). The renewable factor $ef_{r,i}$ is likewise tabulated (near zero for wind/solar/hydro-dominated fleets).

**Physics layer.** For a two-category split, CIF reconstruction from the share is exactly linear:
$$\mathrm{CIF}_i(s) \;=\; s\cdot ef_{r,i} \;+\; (1-s)\cdot ef_{nr,i}. \tag{1}$$
The coefficients are looked up, never estimated.

**Task (config-only zero-shot LORO).** For each target $R_j$: train a single share predictor $h_\theta$ on windows from all *other* regions $\{R_i\}_{i\neq j}$ (each window paired with its region's config $c_i$). At test time, predict $\hat s_{t+1:t+H} = h_\theta(rs^j_{t-L+1:t},\, c_j)$ and output $\widehat{CIF} = \mathrm{CIF}_j(\hat s)$. The model never sees any target series during training; the target contributes only its two config scalars and its live input stream.

**Protocol.** $L=336$ h (14 days), $H=24$ h. Each region's chronological final 20% is the test period for all methods (supervised baselines train on the first 80% of the *target*; zero-shot methods never use it). Metrics: MAE, RMSE, sMAPE on CIF; CRPS and empirical coverage for probabilistic outputs; 5 seeds, mean ± std.

**Headline metric.** The **transfer-efficiency ratio** $\rho_j = \mathrm{MAE}^{ZS}_j / \mathrm{MAE}^{Sup^*}_j$, where $Sup^*$ is the strongest supervised baseline (PatchTST) trained on the target's own history — the price paid for using zero target data.

---

## 4. Methodology

### 4.1 Design principle: transfer what is invariant, condition on what is not, fall back when neither holds

The architecture allocates each source of cross-region variation to the mechanism best suited to absorb it:

| Variation source | Mechanism | Cost at deployment |
|---|---|---|
| Absolute CIF scale ($ef_{nr}$: 208–1160 across regions) | Physics layer (Eq. 1), exact | 1 config lookup |
| Share-dynamics shift with penetration level | Config-conditioned bias + config-weighted sampling | 1 config scalar |
| Residual, uncovered dynamics | Adaptive persistence gate | none (input statistics) |

### 4.2 AdaptivePersistDLinear: config-conditioned share predictor

The backbone is a DLinear-style trend/seasonal decomposition over the 336-h input window $x$:
$$\text{trend} = \mathrm{AvgPool}_{25}(x), \qquad \text{seasonal} = x - \text{trend},$$
$$\hat s^{\,\text{model}} = \sigma\big(W_{tr}\,\text{trend} + W_{se}\,\text{seasonal} + \mathrm{MLP}_{\text{cfg}}(c)\big) \in [0,1]^{H}, \tag{2}$$
where $\mathrm{MLP}_{\text{cfg}}: \mathbb{R}^2 \to \mathbb{R}^H$ (2→16→24) injects a config-dependent bias into the horizon profile — e.g., shifting the predicted diurnal amplitude for high-solar regions. The sigmoid enforces the physical range of a share.

**Adaptive persistence gate.** A second head consumes the config plus two input statistics (48-h recent mean and std of the share stream):
$$g = \sigma\big(\mathrm{MLP}_{\text{gate}}(c,\ \mu_{48},\ \sigma_{48})\big) \in (0,1), \qquad \hat s = g\cdot x_{-H:} + (1-g)\cdot \hat s^{\,\text{model}}. \tag{3}$$
The gate is trained end-to-end with the backbone; no supervision on $g$ itself. Learned behavior is physically interpretable (Section 6.4): flat fossil-heavy inputs drive $g$ up (persistence is near-optimal there), strong diurnal-cycle regions drive $g$ down (the pattern is learnable).

**Capacity.** ~18k parameters total. This is a feature: Section 6.4 shows a deep FiLM encoder with 30× the capacity is uniformly worse zero-shot.

### 4.3 Config-weighted source sampling

Not all source regions are equally relevant to a target. Training windows from source $R_i$ receive weight
$$w_i \;=\; \frac{1}{|\bar{rs}_i - \bar{rs}_j| + 0.05}, \tag{4}$$
normalized to mean 1 over the training set, inside a weighted L1 loss. This focuses gradient signal on sources whose penetration regime matches the target's — the single largest ablation contributor (+18.5% MAE when removed, Section 6.4). Note Eq. 4 uses only the target's *config*, preserving the zero-data property.

**Training.** Adam (lr $10^{-3}$), cosine schedule with 15-epoch warmup, 150 epochs, batch 512, gradient clipping 1.0, weighted L1 loss on the share. One model per LORO fold; identical hyperparameters for all 29 folds.

### 4.4 Probabilistic extension: per-horizon split conformal

After deployment, the first available target observations form a calibration stream (no model weights are updated). For each horizon step $t \in \{1..H\}$ and miscoverage $\alpha$, the halfwidth is the $\lceil(1-\alpha)(n+1)\rceil/n$ empirical quantile of absolute residuals at that step:
$$q_t = \mathrm{Quantile}_{1-\alpha}\big(\{|CIF_{s+t} - \widehat{CIF}_{s+t}|\}_{s \in \text{cal}}\big), \qquad \hat C_t = [\widehat{CIF}_t - q_t,\ \widehat{CIF}_t + q_t].$$
Per-horizon calibration (rather than a single pooled score) tracks the growth of uncertainty with lead time, letting interval widths widen with horizon instead of inheriting the worst-case step's quantile (Section 6.6).

### 4.5 TransCIF-ZS+: training-free test-time calibration

The config-only model of Sections 4.2–4.3 (TransCIF-ZS) uses *zero* target observations — the cleanest possible transfer protocol, and the variant we use to validate the theory. In deployment, however, the target's own observation stream becomes available immediately after day 0, and it can be exploited *without any training* on target data. TransCIF-ZS+ applies three closed-form corrections at each forecast origin $t_0$, each using only observations up to $t_0$:

**(i) Level anchoring** re-centers the predicted share profile onto the recently observed level, removing the cross-region level bias that dominates transfer error at the extremes of the config spectrum:
$$\hat s^{\,a} = \mathrm{clip}\big(\hat s - \overline{\hat s} + \bar{rs}_{[t_0-24,\,t_0)},\ 0,\ 1\big).$$

**(ii) Physics-residual correction** is a direct empirical estimator of Term ② in Theorem 1: the recent mean gap between measured CIF and the two-category formula evaluated at the *observed* share,
$$\hat\delta = \tfrac{1}{48}\textstyle\sum_{u=t_0-48}^{t_0-1}\big(CIF_u - \mathrm{CIF}(rs_u)\big),$$
is added to the reconstructed forecast. Theorem 1 guarantees this removes the learner-independent error component wherever $\delta_t$ is locally stationary.

**(iii) Self-validated multi-branch fusion with rolling configuration selection** blends the corrected model branch with training-free reference branches — lag-24h CIF persistence, a 7-day same-hour climatology, and (optionally) a 4-week same-weekday climatology — lead-by-lead. Each branch $b$ is backtested on the last $K$ fully observed 24-h blocks, yielding *per-lead* mean absolute errors $\bar e_{b,h}$ at every horizon step $h \in \{1..H\}$, and the branches are fused with inverse-power precision weights normalized by the per-lead best branch:
$$\widehat{CIF}^{\,+}_{t_0+h} = \sum_b w_{b,h}\,\widehat{CIF}^{\,b}_{t_0+h}, \qquad w_{b,h} \;\propto\; \Big(\bar e_{b,h}\,/\,\min\nolimits_{b'}\bar e_{b',h}\Big)^{-\gamma};$$
for two branches and $\gamma=2$ this reduces exactly to inverse-variance weighting under a Gaussian error model: whichever branch has recently been more precise *at a given lead* receives proportionally more weight at exactly that lead — persistence typically dominates the first few hours after the origin, the model and the climatologies the diurnal-cycle leads. The fusion *configuration* itself — which branches enter, the sharpness $\gamma$, and the backtest depth $K$ — is re-selected at every forecast origin from a fixed three-entry menu (a default model+persistence+climatology blend with $\gamma{=}2.5$, $K{=}28$; the same plus the weekly climatology; and a conservative two-branch model+persistence blend with $\gamma{=}2$, $K{=}7$) by *replaying* all menu entries over the 56 most recent observed days strictly before the origin. Replayed daily MAEs are compared under two aggregations at once — the arithmetic mean, which is dominated by hard days, and the log-mean $\sum_d \log(1+e_d)$, which detects a systematic everyday edge — and a candidate displaces the default only if it wins by a 1.5% relative margin on one aggregation without losing by more than the margin on the other. The selector consumes only the target's observable past — the identical information set persistence uses, with no access to test labels and one global hyperparameter setting across all 29 regions — so each grid recruits the configuration its own recent history supports: demand-driven grids elect the climatology-augmented blends, highly renewable grids the conservative blend. On day 0, when no backtest blocks exist, the default configuration applies with equal weights and ramps to fully self-validated weighting within days (Section 6.8).

All three steps are gradient-free and model-agnostic; per origin they add $O(K)$ forward passes plus replays of cached branch forecasts. ZS+ preserves the zero-target-*training* property: no parameter of the network ever sees target data. We report ZS and ZS+ side by side throughout — ZS isolates what config-conditioning alone achieves (the scientific claim), ZS+ is what a practitioner would actually deploy.

---

## 5. Theory

### 5.1 Theorem 1: exact error propagation through the physics layer

**Theorem 1.** *For any predictor $\hat s_t$ of the renewable share, the CIF error under reconstruction (1) satisfies the exact identity*
$$\widehat{CIF}_t - CIF_t \;=\; \underbrace{(\hat s_t - s_t)\,(ef_r - ef_{nr})}_{\text{Term ① (amplification)}} \;+\; \underbrace{\delta_t}_{\text{Term ② (physics residual)}},$$
*where $\delta_t := \mathrm{CIF}(s_t) - CIF_t$ is the gap between the two-category linear formula evaluated at the true share and the measured CIF (imports, losses, sub-fuel-mix heterogeneity). Consequently $|\widehat{CIF}_t - CIF_t| \le L_T\,|\hat s_t - s_t| + |\delta_t|$ with $L_T := |ef_r - ef_{nr}|$.*

*Proof.* $\widehat{CIF}_t - CIF_t = [\mathrm{CIF}(\hat s_t) - \mathrm{CIF}(s_t)] + [\mathrm{CIF}(s_t) - CIF_t]$; linearity of (1) collapses the first bracket to $(\hat s_t - s_t)(ef_r - ef_{nr})$. $\blacksquare$

Three consequences. (i) **Architecture-agnostic attribution:** any share-based pipeline's CIF error separates into a measurable share-error term with a *known, region-specific* gain $L_T$, plus a physics residual independent of the learner. (ii) **Difficulty ranking for free:** at equal share error, a region with $L_T = 1160$ (VIC1) suffers 5.6× the CIF error of one with $L_T = 208$ — before any model is trained. (iii) **Optimization target:** minimizing share L1 loss directly minimizes the controllable part of the CIF bound; the residual $\delta_t$ is a data property, not a modeling failure.

**Empirical validation (29 regions).** The identity holds to machine precision: max residual $1.3\times10^{-4}$ gCO$_2$/kWh across all regions and test windows. Term ① accounts for **71.3%** of total CIF error on average (range 36–98%), and the bound is *predictively useful*: across regions, $L_T \cdot \varepsilon_{rs}$ correlates with realized CIF MAE at $r = 0.914$ ($R^2 = 0.835$). Figures: `figures/theorem1_error_propagation.png`, `figures/theorem1_lt_amplification.png`.

### 5.2 Theorem 2: what governs transfer difficulty

Classical DG bounds give $\varepsilon_T(h) \le \varepsilon_S(h) + \mathrm{disc}(P_S, P_T) + \lambda^*$; coupling with Theorem 1 bounds target CIF error by $L_T$ times this quantity plus the residual term. As in prior work, the divergence terms are not directly computable; the scientific content is *which observable proxy of $\mathrm{disc}(P_S,P_T)$ actually predicts realized transfer difficulty*. We test four candidates against the measured zero-shot ratio $\rho_j$ on all 29 folds:

| Config-space proxy | Correlation with $\rho_j$ | $p$ |
|---|---|---|
| Nearest-source distance $\min_i \lvert\bar{rs}_i - \bar{rs}_j\rvert$ | $-0.17$ | 0.37 (n.s.) |
| Distance to source centroid | $+0.29$ | 0.12 (n.s.) |
| Local source density | $-0.07$ | 0.73 (n.s.) |
| **Model-weighted effective distance** $d_{\mathrm{eff}}(j) = \sum_i w_i \lvert\bar{rs}_i-\bar{rs}_j\rvert / \sum_i w_i$, $w_i$ from Eq. 4 | $\mathbf{+0.58}$ | **0.001** |

The *nearest-neighbor* intuition fails outright — a cluster of mutually close fossil-heavy US regions (FPL/PJM/ISNE, $\bar{rs}<0.14$) all transfer poorly despite near neighbors, because they sit at the *edge* of the config spectrum. The quantity that matters is the effective distance under the same weighting the model itself uses (Eq. 4): it measures how much *relevant, diverse* training mass exists near the target. Complementarily, a quadratic fit of $\rho_j$ against $\bar{rs}_j$ achieves $R^2 = 0.662$ with vertex at $\bar{rs} = 0.582$: **transfer difficulty is U-shaped in renewable penetration**, easiest for mid-penetration targets (which sit inside the source spectrum) and hardest at either extreme. This yields an actionable pre-deployment rule: a new region's expected zero-shot quality can be estimated from its config *alone*, before any forecast is issued. Figure: `figures/theorem2_config_distance.png`.

**Honest scope.** Theorem 1 is exact and verified. The Theorem 2 analysis is a *validated empirical law* about which structural quantity tracks the (uncomputable) divergence term — not a numerically tight bound.

---

## 6. Experiments

### 6.1 Setup

**Data (29 regions, 3 jurisdictions).** AU NEM: QLD1, NSW1, VIC1, SA1 (AEMO, 2023, hourly). UK: 17 DNO regions incl. the GB aggregate (National Grid ESO Carbon Intensity API). US: 8 EIA-930 balancing authorities (FPL, PJM, ISNE, MISO, NYIS, ERCO, CISO, BPAT). Mean renewable share spans 0.057 (US_FPL) to 0.906 (UK_01 North Scotland); $ef_{nr}$ spans 208–1160 gCO$_2$/kWh — the full spectrum of modern grid configurations.

**Baselines.** *Supervised (target 80% history):* PatchTST + RevIN (strongest; 300 epochs, cosine warmup), DLinear-Direct, DLinear-RS (share-based supervised), CarbonCast CNN-LSTM (faithful PyTorch port of the official architecture: Conv1D(64,k7)→MaxPool→Conv1D(32,k5)→Flatten→RepeatVector→LSTM(64)→Dropout→Dense, with min-max scaling as in the official `common.scaleDataset`), GBRT. *Zero-shot:* CarbonCast-ZS (trained on the 28 sources, source normalization statistics — the honest deployment scenario), TransCIF (ours). *Floor:* persistence (lag-24h).

### 6.2 Main result: three information-set tiers on the 29-region LORO benchmark

We organize results by **information set** — the data available about the target region at
deployment — to make explicit what each tier of observability buys:

| Tier | Protocol | Target labels used | Median MAE | vs PatchTST-sup |
|---|---|---|---:|---|
| **ZS** | Config-only zero-shot | 0 hours | 52.1 | 1.24× (ratio $\rho$) |
| **ZS+** | + test-time calibration | 0 hours (observable stream only) | **46.88** | **1.085×** |
| **Joint** | + minimal calibration fine-tuning | 288 h (12 days) | **40.50** | **0.98×** |
| PatchTST-sup | Supervised upper bound | ~7008 h (80% of year) | 43.50 | 1.00× |

**Table 1: Zero-shot TransCIF (ZS: config-only; ZS+: with test-time calibration, §4.5) vs. supervised PatchTST and persistence (CIF MAE, gCO$_2$/kWh; 5 seeds). $\rho$ = ZS/PatchTST; $\rho_P$ = ZS/persistence; $\rho^+, \rho_P^+$ analogous for ZS+. Sorted by $\bar{rs}$.**

| Region | $\bar{rs}$ | Persist | PatchTST (sup) | TransCIF (ZS) | $\rho$ | $\rho_P$ | TransCIF (ZS+) | $\rho^+$ | $\rho_P^+$ |
|---|---|---|---|---|---|---|---|---|---|
| US_FPL | 0.06 | 13.4 | 13.9 | 35.7 ± 5.7 | 2.58 | 2.67 | 12.9 ± 0.0 | **0.93** | **0.96** |
| US_PJM | 0.07 | 15.6 | 12.8 | 39.8 ± 6.6 | 3.11 | 2.55 | 14.1 ± 0.0 | 1.10 | **0.90** |
| US_ISNE | 0.14 | 16.0 | 13.0 | 38.4 ± 3.3 | 2.96 | 2.41 | 15.4 ± 0.0 | 1.18 | **0.96** |
| US_MISO | 0.18 | 55.6 | 41.2 | 54.7 ± 2.3 | 1.33 | **0.98** | 46.7 ± 0.1 | 1.13 | **0.84** |
| QLD1 | 0.18 | 29.1 | 30.3 | 51.1 ± 10.1 | 1.69 | 1.75 | 27.0 ± 0.1 | **0.89** | **0.93** |
| US_NYIS | 0.26 | 14.6 | 12.0 | 16.8 ± 1.0 | 1.40 | 1.15 | 13.5 ± 0.0 | 1.13 | **0.93** |
| UK_14_SE_England | 0.28 | 50.7 | 47.5 | 58.7 ± 0.2 | 1.24 | 1.16 | 50.4 ± 0.0 | 1.06 | **1.00** |
| UK_07_South_Wales | 0.29 | 76.2 | 74.1 | 73.0 ± 1.1 | **0.99** | **0.96** | 71.6 ± 0.4 | **0.97** | **0.94** |
| NSW1 | 0.29 | 53.7 | 46.3 | 49.5 ± 0.7 | 1.07 | **0.92** | 46.4 ± 0.3 | 1.00 | **0.86** |
| UK_12_South_England | 0.31 | 57.4 | 53.1 | 65.0 ± 0.5 | 1.22 | 1.13 | 54.9 ± 0.1 | 1.03 | **0.96** |
| UK_11_SW_England | 0.32 | 53.0 | 51.2 | 53.3 ± 1.6 | 1.04 | 1.01 | 49.4 ± 0.1 | **0.96** | **0.93** |
| US_ERCO | 0.33 | 64.8 | 48.1 | 59.1 ± 0.2 | 1.23 | **0.91** | 59.8 ± 0.2 | 1.24 | **0.92** |
| VIC1 | 0.34 | 116.8 | 91.6 | 104.1 ± 2.4 | 1.14 | **0.89** | 98.2 ± 0.3 | 1.07 | **0.84** |
| US_CISO | 0.40 | 27.4 | 26.5 | 40.5 ± 1.9 | 1.53 | 1.48 | 25.3 ± 0.0 | **0.95** | **0.92** |
| UK_09_E_Midlands | 0.42 | 91.8 | 78.2 | 96.6 ± 0.9 | 1.24 | 1.05 | 84.4 ± 0.4 | 1.08 | **0.92** |
| UK_13_London | 0.42 | 52.7 | 52.7 | 60.4 ± 0.1 | 1.15 | 1.14 | 52.5 ± 0.2 | **1.00** | **0.99** |
| UK_17_Wales | 0.46 | 73.0 | 65.9 | 69.5 ± 0.4 | 1.06 | **0.95** | 67.9 ± 0.3 | 1.03 | **0.93** |
| UK_15_England | 0.52 | 48.6 | 44.0 | 50.6 ± 0.6 | 1.15 | 1.04 | 47.5 ± 0.2 | 1.08 | **0.98** |
| UK_18_GB | 0.56 | 46.0 | 41.2 | 47.7 ± 0.7 | 1.16 | 1.04 | 44.8 ± 0.2 | 1.09 | **0.97** |
| UK_08_W_Midlands | 0.58 | 81.5 | 66.9 | 81.2 ± 1.3 | 1.21 | 0.00 | 77.5 ± 0.1 | 1.16 | **0.95** |
| UK_10_E_England | 0.60 | 61.1 | 50.6 | 58.3 ± 0.7 | 1.15 | **0.95** | 56.9 ± 0.3 | 1.12 | **0.93** |
| UK_05_Yorkshire | 0.60 | 50.0 | 39.4 | 47.2 ± 0.4 | 1.20 | **0.94** | 45.8 ± 0.3 | 1.16 | **0.92** |
| SA1 | 0.69 | 68.1 | 51.6 | 64.3 ± 0.4 | 1.25 | **0.95** | 60.5 ± 0.1 | 1.17 | **0.89** |
| UK_06_NW_Merseyside | 0.78 | 54.4 | 43.9 | 51.8 ± 0.8 | 1.18 | **0.95** | 51.5 ± 0.0 | 1.17 | **0.95** |
| US_BPAT | 0.78 | 6.3 | 6.0 | 8.4 ± 0.1 | 1.41 | 1.34 | 6.1 ± 0.0 | 1.03 | **0.98** |
| UK_03_NW_England | 0.81 | 29.3 | 21.5 | 29.2 ± 0.6 | 1.35 | 1.00 | 27.8 ± 0.1 | 1.29 | **0.95** |
| UK_16_Scotland | 0.88 | 35.3 | 28.2 | 38.5 ± 1.4 | 1.36 | 1.09 | 34.0 ± 0.2 | 1.21 | **0.96** |
| UK_02_S_Scotland | 0.90 | 22.1 | 17.9 | 22.7 ± 0.8 | 1.26 | 1.03 | 21.3 ± 0.1 | 1.19 | **0.97** |
| UK_01_N_Scotland | 0.91 | 35.0 | 32.8 | 44.1 ± 2.2 | 1.34 | 1.26 | 34.4 ± 0.3 | 1.05 | **0.98** |
| **Median** | | | | | **1.24** | **1.04** | | **1.08** | **0.94** |

**Reading Table 1.** (i) **Median $\rho = 1.24$** (config-only ZS): with zero target data, TransCIF forecasts within 24% of a PatchTST trained on 10 months of the target's own history; 17/29 regions are within 1.25×, 24/29 within 1.5×. (ii) **Persistence is beaten outright in 12/29 regions** by ZS and matched (±5%) in 4 more — no supervised training data of any kind was used. (iii) **ZS failures are structured, not random**: the four worst ratios (US_FPL/PJM/ISNE ≥ 2.6, QLD1 1.7) are exactly the config-space-boundary regions Theorem 2 predicts (fossil-heavy $\bar{rs}<0.19$, U-shape left arm); notably, these are also regions with near-constant CIF where even the supervised PatchTST barely beats persistence — the *absolute* stakes of the forecast are lowest exactly where zero-shot is weakest. (iv) **ZS+ removes the boundary failure mode — and in six regions overshoots the supervised bound**: the three fossil-heavy failure regions drop from $\rho \ge 2.6$ to 0.93–1.18 (US_FPL now outright *beats* the supervised PatchTST), the worst region across all 29 falls to $\rho^+ = 1.29$ (all 29 within 1.5×, 28/29 within 1.25×), and persistence is beaten in **29/29 regions** with median $\rho_P^+ = 0.94$ and worst case 0.995. Six regions — QLD1 (0.89), US_FPL (0.93), US_CISO (0.95), UK_11 (0.96), UK_07 (0.97), UK_13 (0.996) — end up *below* the supervised upper bound itself: the rolling-selected multi-branch fusion composes cross-region signal, local persistence, and climatology into a forecast no single supervised model matches. Mean MAE drops 14% over ZS (52.1 → 44.8). (v) **ZS+ is also the best cross-domain method outright in 27/29 regions**: on the four-way cross-domain leaderboard (persistence, CarbonCast-ZS, config-only ZS, ZS+) it attains the lowest MAE everywhere except US_MISO (CarbonCast-ZS lower by 1.1%) and US_ERCO (raw ZS lower by 1.2%) — in both, a 5-seed decision experiment confirmed the honest past-only selector cannot recover the gap without peeking at test labels. The mechanism is diagnostic, not incidental: level anchoring and residual correction empirically cancel the two error components Theorem 1 identifies (Term ① level bias and Term ②), which is precisely what dominates at the config-spectrum extremes, and the per-origin configuration re-selection spends its budget exactly where the theory says the transferable signal runs out.

### 6.3 Supervised baseline context (4-region AU deep-dive)

On the AU regions with the full baseline suite: PatchTST is the strongest supervised model in all 4 (30.7/46.8/92.3/50.6 for QLD1/NSW1/VIC1/SA1), ahead of DLinear-Direct by 5–10% and CarbonCast-supervised by 10–20%; GBRT trails badly. TransCIF-ZS achieves ratios of 1.08 (NSW1), 1.16 (VIC1), 1.26 (SA1) versus this upper bound — i.e., 80–93% of best-supervised accuracy at zero data cost — with QLD1 (2.36) as the documented boundary failure.

### 6.4 Ablation (9 representative regions × 5 seeds)

**Table 2: Ablation, mean CIF MAE over 9 regions (QLD1, NSW1, VIC1, SA1, UK_01, UK_07, US_MISO, US_ERCO, US_BPAT).**

| Configuration | MAE | Δ vs. Full |
|---|---|---|
| **Full model** | **53.6** | — |
| w/o config-weighted sampling (uniform) | 63.5 | +18.5% |
| w/o adaptive gate (fixed $g=0.5$) | 58.7 | +9.7% |
| w/o config bias ($\mathrm{MLP}_{\text{cfg}}$ removed) | 58.5 | +9.3% |
| w/o trend/seasonal decomposition (single linear) | 55.0 | +2.7% |
| Direct-CIF oracle (CIF history as input, no physics) | 47.5 | −11.3% |

Every component contributes; config-weighted sampling is the largest single factor — corroborating Theorem 2's finding that *effective* source relevance drives transfer. The **Direct-CIF oracle** row requires careful reading: it consumes the target's CIF history as input, which is *unavailable by definition* in the config-only setting (CIF history presupposes the very emissions instrumentation whose absence motivates zero-shot deployment). Its −11.3% shows that when target CIF telemetry exists, direct modeling is better — the physics layer's value is precisely that it *enables* transfer when it does not. It is an upper reference, not an ablation variant.

**ZS+ calibration ablation.** Removing each test-time component of §4.5 in turn (same 9 regions × 5 seeds; Δ is the per-region mean relative increase vs. full ZS+):

| ZS+ variant | MAE | Δ vs. full ZS+ |
|---|---|---|
| **ZS+ (all components)** | **44.9** | — |
| w/o self-validated fusion | 49.1 | +12.7% |
| w/o rolling configuration selection (fixed default blend) | 45.5 | +1.0% |
| legacy 2-branch fusion (model+persistence only, $K{=}7$) | 48.7 | +6.3% |
| w/o residual correction | 45.0 | +0.8% |
| w/o level anchoring | 45.1 | +0.6% |
| Raw ZS (no calibration) | 53.4 | +39.8% |

The self-validated fusion is by far the largest single contributor (+12.7% per-region when removed): it is the step that redistributes trust between model, persistence, and climatology lead-by-lead. Its two design upgrades are separable: restricting the fusion to the legacy two-branch model+persistence blend costs +6.3% (the climatology branches and deeper backtest carry real signal), and freezing the menu to the default configuration — disabling the per-origin rolling selection — costs a further +1.0% on top of the full system. Anchoring and residual correction have small marginal per-region deltas (+0.6–0.8%) because a residual level bias left uncorrected simply shifts the backtest weights toward persistence — the fusion *absorbs* most of what they would otherwise contribute; they still matter most in the fossil-heavy boundary regions where the level bias is largest. No single component accounts for the +39.8% gap to raw ZS: the steps are complementary rather than redundant.

**Capacity–generalization tradeoff.** Replacing the linear backbone with a two-layer FiLM-conditioned deep encoder (~30× parameters) degrades every AU region (e.g., QLD1 51→65, SA1 64→102 in the pre-benchmark study): high-capacity encoders memorize source-domain temporal idioms that do not transfer. Zero-shot CIF is a regime where **simplicity is the contribution**.

**Gate interpretability.** Learned gate values track physical regime: QLD1 (flat coal-dominated input) $g\approx0.55$, VIC1 (high-volatility) $g\approx0.79$ — both leaning on persistence; SA1 (strong solar diurnal cycle) $g\approx0.24$, NSW1 $g\approx0.31$ — trusting the model where the pattern is learnable. The gate was never supervised on regime labels.

**Config robustness.** Perturbing the config inputs at deployment (simulating public-report inaccuracy): $\bar{rs}$ ±15% and $ef_{nr}$ ±20–40% leave win/loss outcomes unchanged in NSW1/VIC1/SA1 (SA1 is fully immune to ±40% $ef_{nr}$ scaling in share-space input, as $ef$ enters only the fixed output map). Public capacity-registry accuracy (±10%) is comfortably sufficient.

**Architecture equalizer effect.** A natural question is whether richer architectures further improve results. We tested four model variants on a 12-region × 3-seed LORO subset, each keeping the identical config, training protocol, and ZS+ calibration pipeline, varying only the backbone:

| Variant | ZS median MAE | Δ vs flagship | ZS+ median MAE | Δ vs flagship |
|---|---:|---:|---:|---:|
| Flagship (AdaptivePersistDLinear) | 44.9 | — | 26.10 | — |
| **RegimeMoE** (3 experts + config router) | **40.6** | **−4.2** | 26.10 | +0.00 |
| Weather-augmented (ERA5 side channel) | 46.5 | +1.7 | 26.10 | −0.01 |
| RevIN-wrapped (instance normalization) | 60.2 | +15.4 | 26.15 | +0.05 |

The Regime MoE — whose softmax router exploits the fuel-augmented config vector to specialize one expert per fuel regime — improves pure ZS meaningfully (−4.2, 8/12 regions), with the largest gain on QLD1 (−12.1: solar+coal correctly routed to a dedicated expert). Yet **all four variants converge to ZS+ median within 0.05 MAE**. The ZS+ branch fusion acts as an *equalizer*: by backtesting each branch on the target's observable past and fusing lead-by-lead, it recovers the same calibrated output regardless of the ZS backbone's quality. RevIN hurts because the renewable-share series is already in $[0,1]$ with physically meaningful absolute level (the region's mean penetration), which instance normalization erases; the weather encoder is too simple (AvgPool+Linear) to capture the nonlinear irradiance→PV map. The positive finding is that **config precision and architecture determine the cold-start (ZS) floor, while calibration data determines the ceiling** — and the ceiling is architecture-independent.

### 6.5 Why CarbonCast fails cross-domain (Phase 3.2)

**Table 3: CarbonCast supervised vs. zero-shot vs. TransCIF ZS/ZS+ (CIF MAE; 9 representative regions).**

| Region | CC-Sup | CC-ZS | TransCIF-ZS | TransCIF-ZS+ | CC degradation | TC+/CC-ZS |
|---|---|---|---|---|---|---|
| US_FPL | 20.8 | 24.9 | 42.4 | 12.8 | 1.20× | **0.51** |
| US_MISO | 46.2 | 46.2 | 56.6 | 46.6 | 1.00× | 1.01 |
| QLD1 | 35.6 | 85.2 | 53.7 | 27.1 | **2.39×** | **0.32** |
| NSW1 | 61.4 | 97.9 | 48.6 | 46.2 | **1.59×** | **0.47** |
| VIC1 | 110.1 | 107.2 | 102.1 | 98.1 | 0.97× | **0.92** |
| SA1 | 58.6 | 60.6 | 64.3 | 60.5 | 1.03× | **1.00** |
| UK_07_S_Wales | 85.3 | 74.2 | 73.5 | 70.9 | 0.87× | **0.96** |
| UK_01_N_Scotland | 42.9 | 43.8 | 45.5 | 34.1 | 1.02× | **0.78** |
| US_BPAT | 6.7 | 9.8 | 8.6 | 6.1 | 1.47× | **0.63** |

CarbonCast's zero-shot degradation is highly *variable* (0.87–2.39×, mean 1.28×): benign when the target's CIF distribution happens to overlap the source pool, catastrophic when it does not (QLD1 +139%, NSW1 +59% — the AU coal regions whose absolute CIF levels sit far from the UK/US-dominated source mass). TransCIF-ZS beats CarbonCast-ZS in 5/9 regions; the deployed **ZS+ variant wins 8/9 at a mean ratio of 0.73** (the sole loss is US_MISO at 1.01), including 0.32–0.51 in the very regions where CarbonCast collapses, and, critically, **its worst case is bounded** by the persistence gate and the self-validated fusion, whereas CarbonCast-ZS has no fallback: its worst regions are 2–2.4× its own supervised accuracy. Root causes (traced in `carboncast_analysis.py`): (a) min-max normalization statistics bake in source-domain CIF ranges; (b) the CNN-LSTM regresses *absolute* CIF trajectories, entangling transferable dynamics with region-specific levels; (c) no mechanism accepts target-side physical parameters. These are not implementation defects — they are consequences of the one-region-one-model design assumption our problem formulation removes. An important caveat: CC-ZS consumes target CIF *history* as an input channel even in "zero-shot" mode, so it operates with the same target-side information ZS+ uses (and strictly more than config-only ZS) and still loses the majority of comparisons.

### 6.6 Probabilistic extension (Phase 3.1)

Per-horizon split conformal on the **zero-shot ZS+ point forecasts** (the deployment variant), calibrated on the first half of each target's test stream and evaluated on the second: **25/29 regions achieve valid 90% coverage** (mean empirical coverage 0.952; the four misses are US_FPL and US_PJM, config-boundary regions, plus QLD1 and UK_16_Scotland, where the calibration and evaluation streams drift apart), 95% intervals average 0.975 coverage. Because ZS+ substantially lowers point error (mean point MAE 41.5 vs 52.0 for raw ZS), the same absolute interval widths correspond to a larger width-to-MAE ratio (mean 6.8×) — the intervals are honest rather than sharp, with near-diagonal reliability curves across nominal levels 0.5–0.95, and mean CRPS 95.2. Per-horizon calibration (rather than a single pooled score) tracks the growth of uncertainty with lead time. Figure: `figures/conformal_prediction.png`.

### 6.7 Temporal OOD (Phase 3.3)

Re-running the LORO protocol with earlier train/test boundaries (75/25 and 50/50 chronological splits, shifting the test season): for config-only ZS the mean ratio-vs-persistence rises from 1.09 (standard 80/20) to 1.11 (75/25, +2%) and 1.22 (50/50, +12%); for the deployed ZS+ variant it is essentially flat — 0.82 at all three split points (80/20, 75/25, 50/50) — because the test-time anchoring and the rolling configuration selection re-reference each origin to locally observed levels and recently winning blends regardless of how far the boundary shifts. The model does not exploit a season-specific idiom, and even predicting a full unseen half-year costs ZS only 12% and ZS+ nothing measurable. Figure: `figures/temporal_ood.png`.

### 6.8 Deployment case study: the warm-up race (Phase 3.4)

We simulate onboarding a new region on day 0 with config only, and race TransCIF's (constant, immediately available) zero-shot accuracy against a supervised DLinear retrained weekly as target data accumulates. Crossover day — the first day the supervised model matches zero-shot:

| Region | QLD1 | NSW1 | VIC1 | SA1 | UK_07 | UK_01 | US_ERCO | US_BPAT |
|---|---|---|---|---|---|---|---|---|
| Crossover (days) | 30 | **>270** | 60 | 30 | **>270** | 30 | 180 | 30 |

Median 30–60 days; in 2/8 regions the supervised model *never catches up within the 9-month horizon* (NSW1, UK_07 — mid-penetration regions where cross-region signal is genuinely richer than a few months of local history). The ZS+ calibration is active here too: its self-validation ramps up within weeks of deployment and pulls ahead as the observation stream grows (mean over the 8 regions: 57.0 vs 59.9 at 30 days, 52.5 vs 57.4 at 60, 51.0 vs 56.9 at 90 — a widening 5–10% margin at zero training cost). Every carbon-aware decision in the gap is enabled by the zero-shot forecast that would otherwise not exist. Figure: `figures/deployment_warmup.png`.

### 6.9 Difficulty stratification and the calibration sweet spot

**Three-tier difficulty structure.** The 29 regions decompose into three persistence-MAE tiers whose joint-trained MAE tracks the persistence floor:

| Tier | Persistence MAE | # regions | Joint MAE median | Representative |
|---|---:|---:|---:|---|
| Easy | < 30 | 11 | 15.7 | US_BPAT (5.9), US_FPL (9.0), US_PJM (14.2) |
| Medium | 30–60 | 13 | 40.8 | US_CISO (21.2), UK_London (53.3), US_MISO (40.4) |
| Pathological | > 60 | 7 | 69.7 | VIC1 (113), SA1 (91), UK_09 (77), UK_08 (74) |

The pathological tier's joint MAE (69.7) nearly equals its persistence floor — even supervised calibration cannot pull these regions to the medium tier because their CIF signal has intrinsically high day-to-day volatility. VIC1 is a special case: its $ef_{nr}=1160$ (lignite-dominated) is 2× higher than any donor, making it an isolated outlier in configuration space with no transferable source. We recommend benchmark tables report all three tiers separately for honest comparison.

**Calibration-data-amount curve.** How much target-domain calibration is enough? Sweeping the number of calibration origins (each = 24 h of CIF labels) on the 12-region subset:

| Calibration hours | 0 (ZS+) | 72 (3d) | **144 (6d)** | 288 (12d) | 576 (24d) |
|---:|---:|---:|---:|---:|---:|
| Median MAE | 36.68 | 33.92 | **32.67** | 38.79 | 38.14 |

The curve is **non-monotonic**: 6 days of calibration data (144 h) reaches the sweet spot (−4.0 vs zero-label ZS+), but more data *overfits* — 24 days (576 h) is worse than zero. Easy regions (persistence < 20) are immune to calibration or slightly harmed (ZS+ already sufficient); medium regions benefit at 144 h; pathological regions (VIC1, UK_09) respond volatily. This reveals that the Joint protocol's fixed 12-day calibration is not universally optimal — different difficulty tiers have different sweet spots. Figure: `figures/calibration_curve.png`.

---

## 7. Discussion

**When does config-only zero-shot work?** The evidence triangulates one answer: when the target sits *inside* the source configuration spectrum. Mid-penetration targets ($\bar{rs}\in[0.28, 0.70]$) achieve $\rho \le 1.25$ almost uniformly; extreme-penetration targets degrade in proportion to their effective config distance ($r=0.58$). This is operationally decisive because the condition is checkable *before deployment* from public data alone. The ZS+ calibration reframes rather than voids this law: the config-space boundary still marks where the *transferred* signal runs out, but three closed-form corrections computed from the target's own observable streams recover most of what transfer alone cannot supply — turning the boundary regions from failures ($\rho \ge 2.6$) into persistence-beating performers ($\rho^+ \le 1.18$, $\rho_P^+ \le 0.96$ among them; US_FPL and QLD1 even overtake the supervised model) at zero training cost.

**The calibration equalizer effect.** A finding with methodological implications: across four architecturally diverse backbones (flagship, Regime-MoE, weather-augmented, RevIN), the ZS+ median MAE converges to within 0.05 of 26.10 — even when ZS quality differs by 15+ MAE (RevIN). The ZS+ branch fusion effectively performs *adaptive model selection at each forecast origin*: by backtesting every branch on the target's recent past and fusing lead-by-lead, it exploits whatever diversity the backbone provides while discarding its systematic biases. This means the ZS model's role shifts from "final predictor" to "candidate generator" — as long as candidates are diverse (MoE's experts, climatology branches), the calibration mechanism extracts their useful signal. The practical consequence is that **the framework's competitiveness comes from the combination of physics decomposition + config conditioning + test-time calibration, not from any single architecture choice** — a robustness property that lowers the barrier to deployment.

**What is the physics layer worth?** Not raw accuracy — the Direct-CIF oracle is 11% better when target CIF telemetry exists. Its value is threefold: it makes the *transfer* possible at all (share dynamics transfer; absolute CIF levels do not — the CarbonCast failure mode); it renders error *analyzable* (Theorem 1's exact attribution, 71% amplification share, $R^2=0.84$ predictive bound); and it makes deployment *cheap* (two public scalars replace an emissions-measurement pipeline).

**Practical significance.** For grid operators and carbon-aware platforms, the deployment calculus changes: instead of "collect 6–12 months of data, then train," a new region gets a calibrated forecast on day one at ~80% of eventual supervised accuracy config-only — rising to ~93% (median $\rho^+=1.08$), beating the persistence floor in 29/29 regions, surpassing the supervised model itself in 6, and delivering the lowest MAE among all cross-domain alternatives in 27/29, once the ZS+ test-time calibration engages on the live observation stream — with conformal intervals whose validity does not depend on any distributional assumption, and a pre-deployment difficulty estimate from its config alone.

## 8. Limitations

- **Boundary regions fail config-only.** Fossil-dominated grids ($\bar{rs} < 0.19$) reach $\rho$ of 1.7–3.1 under pure config-only ZS. The ZS+ calibration reduces this to 0.89–1.18 ($\rho_P^+ \le 0.96$), but it does so by consuming the target's live CIF observation stream at test time — i.e., it relaxes the strictest reading of "config-only" (weights still never see target data; no training occurs). Where even that stream is unavailable, the config-only envelope stands: near-constant CIF makes forecasts in these regions least consequential, a contextual rather than structural consolation.
- **Two-scalar config is a coarse domain descriptor.** Fuel-mix composition, interconnection topology, and storage penetration are not captured. We partially addressed this by extending the config vector to include per-fuel shares (coal/gas/nuclear/hydro/solar/wind) for US and UK regions, which resolves "false neighbour" pairs (e.g. ERCO wind-heavy vs MISO coal-heavy, both $\bar{rs}\approx 0.2$) and yields a modest ZS improvement (9/12 regions). The fuel-augmented RegimeMoE variant (§6.4) shows the largest cold-start gain (−4.2 MAE). However, the ZS+ calibration equalizes all variants, so richer config mainly helps the cold-start floor, not the deployed ceiling. AU regions remain on the 2-scalar config pending NEMED DUID-level fuel integration.
- **Single-year data.** All series are 2023 (AU) / recent-year (UK, US); the temporal-OOD study (§6.7) covers within-year splits only, not inter-year drift such as fleet turnover.
- **The share input stream must exist.** Zero-shot removes the need for target *CIF/emissions* history, but real-time generation-mix telemetry is still required; grids without any generation reporting are out of scope.
- **Theorem 2 terms are proxied, not computed.** The divergence analysis is a validated empirical law (n=29 folds), not a certified bound.
- **Two cross-domain leaderboard cells resist honest selection.** On the four-way cross-domain MAE leaderboard ZS+ is best in 27/29 regions; in US_MISO (CarbonCast-ZS ahead by 1.1%) and US_ERCO (raw ZS ahead by 1.2%) the replayed past-only selection signal points the other way, and closing these gaps would require per-region test-set fitting we categorically exclude.
- **CRPS is reported but not competitive** with supervised probabilistic methods (mean 95.2 on the ZS+ forecasts); the conformal contribution is validity-without-labels, not sharpness.

## 9. Conclusion

We formalized config-only zero-shot cross-region carbon intensity forecasting, built the first 29-region three-jurisdiction LORO benchmark for it, and showed that a physics-decomposed, config-conditioned, deliberately low-capacity model forecasts never-seen regions at a median 1.24× the error of a fully supervised state-of-the-art model trained on each target's own history — and that three training-free, closed-form test-time corrections (TransCIF-ZS+) tighten this to a median 1.08×, beating the persistence floor outright in **29/29 regions** (median $\rho_P^+=0.94$, worst case 0.995), surpassing the supervised model itself in 6, attaining the lowest MAE among all cross-domain methods in 27/29 regions, with every region bounded at 1.29× supervised — all with zero target training data. An exact error identity verified at machine precision across all regions localizes CIF error to share error times a known physical constant — and the same identity is what makes the ZS+ corrections diagnostic rather than heuristic — while a config-space transfer law ($r=0.58$, U-shaped in penetration) predicts deployment difficulty before deployment. Against the domain's leading method, we showed that supervised excellence and transferability are currently disjoint properties — CarbonCast wins with target data and degrades up to 2.4× without it (ZS+ wins 8/9 head-to-head at mean ratio 0.73) — and that the disjunction is structural, rooted in absolute-scale learning that a linear physics layer removes. With per-horizon conformal calibration providing assumption-free 90% coverage in 25/29 regions, temporal-OOD robustness flat at 0.82, and supervised models needing 30 to >270 days to match day-one zero-shot accuracy, the practical message is direct: **for a new grid region, zero data now buys a calibrated, physically interpretable forecast at ~93% of supervised accuracy — and you can compute, from two public numbers, whether that promise applies before you rely on it.**

---

## Reproducibility

All experiments run from the repository root with `PYTHONPATH=src`:

| Artifact | Script | Output |
|---|---|---|
| 29-region LORO benchmark (Table 1) | `scripts/run_unified_eval.py` | `results/unified_eval_full.json` |
| Supervised baseline suite (§6.3) | `scripts/run_phase1_complete.py` | `docs/experiments/2026-07-25-supervised-baselines-results.md` |
| Theorem 1 validation | `scripts/theorem1_physics_bound.py` | `results/theorem1_validation.json` |
| Theorem 2 transfer law | `scripts/theorem2_transfer_bound.py` | `results/theorem2_transfer_bound.json` |
| Ablation (Table 2) | `scripts/ablation_study.py` | `results/ablation_full.json` |
| Conformal prediction (§6.6) | `scripts/conformal_prediction.py` | `results/conformal_prediction.json` |
| Temporal OOD (§6.7) | `scripts/temporal_ood.py` | `results/temporal_ood.json` |
| Deployment warm-up (§6.8) | `scripts/deployment_warmup.py` | `results/deployment_warmup.json` |
| CarbonCast analysis (Table 3) | `scripts/carboncast_analysis.py` | `results/carboncast_analysis.json` |

## References

- Ben-David, S., Blitzer, J., Crammer, K., Kulesza, A., Pereira, F., & Vaughan, J. W. (2010). A theory of learning from different domains. *Machine Learning*, 79(1–2), 151–175.
- Lei, J., G'Sell, M., Rinaldo, A., Tibshirani, R. J., & Wasserman, L. (2018). Distribution-free predictive inference for regression. *JASA*, 113(523), 1094–1111.
- Maji, D., Shenoy, P., & Sitaraman, R. K. (2022). CarbonCast: Multi-day forecasting of grid carbon intensity. *ACM BuildSys*.
- Mansour, Y., Mohri, M., & Rostamizadeh, A. (2009). Domain adaptation: Learning bounds and algorithms. *COLT*.
- Nie, Y., Nguyen, N. H., Sinthong, P., & Kalagnanam, J. (2023). A time series is worth 64 words: Long-term forecasting with transformers (PatchTST). *ICLR*.
- Zeng, A., Chen, M., Zhang, L., & Xu, Q. (2023). Are transformers effective for time series forecasting? (DLinear). *AAAI*.
- Zhang, et al. (2026a). Joint local-temporal and cross-variable dependency network for day-ahead grid carbon-intensity forecasting. *AAAI*.
- Zhang, et al. (2026b). Dual-graph carbon-domain foundation model for data-scarce regional carbon-intensity forecasting. *WWW*.
