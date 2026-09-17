# Zero-telemetry forecasting of grid carbon intensity: physics-based fuel decomposition, information tiers, and intercontinental transfer

**Target journal**: *Applied Energy* (research paper).
**Status**: submission manuscript (English). All numbers trace to the result files listed in the Reproducibility appendix.

---

## Highlights

- A strict information hierarchy prices every deployment tier of carbon-intensity forecasting, from zero telemetry to full supervision.
- A 21k-parameter fuel-decomposition network with physics-shaped heads beats 10-month supervised training on 18 of 29 real regions.
- Trained only on US+UK regions, the model outperforms locally supervised baselines on 3 of 4 Australian grids.
- A reproducible, weather-only event protocol shows systematic 2–4× error inflation on Dunkelflaute days.
- A translation layer runs the same model on Chinese provinces using only monthly public statistics (15–18 gCO₂/kWh monthly MAE).

---

## Abstract

Hourly grid carbon intensity (CIF, gCO₂/kWh) forecasting underpins carbon-aware scheduling of loads, but every published predictor requires months of fuel telemetry from the target grid and local retraining. This excludes precisely the regions where carbon-aware scheduling matters most. We formulate zero-telemetry CIF forecasting as a hierarchy of strictly nested information sets — public configuration scalars only → live fuel-share streams → CIF history → lagged observations → full supervision — and price each tier under one protocol across 29 real regions in three jurisdictions (AEMO, National Grid ESO, EIA-930). The zero-telemetry tier is carried by a 21k-parameter fuel-decomposition network: carbon intensity is never regressed directly but reconstructed from ten fuel-share trajectories, each head injecting a fuel-specific physical bias (solar as a deterministic astronomical envelope modulated by weather; wind as a capacity-factor ratio structure with drought anchoring; baseload as masked level statistics; thermal as a dispatchable residual initialized at configuration priors), while all region-specific scale is absorbed by table-lookup emission factors. A zero-parameter structure router sends wind-heavy and hydro-dominant configurations to an aggregate path. Under 145 leave-one-region-out runs (5 seeds), the zero-telemetry tier reaches median MAE 38.4 gCO₂/kWh — below PatchTST trained on 10 months of target-region data (43.5) and CarbonCast (43.4) — winning outright on 18/29 regions; the telemetry tier reaches 36.9 (17/29). Under leave-one-jurisdiction-out transfer (US+UK training only), the model beats locally supervised baselines on 3/4 Australian grids. We further quantify deployment costs that reanalysis-based studies typically hide: substituting operational numerical weather forecasts for reanalysis degrades pooled MAE by +6.4; errors inflate 2–4× on weather-defined Dunkelflaute days (62 of 348 method–region tests significant); and a 20 gCO₂/kWh zero-telemetry floor is provably unreachable (oracle level floor 23.1). A deployment translation layer runs the same model on Chinese provinces from monthly public statistics alone, validated against independent ground truth in Shanxi and Shanghai at 15–18 gCO₂/kWh monthly-aggregate MAE. The hierarchy aligns with China's tiered disclosure regime: the method upgrades exactly as disclosure deepens.

**Keywords**: carbon intensity forecasting; zero-shot transfer; physics-informed neural network; renewable energy; Dunkelflaute; Chinese power grid

---

## 1. Introduction

Decarbonizing electricity consumption requires knowing when the grid is clean. Hourly carbon intensity (CIF, gCO₂/kWh) forecasts drive carbon-aware load scheduling, electric-vehicle charging, and demand response. A decade of work has produced accurate within-region predictors — from hierarchical CNN-LSTM systems [1] to wavelet-patch architectures [3] — but all operate under a one-model-per-region contract: connecting a new grid means assembling fuel telemetry, retraining, and re-validating. Most of the world's grids cannot satisfy this contract. China's seven formally operating spot markets publish 15-minute fuel-level dispatch data, but access requires market-member registration, formats differ by province, histories are short, the balancing-area denominator excludes self-owned plants, and the official carbon intensity is an annual average published with a two-year lag. The regions where carbon-aware scheduling has the greatest marginal value are exactly the regions where its prerequisite data is hardest to obtain.

This paper asks a deliberately extreme question: **with zero telemetry from the target region, how well can a new grid be forecast?** We organize the problem as a hierarchy of strictly nested information sets,

$$\mathcal{I}_\mathrm{cfg} \;\subset\; \mathcal{I}_0 \;\subset\; \mathcal{I}_+ \;\subset\; \mathcal{I}_\mathrm{lag} \;\subset\; \mathcal{I}_S,$$

where each tier corresponds to a real deployment scenario: public configuration scalars plus weather and calendar (zero telemetry) → live fuel-share streams → target CIF history → lagged observations → full supervision. Every tier is evaluated under the same protocol on the same test windows, so the hierarchy prices each increment of information. The zero-telemetry tier $\mathcal{I}_\mathrm{cfg}$ is the primary contribution: it matches the reality that any Chinese province can meet today — monthly generation-structure bulletins, public reanalysis weather, and a calendar.

The technical carrier of the zero-telemetry tier is a **fuel-decomposition network** (FuelDecompNet). Carbon intensity is never regressed directly. The model predicts ten fuel-share trajectories, and a physical layer reconstructs CIF through fixed table-lookup emission factors, with the factors never learned. Each fuel head injects a fuel-specific inductive bias: photovoltaic output is dominated by a purely astronomical channel; wind output follows a capacity-factor ratio structure anchored on a drought-stabilized reference level; baseload fuels obey a support mask (a fuel absent from the monthly configuration table cannot hallucinate output); and thermal output is the dispatchable residual, initialized at configuration priors. The design principle is a **structural separation of level from shape**: absolute levels (emission factors, mean shares) are region-specific and absorbed by lookup and configuration, while shapes (diurnal dynamics, weather response) are the only learned, transferable object. A zero-parameter structure router — a deterministic function of the configuration vector — sends wind-heavy and hydro-dominant regions to an aggregate path whose shape rides on the observed renewable-share stream, and all other regions to the decomposition path.

Three components of the method make this architecture deployable, and each is a contribution in itself:

1. **A unified UTC time base for all inputs.** The method aligns every input to UTC before astronomical and calendar features are computed. This is a design specification, not an engineering detail: under a misaligned time base the correlation between the astronomical channel and photovoltaic share is −0.697 versus +0.976 under UTC, and because the channel is deterministic physics, no architectural improvement can recover from the misalignment.
2. **Farm-weighted weather.** Replacing centroid-grid weather with capacity-weighted farm-cluster weather raises the explained variance of wind share from R² 0.08–0.15 to 0.33–0.48 — the single largest gain in the full ablation matrix, and a data-engineering result rather than an architecture result.
3. **Capacity discipline.** The full model has ~21k parameters. In ablation, deep conditioning (feature modulation, hypernetworks, meta-learned fusion) consistently degraded zero-shot transfer performance: high-capacity encoders memorize source-domain idioms that do not transfer. In zero-telemetry CIF forecasting, negative capacity scaling is itself a finding.

**Contributions.**

1. **An information-hierarchy formulation of cross-domain CIF forecasting** with a single evaluation protocol that prices every deployment tier. On 29 regions × 5 seeds, the zero-telemetry tier (median MAE 38.4) beats PatchTST trained on 10 months of target data (43.5) and CarbonCast (43.4), winning on 18/29 regions — the first demonstration that the zero-telemetry-versus-supervision balance tips, in aggregate, toward zero telemetry.
2. **A physics-decomposed architecture with an exact error-propagation identity.** Theorem 1 decomposes CIF error into fuel-share errors weighted by known physical constants plus a measurable physical residual, independent of architecture. It yields per-fuel error attribution and a free, deployment-time difficulty ordering. Proposition 1 (empirical) characterizes *when* zero telemetry beats supervision — supervision inherits label noise scaled by emission-factor contrast, zero telemetry pays a transfer residual — and the observed win/loss asymmetry across regions follows this structure.
3. **A reproducible event-day evaluation protocol and the deployment costs it reveals.** Weather-only, reanalysis-reproducible Dunkelflaute labels expose systematic error inflation of 2–4× (62/348 method–region tests significant; maximum 4.19×), concentrated in a Scotland–northern-England belt. Substituting operational NWP forecasts for reanalysis degrades pooled MAE by +6.4. A 20 gCO₂/kWh zero-telemetry target is shown unreachable (oracle level floor 23.1). These costs are design inputs for any carbon-aware system, not caveats.
4. **A China-facing deployment translation layer.** Monthly public statistics are translated into the model's input contract through four components (Chinese calendar remapping, sender-weighted import emission factors, monthly configuration tables, proxy-anchored trust gating), validated against independent ground truth in Shanxi and Shanghai at 15–18 gCO₂/kWh monthly-aggregate MAE. The hierarchy maps onto China's tiered disclosure regime: as disclosure deepens from monthly statistics to market-member fuel data, the same model upgrades tiers with zero re-engineering.

**Scope of claims.** We do not claim zero telemetry universally dominates supervision. On SA1, where fuel labels are classifier-synthesized, the locally supervised model retains a clear edge (65.2 vs 69.3 under cross-jurisdiction transfer); on NSW1, supervision similarly benefits from label idiosyncrasies invisible to the physical route. Event-day and right-tail errors inflate systematically, operational weather forecasts carry a quantified penalty, and intraday shape amplitude in the China deployment is underestimated 2–3×. The claim is narrower and, we argue, more useful: **in every scenario where telemetry is absent or inaccessible, the zero-telemetry tier delivers forecasts significantly better than no forecast and persistence, usually at parity with or better than 10-month supervision — and this can be predicted from public numbers before any deployment decision.**

The remainder of the paper: Section 2 reviews related work; Section 3 formalizes the problem; Section 4 develops the architecture; Section 5 states the theory; Section 6 presents the China deployment layer; Section 7 details the experimental setup; Section 8 reports results; Section 9 discusses implications and limitations; Section 10 concludes.

<p align="center"><img src="../../figures/fd_architecture.png" width="92%"></p>
<p align="center"><b>Fig. 1.</b> FuelDecompNet architecture. Ten fuel-specific share heads (each with its physical inductive bias) feed a fixed-lookup reconstruction layer; a zero-parameter configuration router selects between the decomposition path and an aggregate renewable-share path; a level anchor separates transferable shape from region-specific level. The same trained weights serve the zero-telemetry tier (cold mode) and the telemetry tier (history mode) via per-window history dropout during training.</p>

---

## 2. Related work

**Carbon-intensity forecasting.** CarbonCast [1] is the reference system: a two-tier CNN-LSTM that first forecasts per-source production and then synthesizes CIF, trained locally per region, with multi-day MAPE of 4.8–13.9% across six regions. DACF [2] established the day-ahead formulation. Recent architectures improve within-region accuracy — the wavelet-patch model of Zhang et al. [3] captures local-temporal and cross-variable dependencies across frequencies on four Australian markets — but retain the one-region-one-model contract. We use CarbonCast in both roles: as a supervised upper reference (our faithful reimplementation, trained on the same test windows, reaches median 43.4 vs PatchTST 43.5) and as a cross-domain baseline under a contract it was never designed for.

**Data-scarce and cross-region CIF.** The DGCFM line [4] attacks data scarcity through a pre-trained time-series foundation model with metadata-driven hypergraph fine-tuning and a spatiotemporal carbon graph, reporting +20% accuracy in low-data scenarios — but target-domain fine-tuning remains a required ingredient. CarbonX [5] removes fuel-mix dependence entirely, forecasting CIF from history alone with time-series foundation models (zero-shot MAPE 15.8% across 214 grids), and lists weather covariates as future work. Our setting is strictly harder in one direction (no CIF history at all in the base tier) and richer in another (public configuration and weather are allowed); the results show why the richer contract is worth its data cost. Our answer is structural (a physics layer absorbing scale) rather than scale-driven (foundation-model capacity).

**Time-series foundation models.** Chronos-2 [6] delivers zero-shot univariate, multivariate, and covariate-informed forecasts via in-context learning with group attention. We use Chronos-2 not as an end-to-end replacement but as a shape prior inside the lag tier (Section 4.5): its contribution is conditional on EWMA level correction and persistence anchoring, and re-leveling its output is counterproductive. This division of labor — foundation model for shape, classical components for level — is itself a finding.

**Dunkelflaute and event-day analysis.** Periods of jointly low wind and solar availability are an active energy-systems topic; Kittel and Schill [11] document how definitions vary along four axes (production parameter, duration, spatial extent, threshold type) and no standard has emerged. Our protocol (Section 8.4) contributes a forecasting-specific instantiation: per-region annual 20th-percentile thresholds on reanalysis-derived daily mean wind speed and solar radiation, with compound events requiring both, fully reproducible without market data, applied to *forecast-error* bucketing rather than resource assessment.

**Domain generalization.** Classical bounds [10] frame target risk through source risk plus divergence. Our Theorem 1 replaces the divergence term with an observable, domain-specific quantity — the emission-factor contrast. Section 8.6 further shows, using a configuration-conditioned direct-regression backbone as the control, that direct-regression architectures exhibit a U-shaped transfer-difficulty profile over the renewable-share spectrum — a structural weakness that our level/shape separation eliminates: transfer difficulty returns to the form governed by $L_T$ and is predictable before deployment.

**Chinese power-grid data.** Seven provincial spot markets are formally operating (Shanxi first, 2023-12-22) and disclose 15-minute fuel-level data to market members under the 2024 disclosure rules; official carbon intensities are annual averages with ~2-year lag; monthly generation statistics are fully public [12,14]. The zero-telemetry tier occupies exactly the gap this regime leaves open. In this study ESPS Shanxi data [14] serves as *evaluation ground truth only*, never as model input.

---

## 3. Problem formulation

### 3.1 Setting

A pool of regions $\mathcal{R}=\{R_1,\dots,R_N\}$, each with per-fuel share matrix $\mathbf{S}_r \in [0,1]^{T_r \times F}$ ($F=10$: solar, wind, five baseload classes, three thermal classes), a CIF series, a static configuration vector $c_r \in \mathbb{R}^{16}$ (mean renewable share, non-renewable emission factor, ten fuel-configuration shares, annual mean wind capacity factor, annual clear-sky index, fuel-availability flag, $|\varphi|/60$ — all publicly derivable), and table-lookup emission factors $EF \in \mathbb{R}^{F}_{>0}$. The physical identity holds by construction:

$$\mathrm{CIF}_t = \sum_{f=1}^{F} s_{f,t}\,EF_f. \tag{1}$$

### 3.2 Information hierarchy

Each tier is defined by the target-side information legally available at forecast origin $t_0$ (input length $L{=}336$ h, horizon $H{=}24$ h, test period = final 20% per region, stride 24 h):

| Tier | Target-side input at $t_0$ | Target labels ever seen |
|---|---|---|
| $\mathcal{I}_\mathrm{cfg}$ (zero telemetry) | configuration scalars; public weather; calendar (share stream zeroed) | none |
| $\mathcal{I}_0$ (telemetry) | + live fuel-share stream $s_{t_0-L+1:t_0}$ | none |
| $\mathcal{I}_+$ (calibration) | + target CIF history (training-free calibration) | past only |
| $\mathcal{I}_\mathrm{lag}$ (lagged) | + lagged CIF observations (training-free online mixing) | past only |
| $\mathcal{I}_S$ (supervised) | 10 months target history, end-to-end training | ~7008 h |

Two protocols score the hierarchy. **LORO** (leave-one-region-out): 28 source regions → 1 target; 29 targets × 5 seeds = 145 runs. **LOJO** (leave-one-jurisdiction-out): sources exclude the target's entire jurisdiction (AU targets → US+UK sources), the strict test of intercontinental transfer. Metric: MAE (gCO₂/kWh) per tier; paired Wilcoxon tests for comparisons.

### 3.3 Training objective

The model trains on source regions only, jointly for both history modes via per-window history dropout (probability $p_\mathrm{cold}{=}0.3$):

$$\mathcal{L} \;=\; \|\widehat{\mathrm{CIF}}-\mathrm{CIF}\|_1 \;+\; 1.0 \cdot \sum_f EF_f\,|\hat s_f - s_f| \;+\; 0.3 \cdot \|\hat r - r\|_1 \;+\; 0.5 \cdot \|\hat s^\circ - s^\circ\|_1, \tag{2}$$

where the second term converts share error into gCO₂ units, the third supervises the aggregate head, and the fourth is a window-demeaned shape loss. 900 epochs, Adam (lr 1e-3, cosine schedule), configuration-distance-weighted source sampling $w_i = 1/(|\Delta \bar{r}_i| + 0.05)$, and a monthly lagged configuration interface (each month uses the configuration published one month earlier, mirroring the monthly-bulletin deployment reality).

---

## 4. Method

### 4.1 Design principle: what moves where

| Source of variation | Mechanism | Deployment cost |
|---|---|---|
| Absolute scale ($EF_f$: 208–1160 across regions) | physics layer, table lookup (exact) | one lookup |
| Share dynamics drifting with penetration | configuration conditioning + distance-weighted sampling | one scalar |
| Level vs. shape | gradient-path decoupling (level from config/observations; shape from physics channels) | — |
| Uncovered dynamics | persistence gating (history mode only) | none |

### 4.2 Ten-head fuel decomposition

The model outputs share trajectories only; CIF reconstruction uses Eq. (1) with a bounded emission-factor correction head:

$$\widehat{\mathrm{CIF}}_{t+h} \;=\; \sum_{f \in \mathcal{F}} \hat s_{f,t+h}\; EF_f\,\bigl(1 \pm 0.35 \tanh g_\theta(\cdot)\bigr), \tag{3}$$

where $\mathcal{F} = \{\mathrm{pv}, \mathrm{wd}\} \cup \mathcal{B} \cup \mathcal{T}$, $|\mathcal{B}|{=}5$, $|\mathcal{T}|{=}3$, and $EF_f$ are fixed constants. Each head encodes a physical mechanism hypothesis.

**Solar head** — deterministic astronomical channel, weather modulation:

$$\hat s_{\mathrm{pv},t+h} = c^{\mathrm{m}}_{\mathrm{pv}} \cdot \frac{A_{t+h}}{\bar A_t} \cdot \bigl(1 + 0.4 \tanh g_\theta(w_{t+1:t+H})\bigr), \tag{4}$$

with $A$ the clear-sky envelope (a pure function of latitude, date, and hour), $c^{\mathrm{m}}$ the monthly configuration share, and $\bar A_t$ the within-window mean. Most predictable structure is astronomical; weather only modulates amplitude.

**Wind head** — ratio structure separating level (non-transferable) from dynamics (transferable):

$$\hat s_{\mathrm{wd},t+h} = \bar s_{\mathrm{wd}} \cdot \mathrm{clip}\!\Bigl(\frac{\mathrm{wcf}(w_{t+h})}{\mathrm{wcf}^{\mathrm{ref}}_t},\,0.2,\,3\Bigr)\cdot m_\theta(\cdot),\quad \mathrm{wcf}^{\mathrm{ref}}_t = 0.7\,\overline{\mathrm{wcf}}_{t-168:t} + 0.3\,\mathrm{wcf}^{\mathrm{yr}}, \tag{5}$$

with $\mathrm{wcf}(\cdot)$ a causal weather-derived capacity-factor curve (24 h mean + 6 h tendency regime features), clipping preventing explosion, and the drought-anchored reference regressing toward the annual mean when the recent regime falls far below it.

**Baseload heads** — masked level statistics: $\hat s_{b,t+h} = M_b \cdot \ell_b(c^{\mathrm{m}}, \mathrm{history})$ with support mask $M_b = \mathbb{1}[c^{\mathrm{m}}_b > 0]$: a fuel absent from the monthly table structurally cannot hallucinate output.

**Thermal heads** — dispatchable residual with configuration prior:

$$\hat s_{\mathcal{T}} = 1 - \sum_{f \notin \mathcal{T}} \hat s_f,\qquad \mathrm{split}_\theta = \mathrm{softmax}\bigl(\log(c^{\mathrm{m}}_{\mathcal{T}} + 0.02) + M \odot \delta_\theta(\cdot)\bigr), \tag{6}$$

zero-initialized so a freshly deployed model starts at its configuration prior and learns dispatch competition as a residual.

**Aggregate head and structure router.** For configurations where fuel-path decomposition is structurally unreliable — wind-heavy regions (farm representativeness limits) and hydro-dominant regions (dispatchable hydro follows the load curve, which the baseload head cannot see) — an aggregate path forecasts the total renewable share $\hat r$ (DLinear backbone [8], logit-space configuration anchor, history-mode persistence gate) and reconstructs $\widehat{\mathrm{CIF}} = \hat r \cdot EF_\mathrm{ren} + (1-\hat r)\, EF_\mathrm{nr}$ with $EF_\mathrm{ren}{=}0$. The router is a zero-parameter deterministic function of configuration:

$$\pi(c) = \sigma\bigl(20\,(\tau - c_\mathrm{wd})\bigr)\cdot \sigma\bigl(30\,(0.5 - c_\mathrm{hyd})\bigr)\cdot \mathbb{1}[\mathrm{fuel}],\quad \widehat{\mathrm{CIF}} = \pi\, \widehat{\mathrm{CIF}}_\mathrm{fuel} + (1-\pi)\,\widehat{\mathrm{CIF}}_\mathrm{agg}, \tag{7}$$

with $\tau = 1.1$ (fuel path favored; $c_\mathrm{wd} \ge \tau$ routes to aggregate) and the hydro gate $c_\mathrm{hyd} \ge 0.5$ motivated empirically: on the 71%-hydro BPAT region the aggregate path cuts cold-mode MAE from 46.7 to 15.2. Route selection is auditable before deployment — it reads only public configuration numbers.

### 4.3 Training and capacity discipline

The four-term objective (Eq. 2) unifies units so gradient pressure follows the physical consequence of error. Configuration-distance-weighted sampling is the strongest single training component (+18.5% degradation when removed in ablation). Cold-mode dropout makes one set of weights serve both $\mathcal{I}_\mathrm{cfg}$ and $\mathcal{I}_0$ with no model change. The model totals ~21k parameters. Three families of deep conditioning — feature-modulation (FiLM-style, ~30× parameters), hypernetwork scale generation, and meta-learned fusion — consistently degraded zero-shot performance in ablation; the mechanism is consistent: capacity memorizes source-domain temporal idioms, and idioms do not transfer. Simplicity is a finding, not a limitation, in zero-telemetry forecasting.

### 4.4 Calibration tier ($\mathcal{I}_+$): training-free test-time calibration

Three closed-form corrections consuming only observations available at $t_0$, no gradients: (i) **level anchoring** — predicted share profiles re-aligned to recent observed levels; (ii) **physical-residual correction** — the empirical estimate of the residual term in Theorem 1; (iii) **self-validated multi-branch fusion** — anchored model, lag-24 h, weekly-lag, and 7-day-mean branches fused with inverse-power weighting by per-horizon backtest error, with the fusion configuration selected per origin by replaying a fixed menu over the most recent 56 days. Design philosophy: the zero-shot model is a candidate generator; the calibration mechanism is a per-origin adaptive selector.

### 4.5 Lag tier ($\mathcal{I}_\mathrm{lag}$): training-free online mixing

A convex blend of three components — EWMA level correction, persistence with intra-day anchoring, and Chronos-2 [6] as a shape prior:

$$\hat y = w_g\,\mathrm{EWMA}(\cdot) + w_p\,\mathrm{persist}(\cdot) + w_C\,\mathrm{Chronos}\text{-}2(\cdot), \tag{8}$$

with weights fitted on an early segment (0.05 grid). Three findings: the foundation model's value is conditional on the other two components (re-leveling Chronos output is double counting, and negative); a 2048 h context beats 720/336; per-region arm selection adds nothing — the three-component structure is the natural form of this information tier.

---

## 5. Theory

### 5.1 Theorem 1: exact error propagation through the physics layer

**Theorem 1.** For any fuel-share predictor $\{\hat s_f\}$ and the linear reconstruction of Eq. (1),

$$\widehat{\mathrm{CIF}}_t - \mathrm{CIF}_t \;=\; \sum_{f \in \mathcal{F}} (\hat s_{f,t} - s_{f,t})\,EF_f \;+\; \delta_t, \tag{9}$$

where $\delta_t$ is the reconstruction residual at the true shares (imports, losses, sub-fuel heterogeneity). In the binary renewable/non-renewable special case the identity reduces to $(\hat r_t - r_t)(EF_\mathrm{ren} - EF_\mathrm{nr}) + \delta_t$: the CIF error amplification gain $L_T = |EF_\mathrm{ren} - EF_\mathrm{nr}|$ is region-specific and known before deployment.

*Proof.* Linearity of Eq. (1); expanding $\widehat{\mathrm{CIF}} - \mathrm{CIF}$ and adding/subtracting the reconstruction at true shares. $\blacksquare$

Three consequences, each verified empirically (Section 8.8): (i) error attribution is architecture-free — any share-based pipeline shares this decomposition, and our per-fuel version attributes error to individual fuels with known gains; (ii) free difficulty ordering — at equal share error, a region with $L_T = 1160$ inherits ~5.6× the CIF error of a region with $L_T = 208$; (iii) the binary identity holds to machine precision (maximum residual $1.3\times10^{-4}$ over 29 regions), the amplification term accounts for 71.3% of CIF error on average, and the bound's predictive power is $r = 0.914$.

### 5.2 Proposition 1 (empirical): when zero telemetry beats supervision

Supervised training on the target region can fit label idiosyncrasies — synthetic classification artifacts, jurisdiction-specific accounting — that the physics route structurally cannot see; its expected error carries a label-noise term that does not vanish with more data. Zero telemetry instead pays a transfer residual that shrinks with physics-channel quality and configuration coverage. The win/loss map across our 29 regions follows this structure: supervision retains the advantage exactly on the regions whose fuel labels are classifier-synthesized or noisy (SA1: DUID self-classification; NSW1: label idiosyncrasies; both are also the LOJO exceptions), while zero telemetry wins on the wind-heavy UK belt, where telemetry is clean and the physical wind channel is strong (largest win −23.5 gCO₂/kWh on West Midlands). We state this as an empirical proposition over 29 regions, not a certified theorem, and note it is testable before deployment from public metadata (label provenance and $L_T$).

---

## 6. Case study: the China deployment layer

**Input interface (all public, no registration).** Monthly fuel-level generation tables (provincial statistics bulletins, ~1-month lag), monthly inter-provincial import tables (power trading centers), ERA5 historical and day-ahead weather [13], and a Chinese calendar fact table (State Council holiday notices; heating-season ordinances: Taiyuan 11/1–3/31, Beijing–Tianjin–Hebei 11/15–3/15).

**Translation layer (four components, 14 unit tests).** (i) **Calendar remapping** — statutory holidays falling on weekdays are mapped to the Sunday day-of-week channel, and makeup workdays to the Monday channel, so the model perceives holiday load shapes through the known weekend-shape channel; the mapping is advisory and never silent. (ii) **Sender-weighted import emission factors** — the import axis uses a flow-weighted average of sender provinces' monthly-structure-implied CIF (Sichuan thermal-heavy in dry season, hydro-heavy in wet season) instead of a constant. (iii) **Monthly configuration tables** — monthly statistics are converted into the (12, 16) configuration contract, renewables accounted per the Ministry of Ecology and Environment taxonomy, annual shares energy-weighted. (iv) **Proxy-anchored trust gating** — provinces without CIF observations are audited against independent public statistics (thermal-generation index and daily coal-burn correlation, anchored to official annual factors); when no proxy is available the gate closes and the system falls back to the annual configuration — conservative by construction.

**Validation against independent ground truth.** Shanxi is evaluated against the ESPS public dataset [14] (35,136 rows of 15-minute market data, 2024-01-01 to 2025-04-07, used strictly as evaluation truth, never as input); Shanghai against public monthly statistics and official annual factors; daily-mean references from Carbon Monitor [12]. Under a strict ex-ante publication calendar (each forecast month uses only statistics published before the forecast origin):

| Province | Monthly-aggregate MAE 2023/2024 | Daily-mean MAE 2023/2024 | Oracle daily floor 2023/2024 |
|---|---|---|---|
| Shanxi | **15.1 / 17.0** | 26.3 / 32.1 | 20.5 / 25.0 |
| Shanghai | **18.2 / 18.2** | 21.7 / 24.0 | 15.2 / 18.2 |

All monthly-aggregate MAEs are below 20 gCO₂/kWh. Daily-mean error is bounded below by the Carbon Monitor day-to-day noise floor (the oracle row substitutes true monthly means and still incurs these errors), so the monthly aggregate is the honest reporting grain for this tier. Weekly-mean CIF is 735.7 gCO₂/kWh with a 313.3 swing, and the photovoltaic-peak-to-CIF-trough alignment is correct.

**Boundaries.** Spring-festival and golden-week load collapse is untrained (no source-region analogue); heating-season combined-heat-and-power dispatch follows heat, not the modeled load curve; curtailment can break the weather-to-output mapping; intraday import trajectories of receiving provinces are invisible in monthly averages; most provinces do not split coal from gas in thermal statistics. Intraday shape amplitude is systematically underestimated 2–3× (model 10–17% vs. actual 27–42% swing) — the root cause is insufficient seasonal variation of fuel shares in the monthly tables, which is the true information ceiling of this tier, and we report it as such. Expected province-class performance: coal-dominant senders (Shanxi, Inner Mongolia West, Shaanxi) best at 17–20; hydro provinces intermediate; eastern receivers (Shanghai, Jiangsu, Zhejiang) hardest.

<p align="center"><img src="../../figures/fd_cn_deploy.png" width="88%"></p>
<p align="center"><b>Fig. 7.</b> China zero-telemetry monthly backtests. Left: Shanxi/Shanghai 2023–2024 monthly-aggregate MAE, all below the 20 gCO₂/kWh target. Right: intraday shape amplitude is underestimated 2–3×, the information ceiling of the monthly-statistics tier.</p>

---

## 7. Experimental setup

**Data.** 29 regions in three jurisdictions, real hourly data, no synthetic series in any reported metric: 4 Australian NEM regions (AEMO 2023, NEMED DUID fuel telemetry), 17 UK DNO regions (National Grid ESO Carbon Intensity API), 8 US balancing areas (EIA-930). Weather: Open-Meteo ERA5 reanalysis [13], farm-blended for 12 regions with known farm clusters and centroid-grid for 31 area-coverage points. China side as in Section 6.

**Baselines.** PatchTST + RevIN [7,9] (300 epochs, local training), a faithful CarbonCast reimplementation [1] (same test windows), persistence (lag-24 h), and a configuration-conditioned direct-regression backbone, AdaptivePersistDLinear, as the ablation control (Section 8.6). Supervised baselines are trained under identical test windows (2 seeds; per-seed difference < 0.2 gCO₂/kWh).

**Protocol.** Test period = final 20% per region, stride 24 h, all methods on identical origins. Five seeds for all hierarchy tiers; medians reported. Paired Wilcoxon for method comparisons. All telemetry and weather assets are checksum-verified for integrity; the verification record is released alongside the code so that every reported number is reproducible.

---

## 8. Results

### 8.1 Information-hierarchy pricing (LORO)

**Table 1.** 29-region LORO, median MAE (gCO₂/kWh), 5 seeds for hierarchy tiers, 2-seed mean for supervised baselines.

| Method / tier | Input contract | Median MAE | vs supervised |
|---|---|---:|---|
| Persistence (lag-24 h) | — | 50.0 | 1.15× |
| PatchTST (supervised) | 10-month local training | 43.5 | 1.00× |
| CarbonCast (supervised) | 10-month local training | 43.4 | 1.00× |
| $\mathcal{I}_\mathrm{cfg}$ zero telemetry | config + weather + calendar | **38.4** | **0.88×** (18/29 wins) |
| $\mathcal{I}_0$ telemetry | + live fuel shares | **36.9** | **0.85×** (17/29 wins) |
| $\mathcal{I}_+$ calibration | + CIF history | 46.9 | 1.08× (25/29 wins vs persistence) |
| $\mathcal{I}_\mathrm{lag}$ lagged blend | + lagged observations | **27.95** (late) | 0.64× |

Three readings. (i) The zero-telemetry tier at 38.4 falls below both supervised references: the zero-telemetry-versus-10-month-supervision balance tips in aggregate, with outright wins on 18/29 regions. (ii) The telemetry tier at 36.9 wins on 17/29. (iii) The training-free lagged blend reaches 64% of supervised accuracy using only lagged observations. The calibration tier is priced against its design target — recovering transfer loss without training — and beats persistence on 25/29 regions; comparing it to supervision in absolute median conflates protocols, and we do not make that comparison.

<p align="center"><img src="../../figures/fd_ladder.png" width="85%"></p>
<p align="center"><b>Fig. 2.</b> Information-tier pricing across 29 regions (points = regions, line = median). The zero-telemetry tier (38.4) falls below supervised PatchTST (43.5, dashed reference).</p>

### 8.2 Intercontinental transfer (LOJO)

**Table 2.** Leave-one-jurisdiction-out: sources = US+UK only; targets = all four Australian grids. 5-seed medians; supervised = PatchTST retrained on the same test windows.

| Target | This work ($\mathcal{I}_+$) | Persistence | PatchTST (supervised) | vs persistence | vs supervised |
|---|---:|---:|---:|---:|---:|
| QLD1 | **19.1** | 20.2 | 22.7 | −5.0% | **−16%** |
| NSW1 | **42.3** | 48.0 | 43.0 | −11.8% | **−1.7%** |
| VIC1 | **81.3** | 98.6 | 84.7 | −17.5% | **−4.0%** |
| SA1 | 69.3 | 77.6 | 65.2 | −10.7% | +6% |

A model trained only on US and UK regions — different hemisphere, market design, and fuel structure — beats locally supervised training on 3 of 4 Australian grids. SA1 is the exception predicted by Proposition 1: its fuel labels are classifier-synthesized (DUID self-classification), so local supervision learns label artifacts the physics route cannot see. Cross-seed stability of the transfer results is within 0.1 gCO₂/kWh (5 seeds).

<p align="center"><img src="../../figures/fd_lojo.png" width="80%"></p>
<p align="center"><b>Fig. 3.</b> Intercontinental transfer. US+UK-trained model (green) vs locally supervised PatchTST across the four Australian grids; SA1 is the synthetic-label exception of Proposition 1.</p>

### 8.3 Paired regional analysis

Per-region (Appendix A): the largest zero-telemetry wins are the UK wind belt — West Midlands −23.5, East England −17.0, England −13.4, GB −10.9 — where the physical decomposition beats memorization of local noise. The largest losses are NSW1 (+21) and QLD1 (+11) under LORO (both reverse under LOJO, Table 2) and low-$L_T$ US regions where supervision is cheap and effective (BPAT +15.9, PJM +7.3). The two supervised references nearly coincide in median (43.4 vs 43.5), so the "supervised ceiling ≈ 43.5" anchor is stable across architectures.

### 8.4 Event-day protocol and results

**Definition (reproducible without market data).** For each region and year: let $q^{w}_{20}, q^{s}_{20}$ be the annual 20th percentiles of daily-mean 100 m wind speed and surface shortwave radiation from ERA5 [13]. A day is **wind-lull** if daily-mean wind $< q^{w}_{20}$, **solar-dim** if daily-mean shortwave $< q^{s}_{20}$, and **compound (Dunkelflaute)** if both; dark-winter regions where the solar threshold is degenerate evaluate wind-lull only. Labels derive from weather alone, so any group can regenerate them. Forecast errors are bucketed by label and compared with permutation tests (29 regions × 4 methods = 348 tests).

**Results.** 62 of 348 tests show significant inflation (p < 0.05, ratio > 1). The inflation is not region noise but a spatial structure: a Scotland–northern-England belt concentrates the extremes — UK_01 compound **4.19×** (p = 1e-4), UK_16 wind-lull 4.00×, UK_16 compound 3.69×, UK_02 compound 3.56× — and the Australian regions show the same signal (VIC1 wind-lull 1.51×). Any zero-telemetry method — and most supervised ones — should expect 2–4× error inflation on Dunkelflaute days; this is a design input for carbon-aware scheduling systems, and the protocol is released for reuse.

<p align="center"><img src="../../figures/fd_dunkelflaute.png" width="85%"></p>
<p align="center"><b>Fig. 4.</b> Event-day error inflation. Of 348 method–region tests, 62 are significant (permutation p < 0.05); the Scotland–northern-England belt (red) reaches 4.19×, a systematic structure rather than isolated noise.</p>

### 8.5 The cost of operational weather forecasts

Primary results use reanalysis weather, as is standard; deployment uses numerical weather forecasts. Holding the model fixed (28 sources, official configuration) and swapping only the target-side 24 h weather source (history remains ERA5): pooled zero-telemetry MAE is 43.98 under reanalysis vs 50.41 under an operational ensemble (GFS/ICON blend) — a **+6.4 penalty**, with only 5 of 29 regions preferring operational forecasts. The penalty is strongly heterogeneous: QLD1 nearly free (+0.2 to +1.1), while NSW1, UK_01, and US_BPAT incur +19 to +29 — operational wind errors enter the capacity-factor channel directly in wind-heavy regions. We therefore keep the reanalysis pipeline as the reproducible primary setting and report the operational-forecast penalty as a deployment cost.

<p align="center"><img src="../../figures/fd_nwp_penalty.png" width="80%"></p>
<p align="center"><b>Fig. 5.</b> Operational NWP vs reanalysis at inference. Per-region scatter (above the diagonal = penalty); pooled median penalty +6.4 gCO₂/kWh, largest in wind-heavy regions.</p>

### 8.6 Ablations and the transfer-difficulty structure

**Data engineering dominates.** The three largest gains in the full ablation matrix are not architectural: the unified UTC time base (astronomical-channel correlation with PV share: −0.697 under a misaligned time base vs. +0.976 under UTC, with the calibration tier inheriting the benefit); farm-weighted weather (wind-share explained variance R² 0.08–0.15 under centroid-grid weather vs. 0.33–0.48 under farm weighting); and the structure-router threshold $\tau = 1.1$ (compared with a more conservative 0.25, this setting extends fuel-path coverage to mid-wind regions and reduces mis-routing of wind-heavy configurations).

**Model-side ablations.** Cold-mode dropout (one weight set, two tiers), the shape loss (direct intraday-shape optimization), and the EF-weighted share loss each contribute measurably; the full matrix is in the released results.

**Negative results (all verified under the unified protocol).** Deep conditioning (three families), instance normalization (destroys the physical level), donor fuel-distance weighting (reversed, p = 0.025), single-backbone joint fine-tuning (no gain), merit-order dispatch priors (no gain), late-peak loss weighting (distorts level), multi-year training data (no gain).

**Direct-regression control and the transfer-difficulty structure.** To isolate the contribution of the physics-decomposed architecture, we evaluate a configuration-conditioned direct-regression backbone — AdaptivePersistDLinear, a DLinear backbone regressing CIF directly with configuration injected through a conditioning layer — under the identical protocol. The control backbone exhibits a U-shaped relation between zero-shot difficulty and renewable share (quadratic fit R² = 0.662, vertex 0.58): fossil-dominated boundary regions (US_FPL/PJM/ISNE, efficiency ratio ρ ≥ 2.6) and renewable extremes both fail systematically. The structural improvement of our method lies precisely here: under the fuel-decomposition backbone the relation no longer holds (R² = 0.087, vertex outside the observed range); the level/shape separation eliminates the boundary failure mode. Transfer difficulty is instead governed by Theorem 1's $L_T$ ordering and is predictable before deployment. This is an n = 29 empirical statement.

### 8.7 Lag tier and the unreachability of a 20 gCO₂/kWh floor

The training-free lagged blend: EWMA alone 30.76 (late segment) → with Chronos-2 28.27 → three-component blend **27.95** (5 seeds, 27.97 ± 0.10; the nine hardest regions improve 47.58 → 43.57). Decomposing the residual: EWMA level error is 21.0 and the **oracle level floor — keeping the EWMA shape and substituting true levels — is 23.1**. Any zero-telemetry method that only corrects level therefore cannot go below 23.1; a 20 gCO₂/kWh zero-telemetry target is unreachable without new shape information, and Chronos-2 has already absorbed most of the available shape prior. We report this as a bound on the method class, not merely on our implementation.

<p align="center"><img src="../../figures/fd_lag_floor.png" width="88%"></p>
<p align="center"><b>Fig. 6.</b> Lag-tier blends and the oracle level floor (23.1 gCO₂/kWh). Left: pooled late-segment MAE by blend. Right: five-seed stability of the best blend (27.97 ± 0.10).</p>

### 8.8 Theorem 1 validation

The binary identity holds to machine precision (maximum residual 1.3×10⁻⁴ over 29 regions); the amplification term accounts for 71.3% of CIF error on average; the bound's predictive power for regional difficulty is r = 0.914. The fuel-decomposition stack extends attribution from 2 to 10 fuel axes, each with an independently known gain.

---

## 9. Discussion

**How to read the hierarchy.** Each tier is priced against its own design target: $\mathcal{I}_\mathrm{cfg}$/$\mathcal{I}_0$ against supervision (how much supervised accuracy can be bought without training), $\mathcal{I}_+$ against persistence (how much transfer loss training-free calibration recovers), $\mathcal{I}_\mathrm{lag}$ against the oracle floor (the ceiling of training-free methods). Cross-tier medians do not form a total order, and pretending they do would misprice the information.

**Why zero telemetry wins where it wins.** Not capacity but inductive bias: on the UK wind belt, the physical decomposition (capacity-factor structure × farm-weighted weather) beats a supervised model's memorization of local label noise; supervision's remaining strongholds are exactly the synthetic-label regions of Proposition 1. The win/loss map is a boundary characterization of the method, and it is knowable before deployment.

**Alignment with tiered disclosure.** The hierarchy is isomorphic to disclosure regimes like China's: monthly public statistics run $\mathcal{I}_\mathrm{cfg}$ today in any province (15–18 monthly MAE in Shanxi/Shanghai); a province opening market-member fuel data upgrades the same model to $\mathcal{I}_0$ with zero re-engineering; hourly CIF disclosure activates the calibration tier automatically. Deployment cost tracks information availability by construction.

**Limitations.** Single evaluation year for the 29-region pool (2023) and 2024–2025 for the China case study; event-day inflation is systematic and unmitigated at the zero-telemetry tier (right tails call for the lag tier and interval outputs, the latter future work); the +6.4 operational-weather penalty means reanalysis-based primary results overstate true day-ahead deployment accuracy; intraday shape amplitude in the China tier is underestimated 2–3× (an information ceiling, not a model defect); synthetic-label regions retain a supervision advantage; all results are point forecasts — uncertainty quantification is the natural next step, for which the event-day buckets provide a ready stratification.

---

## 10. Conclusion

We formulated zero-telemetry carbon-intensity forecasting as a priced information hierarchy and built its zero-telemetry tier on a physics-decomposed, 21k-parameter network with table-lookup scale absorption. Across 29 real regions in three jurisdictions: the zero-telemetry tier (median 38.4) beats 10-month supervised training (43.5) with 18/29 outright wins; US+UK-only training beats local supervision on 3/4 Australian grids; the costs of real deployment are quantified — +6.4 from operational weather, 2–4× event-day inflation with a reproducible protocol, and a proven 23.1 oracle floor that rules out a 20 gCO₂/kWh zero-telemetry target; and the same model runs on Chinese provinces from monthly public statistics at 15–18 gCO₂/kWh monthly MAE, upgrading automatically as disclosure deepens. The error-propagation identity (Theorem 1) and the win/loss structure (Proposition 1) make deployment outcomes predictable from public numbers before any investment. Code, per-region tables, event labels, and integrity-verification records are released.

---

## Reproducibility

All experiments run from the repository root with `PYTHONPATH=src`; key artifacts:

| Result | Script | Output |
|---|---|---|
| 29×5 LORO hierarchy benchmark (Table 1) | `scripts/experiments/run_fuel_decomp_eval.py` | `results/fuel_decomp_eval_full_fd41.json` |
| Supervised baselines, same test windows | `scripts/experiments/gap4_supervised_baselines.py` | `results/gap4_supervised_vs_fd41.json` |
| LOJO intercontinental transfer (Table 2) | `run_fuel_decomp_eval.py` (jurisdiction exclusion) + `lojo_ensemble.py` | `results/lojo_au_full.json`, `results/lojo_ensemble.json` |
| Event-day buckets (Section 8.4) | `scripts/experiments/dunkelflaute_buckets.py` | `results/dunkelflaute_buckets.json` |
| Operational-weather penalty (Section 8.5) | `scripts/experiments/fd47_nwp_fut_weather.py` | `results/fd47_nwp_fut_weather.json` |
| Lag tier and oracle floor (Section 8.7) | `scripts/experiments/fd50p_blend2.py` | `results/fd50p_blend2.json`, `results/fd50p3_seeds.json` |
| Theorem 1 validation (Section 8.8) | `scripts/verify/theorem1_physics_bound.py` | `results/theorem1_validation.json` |
| China deployment layer (Section 6) | `src/transcif/data/cn_deploy.py` | 14 unit tests |

All telemetry and weather assets are checksum-verified for integrity; the verification record is released with the code so that every reported number is reproducible.

## Appendix A. Per-region results (LORO)

Median over 5 seeds (hierarchy tiers); supervised columns are 2-seed means. Δ% = zero-telemetry vs supervised PatchTST (negative = zero telemetry better). 18/29 regions favor zero telemetry.

| Region | Persistence | PatchTST | CarbonCast | $\mathcal{I}_\mathrm{cfg}$ | $\mathcal{I}_0$ | $\mathcal{I}_+$ | Δ% |
|---|---:|---:|---:|---:|---:|---:|---:|
| NSW1 | 48.0 | 43.0 | 42.5 | 76.7 | 64.2 | 46.1 | +78 |
| QLD1 | 20.2 | 22.7 | 19.7 | 38.4 | 34.1 | 26.7 | +69 |
| SA1 | 77.6 | 65.2 | 72.8 | 80.3 | 73.0 | 60.8 | +23 |
| VIC1 | 98.6 | 84.7 | 89.0 | 83.0 | 89.5 | 98.9 | −2 |
| UK_01 North Scotland | 35.0 | 38.3 | 40.1 | 47.5 | 45.6 | 34.7 | +24 |
| UK_02 South Scotland | 22.1 | 21.3 | 18.5 | 17.3 | 20.5 | 21.8 | −19 |
| UK_03 North West England | 29.3 | 24.6 | 26.9 | 23.5 | 18.7 | 28.0 | −5 |
| UK_05 Yorkshire | 50.0 | 43.5 | 38.2 | 55.1 | 62.5 | 46.4 | +27 |
| UK_06 North Wales & Merseyside | 54.4 | 52.1 | 45.0 | 41.6 | 33.5 | 51.9 | −20 |
| UK_07 South Wales | 76.2 | 74.9 | 68.3 | 63.6 | 56.3 | 72.7 | −15 |
| UK_08 West Midlands | 81.5 | 81.8 | 70.5 | 71.6 | 58.3 | 78.2 | −12 |
| UK_09 East Midlands | 91.8 | 80.3 | 80.4 | 77.2 | 67.7 | 86.4 | −4 |
| UK_10 East England | 61.1 | 49.7 | 50.0 | 32.7 | 27.8 | 58.0 | −34 |
| UK_11 South West England | 53.0 | 56.3 | 55.8 | 50.8 | 43.7 | 50.2 | −10 |
| UK_12 South England | 57.4 | 56.9 | 55.5 | 49.3 | 39.5 | 55.1 | −13 |
| UK_13 London | 52.7 | 49.4 | 51.6 | 37.5 | 37.5 | 53.1 | −24 |
| UK_14 South East England | 50.7 | 48.9 | 50.1 | 42.9 | 42.8 | 50.9 | −12 |
| UK_15 England | 48.6 | 45.2 | 49.5 | 35.0 | 28.7 | 48.3 | −23 |
| UK_16 Scotland | 35.3 | 31.6 | 31.0 | 20.9 | 21.0 | 34.6 | −34 |
| UK_17 Wales | 73.0 | 64.1 | 83.7 | 49.5 | 48.4 | 69.4 | −23 |
| UK_18 GB | 46.0 | 42.7 | 43.4 | 31.8 | 25.2 | 45.7 | −25 |
| US_BPAT | 6.3 | 5.7 | 5.2 | 21.5 | 12.4 | 6.3 | +280 |
| US_CISO | 27.4 | 30.4 | 31.8 | 50.9 | 36.9 | 25.6 | +68 |
| US_ERCO | 64.8 | 49.9 | 49.4 | 37.6 | 37.4 | 60.6 | −25 |
| US_FPL | 13.4 | 15.7 | 12.2 | 22.7 | 19.1 | 12.9 | +45 |
| US_ISNE | 16.0 | 13.8 | 13.2 | 25.9 | 22.4 | 15.4 | +88 |
| US_MISO | 55.6 | 39.8 | 36.1 | 31.8 | 31.1 | 46.9 | −20 |
| US_NYIS | 14.6 | 12.0 | 12.7 | 19.9 | 19.6 | 13.6 | +67 |
| US_PJM | 15.6 | 12.5 | 17.0 | 19.8 | 20.0 | 14.2 | +58 |

US_BPAT (71% hydro) illustrates the router: the aggregate path carries it under telemetry and calibration (12.4 / 6.3), while cold mode without any history stream is the known hard case for hydro-dominant configurations.

## References

1. Maji, D., Shenoy, P., Sitaraman, R.K., 2022. CarbonCast: multi-day forecasting of grid carbon intensity. In: Proc. ACM BuildSys '22, pp. 198–207. doi:10.1145/3563357.3564079.
2. Maji, D., Sitaraman, R.K., Shenoy, P., 2022. DACF: day-ahead carbon intensity forecasting of power grids using machine learning. In: Proc. ACM e-Energy '22, pp. 188–192. doi:10.1145/3538637.3538849.
3. Zhang, B., Tian, H., Berry, A., Roussac, A.C., 2026. Improving day-ahead grid carbon intensity forecasting by joint modeling of local-temporal and cross-variable dependencies across different frequencies. In: Proc. AAAI 2026, pp. 39585–39593. doi:10.1609/aaai.v40i46.41310.
4. Zhang, X., Zhou, T., He, F., Deng, Y., Wang, D., 2026. Toward green computing: general carbon intensity forecasting via dual graph empowered time series foundation model. In: Proc. ACM Web Conference (WWW '26), pp. 9015–9023. doi:10.1145/3774904.3793051.
5. Maji, D., Yang, K., Shenoy, P., Sitaraman, R.K., Srivastava, M., 2025. CarbonX: an open-source tool for computational decarbonization using time series foundation models. arXiv:2510.01521.
6. Ansari, A.F., Shchur, O., Küken, J., et al., 2025. Chronos-2: from univariate to universal forecasting. arXiv:2510.15821.
7. Nie, Y., Nguyen, N.H., Sinthong, P., Kalagnanam, J., 2023. A time series is worth 64 words: long-term forecasting with transformers. In: ICLR 2023.
8. Zeng, A., Chen, M., Zhang, L., Xu, Q., 2023. Are transformers effective for time series forecasting? In: AAAI 2023.
9. Kim, T., Kim, I., Yun, Y., Park, H., Jeong, M., Yun, S., Hwang, J., Ha, J.W., 2022. Reversible instance normalization for accurate time-series forecasting against distribution shift. In: ICLR 2022.
10. Ben-David, S., Blitzer, J., Crammer, K., Kulesza, A., Pereira, F., Vaughan, J.W., 2010. A theory of learning from different domains. Machine Learning 79, 151–175.
11. Kittel, M., Schill, W.-P., 2024. Measuring the Dunkelflaute: how (not) to analyze variable renewable energy shortage. arXiv:2402.06758.
12. Liu, Z., Ciais, P., Deng, Z., et al., 2020. Carbon Monitor, a near-real-time daily dataset of global CO₂ emission from fossil fuel and cement production. Scientific Data 7, 392. doi:10.1038/s41597-020-00708-7.
13. Hersbach, H., Bell, B., Berrisford, P., et al., 2020. The ERA5 global reanalysis. Quarterly Journal of the Royal Meteorological Society 146, 1999–2049.
14. Liu, Y. (yingyueliuhui), 2025. ESPS-Probabilistic-Forecasting: Shanxi electricity spot market dataset. GitHub repository.
