# TransCIF: Zero-Shot Cross-Region Carbon Intensity Forecasting

English | [简体中文](README_zh.md)

**TransCIF** is a zero-shot carbon intensity factor (CIF) prediction system for power grids. It forecasts grid-level carbon intensity (tCO2/MWh) for a target region using only that region's publicly available configuration scalars — average renewable share and emission factors — without any target-region training data, fine-tuning, or re-training.

Traditional CIF forecasting requires months of local historical data and per-region model training, which excludes data-scarce regions that need carbon-aware scheduling the most. TransCIF addresses this by decomposing CIF into a renewable share predictor conditioned on region config, plus a closed-form physics layer. It is the first method to demonstrate that zero-shot CIF forecasting across jurisdictions is both feasible and practical.

<p align="center">
  <img src="docs/paper/figures/fig_overall_architecture_v2.png" alt="TransCIF Architecture" width="85%">
</p>

---

## Key Technical Ideas

**Physics decomposition.** CIF is expressed as a linear combination of renewable share and emission factors (`CIF = rs * ef_r + (1-rs) * ef_nr`). Only the renewable share ratio is predicted by the neural network; the emission factors come from the region config. This decomposition yields Theorem 1: CIF prediction error equals renewable share ratio error times a region-specific amplification constant plus a physics residue.

**Config-conditioned lightweight backbone.** A DLinear-inspired linear decomposition encoder (~18k parameters) takes the region's config (average renewable share and non-renewable emission factor) as a conditioning signal, biasing predictions toward the target region. Config-weighted source-region sampling focuses training on regions similar to the target.

**Adaptive persistence gate.** Depending on input volatility and config distance, the model interpolates between its own prediction and a persistence baseline, automatically degrading gracefully in unfamiliar regions.

**TransCIF-ZS+: test-time calibration.** Three-stage closed-form correction — horizontal anchoring, physics-residue correction, and self-verified multi-branch fusion — exploits the target region's observable renewable-share stream without updating any model parameters. This reduces zero-shot MAE by 14% and beats persistence in 29/29 test regions.

**29-region LORO benchmark.** The first leave-one-region-out zero-shot benchmark across three jurisdictions: 4 Australian NEM regions, 17 UK DNO regions, and 8 US EIA-930 balancing authorities.

---

## Key Results

| Method | Performance |
|---|---|
| **TransCIF-ZS** (pure config-only, zero target data) | Median transfer efficiency ratio 1.24 vs. supervised PatchTST (~80% of supervised accuracy); beats persistence in 12/29 regions |
| **TransCIF-ZS+** (with test-time calibration) | Median ratio 1.08; beats persistence in **29/29 regions** (median ratio 0.94); outperforms all cross-domain methods in 27/29 regions |
| **vs. zero-shot CarbonCast** | ZS+ wins in 8/9 representative regions (mean ratio 0.73); CarbonCast degrades up to 2.4× in cross-domain |
| **Theorem 1** (error propagation identity) | Verified at floating-point precision (error 1.3×10⁻⁴) across 29 regions |
| **Theorem 2** (transfer difficulty analysis) | U-shaped difficulty curve over renewable share; weighted config distance correlates with zero-shot ratio r=0.58 (p=0.001) |
| **Conformal prediction** | Valid 90% coverage in 25/29 zero-shot regions, step-stratified split calibration |

<p align="center">
  <img src="figures/main_benchmark.png" alt="Main Benchmark" width="48%">
  <img src="figures/mae_overview_29regions.png" alt="MAE Overview 29 Regions" width="48%">
</p>

<p align="center">
  <em>Left: Main benchmark comparison across methods. Right: Per-region MAE overview covering 29 regions across AUS/UK/US.</em>
</p>

The transfer difficulty follows a U-shaped curve over renewable share, peaking at moderate shares around 50–60%. The weighted config distance between source and target regions correlates significantly with zero-shot performance (see Theorem 2).

<p align="center">
  <img src="figures/theorem2_config_distance.png" alt="Config Distance vs Zero-Shot Ratio" width="65%">
</p>

---

## Data Preparation

The 29-region LORO benchmark spans three jurisdictions. Each region produces a unified hourly CSV in the standard format described below. The data files are not included in the repository due to size; use the following scripts to download and preprocess them.

<p align="center">
  <img src="figures/data_pipeline_overview.png" alt="Data Pipeline Overview" width="90%">
</p>

### Output CSV Format

All download scripts write to `data_2023/` with a uniform schema:

| Column | Type | Description |
|---|---|---|
| `Region` | str | Region identifier (e.g. `QLD1`, `UK_01`, `US_CISO`) |
| `hour` | datetime | UTC hour label |
| `nonrenew_out` | float | Non-renewable sent-out energy (MW) |
| `renew_out` | float | Renewable sent-out energy (MW) |
| `total_emissions` | float | Total CO2 emissions (tCO2) |
| `total_energy_so` | float | Total sent-out energy (MW) |
| `renew_share` | float | Renewable share ratio, ∈ [0, 1] |
| `cif_real_tco2_per_mwh` | float | Real carbon intensity (tCO2/MWh) |
| `cif_real_gco2_per_kwh` | float | Real carbon intensity (gCO2/kWh) |

### Step 1 — AU NEM Regions (4 regions)

The Australian National Electricity Market data is generated via [NEMED](https://github.com/UNSW-CEEM/nemed), which pulls AEMO MMS DISPATCH_UNIT_SCADA records and computes emissions using official emission factors.

```bash
# Requires Python 3.11 and the nemed virtualenv
python -m venv .venv-nemed311
source .venv-nemed311/bin/activate
pip install nemed

# Validate against the test fixture (SA1)
python scripts/generate_nemed_regions.py --validate

# Generate full 2023 hourly data for all 5 NEM regions
python scripts/generate_nemed_regions.py --year 2023
```

This produces `data_2023/{QLD1,NSW1,VIC1,SA1,TAS1}_2023_hourly.csv`. TAS1 is excluded from the main 29-region benchmark (it is physically disconnected and has zero non-renewable emission factor), leaving 4 AU regions: QLD1, NSW1, VIC1, SA1.

**Background**: The NEMED package queries the AEMO generator registration endpoint as of `now - 90 days`. The script patches this to use `ASOF_DATE = 2023/12/01` instead, which is both more correct for 2023 emissions and works around the AEMO archive not yet containing 2026 registrations.

### Step 2 — UK DNO Regions (17 regions)

The UK Carbon Intensity API provides free, no-key-required access to regional generation mix and intensity data for all 18 DNO-level zones. Data is fetched in 14-day chunks at half-hour resolution and aggregated to hourly.

```bash
python scripts/download_uk_regions.py
```

This produces `data_2023/UK_{01..18}_{name}_2023_hourly.csv`. Region ID 18 (Great Britain national) is excluded from the benchmark as it aggregates the DNO regions, leaving 17 UK regions. Renewable fuels are classified as wind, solar, hydro, nuclear, and biomass.

### Step 3 — US EIA-930 Regions (8 regions)

The [EIA-930](https://www.eia.gov/electricity/gridmonitor/) dataset provides hourly generation by fuel source for US Balancing Authorities. The script downloads two 6-month bulk CSVs, merges them, and computes renew_share and CIF using IPCC/EPA emission factors by fuel type.

```bash
python scripts/download_eia930_data.py
```

This produces `data_2023/US_{CISO,PJM,MISO,ERCO,ISNE,NYIS,FPL,BPAT}_2023_hourly.csv` for 8 major US grid operator regions. Emission factors per fuel type (coal=980, natural gas=410, petroleum=650 gCO2/kWh) are sourced from EPA eGRID 2022 and IPCC AR6. Estimated regional ef_nr values are printed at the end of the download script for copying into the region config.

### Data Verification

After downloading all three data sources, run the unified validation to confirm data integrity:

```bash
python scripts/validate_us_data.py
```

This loads all regions, prints a sorted summary table (region, mean_rs, ef_nr, hours, mean_CIF), and runs a quick zero-shot evaluation on target US regions to verify the data pipeline works end-to-end.

### Optional: Temperature Data

The model supports an optional temperature anomaly channel for improved accuracy. Temperature CSVs with columns `(hour, temperature_c)` can be placed alongside the main data files, sourced from [Open-Meteo](https://open-meteo.com/) historical weather archive for a representative city per region. This channel is scale-invariant (same-day-of-year anomaly rather than raw degrees Celsius) to preserve cross-region transfer.

---

## Quick Start

### Requirements

- Python 3.11+ (via `.venv-nemed`)
- PyTorch 2.2+, NumPy, Pandas

### Installation

```bash
cd transcif
python3 -m venv .venv-nemed
source .venv-nemed/bin/activate
pip install torch numpy pandas matplotlib scipy
```

### Download data (3 sources → 29 regions)

```bash
source .venv-nemed/bin/activate

# 1. AU NEM (4 regions) — requires nemed
pip install nemed nemosis nempy
python scripts/generate_nemed_regions.py --year 2023

# 2. UK DNO (17 regions) — free API, no key
python scripts/download_uk_regions.py

# 3. US EIA-930 (8 regions) — bulk download
python scripts/download_eia930_data.py
```

### Basic Usage

Quick evaluation (4 AU regions, 3 seeds):
```bash
source .venv-nemed/bin/activate

# Quick: 4 AU regions, 3 seeds
python scripts/run_unified_eval.py --quick

# Full: 29 regions, 5 seeds
python scripts/run_unified_eval.py

# With all research method comparisons
python scripts/run_unified_eval.py --quick --phys-irm --causal

# Independent method experiments
python scripts/run_phys_irm_eval.py --quick
python scripts/run_causal_eval.py --quick
python scripts/run_rag_eval.py      # retrieval-augmented
python scripts/run_icl_eval.py      # in-context learning
python scripts/run_hier_eval.py     # hierarchical debiased
```

> **Data**: 2023 hourly CSVs under `data_2023/` (not in repo). Download via `scripts/download_*.py` or `scripts/generate_nemed_regions.py`.

---

## Project Structure

```
transcif/
├── scripts/                    # Experiment scripts & model definitions
│   ├── run_unified_eval.py     # Unified LORO evaluation (29 regions, 5 seeds)
│   ├── run_phase1_complete.py  # Phase-1 complete pipeline (all baselines)
│   ├── run_supervised_baselines.py   # Supervised baselines (PatchTST, DLinear, etc.)
│   ├── run_supervised_baselines_v2.py # Supervised baselines v2 (direct CIF prediction)
│   ├── ablation_study.py       # Ablation experiments
│   ├── conformal_prediction.py # Conformal prediction calibration
│   ├── theorem1_physics_bound.py   # Theorem 1 verification
│   ├── theorem2_transfer_bound.py  # Theorem 2 verification
│   ├── carboncast_analysis.py  # CarbonCast cross-domain comparison
│   ├── deployment_warmup.py    # Deployment warmup analysis
│   ├── temporal_ood.py         # Temporal OOD evaluation
│   ├── optimize_weak_regions.py    # Weak region optimization
│   ├── verify_paper_numbers.py     # Paper number verification
│   ├── probe_*.py              # Diagnostic probes (multibranch, multiseed, UK)
│   ├── make_*.py               # Figure generation
│   ├── validate_us_data.py     # US data validation
│   ├── download_eia930_data.py # US EIA-930 data downloader
│   ├── download_uk_regions.py  # UK DNO data downloader
│   └── generate_nemed_regions.py   # AU NEM data generator
├── tests/                      # pytest test suite
│   └── fixtures/               # Test fixtures (real AEMO samples)
├── docs/                       # Documentation
│   ├── paper/                  # Paper drafts (EN + ZH)
│   ├── research/               # Research notes, gap analysis, literature review
│   ├── experiments/            # Experiment reports and findings
│   └── theory/                 # Theorem drafts and derivations
├── results/                    # Experiment results (JSON + logs)
├── figures/                    # Paper figures (PNG + PDF)
├── requirements.txt
├── pytest.ini
└── README.md
```

---

## Core Pipeline

```
Input: renewable share time series + target region config (mean_rs, ef_nr)
         ↓
  Trend/Seasonal Decomposition
    AvgPool(25) → trend | x - trend → seasonal
         ↓
  Config-Conditioned DLinear
    Linear(trend) + Linear(seasonal) + MLP_config_bias → sigmoid
         ↓
  Adaptive Persistence Gate
    gate = σ(MLP(config, recent_mean, recent_std))
    output = gate × persistence + (1 − gate) × model
         ↓
  Physics Layer
    CIF = predicted_share × ef_renew + (1 − predicted_share) × ef_nonrenew
         ↓
  TransCIF-ZS+ (optional test-time calibration, model frozen)
    Level anchoring → physics residue correction → multi-branch fusion → conformal interval
```

---

## Running Tests

```bash
pytest
```

Tests are located under `tests/` and are configured via `pytest.ini` with `pythonpath = src`.

---

## Citation

If you use TransCIF in your research, please cite:

```bibtex
@article{transcif2026,
  title   = {Zero-Shot Cross-Region Carbon Intensity Forecasting
             via Config-Conditioned Physics Decomposition},
  author  = {TransCIF Authors},
  journal = {arXiv preprint},
  year    = {2026}
}
```

Paper drafts are available in `docs/paper/` (English and Chinese versions).

---

## License

This project is for academic research purposes. See `LICENSE.md` for details.
