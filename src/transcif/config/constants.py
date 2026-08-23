"""TransCIF global constants and region configurations.

Extracted from the legacy ``scripts/transcif_pipeline.py`` so that all
experiment scripts share a single source of truth instead of hard-coding
sequence lengths, strides, and per-region emission factors.
"""

from pathlib import Path
import json

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
# Data directory lives at the repository root (excluded from git via .gitignore).
DATA_DIR = Path(__file__).resolve().parent.parent.parent.parent / "data_2023"
RESULTS_DIR = Path(__file__).resolve().parent.parent.parent.parent / "results"
FUEL_DIR = DATA_DIR / "fuel"

# ---------------------------------------------------------------------------
# Sequence / training hyperparameters
# ---------------------------------------------------------------------------
SEQ_LEN = 336          # input history length (hours)
HORIZON = 24           # forecast horizon (hours)
TRAIN_STRIDE = 6       # stride for training window sampling
TEST_STRIDE = 24       # stride for test window sampling
TRAIN_FRACTION = 0.8   # train / test temporal split

EPOCHS_SUPERVISED = 300
EPOCHS_CARBONCAST = 300
EPOCHS_ZERO_SHOT = 150
assert EPOCHS_CARBONCAST == 300  # keep linter happy, used by external imports
BATCH_SIZE = 256

# Seed configurations (shared across all experiment scripts)
SEEDS_FULL = [0, 1, 2, 3, 4]
SEEDS_QUICK = [0, 1, 2]

# ---------------------------------------------------------------------------
# Region configurations
# ---------------------------------------------------------------------------
# Each entry: file = hourly CSV name, ef_r = renewable emission factor
# (tCO2/MWh), ef_nr = non-renewable emission factor (tCO2/MWh).
AU_REGIONS = {
    "QLD1": {"file": "QLD1_2023_hourly.csv", "ef_r": 0.0, "ef_nr": 841.59},
    "NSW1": {"file": "NSW1_2023_hourly.csv", "ef_r": 0.09, "ef_nr": 875.23},
    "VIC1": {"file": "VIC1_2023_hourly.csv", "ef_r": 0.0, "ef_nr": 1160.12},
    "SA1":  {"file": "SA1_2023_hourly.csv",  "ef_r": 0.0, "ef_nr": 490.43},
}

US_REGIONS = {
    "US_CISO": {"file": "US_CISO_2023_hourly.csv", "ef_r": 0.0, "ef_nr": 342.8},
    "US_PJM":  {"file": "US_PJM_2023_hourly.csv",  "ef_r": 0.0, "ef_nr": 347.6},
    "US_MISO": {"file": "US_MISO_2023_hourly.csv", "ef_r": 0.0, "ef_nr": 534.1},
    "US_ERCO": {"file": "US_ERCO_2023_hourly.csv", "ef_r": 0.0, "ef_nr": 470.3},
    "US_ISNE": {"file": "US_ISNE_2023_hourly.csv", "ef_r": 0.0, "ef_nr": 299.1},
    "US_NYIS": {"file": "US_NYIS_2023_hourly.csv", "ef_r": 0.0, "ef_nr": 287.3},
    "US_FPL":  {"file": "US_FPL_2023_hourly.csv",  "ef_r": 0.0, "ef_nr": 340.9},
    "US_BPAT": {"file": "US_BPAT_2023_hourly.csv", "ef_r": 0.0, "ef_nr": 207.5},
}

# UK DNO regions (17) are discovered at runtime from the data directory via
# ``discover_uk_regions``; ef_nr for each is parsed from the CSV.
UK_REGIONS = {}

# Combined lookup used by loaders and experiments.
ALL_REGION_CONFIGS = {**AU_REGIONS, **US_REGIONS}


# ---------------------------------------------------------------------------
# Multi-fuel config (Stage A) — per-fuel share vectors for richer config space
# ---------------------------------------------------------------------------
# Fuel share vectors are loaded lazily from data_2023/fuel/fuel_shares_*.json,
# produced by scripts/data/extract_fuel_breakdown.py (US) and
# extract_uk_fuel_breakdown.py (UK).  When a region has fuel data its config
# vector is extended with per-fuel shares; otherwise it falls back to the
# legacy 2-D [mean_rs, ef_nr/1000] vector (full backward compatibility).
#
# The canonical fuel order is fixed across US/UK so the multi-fuel config
# dimensions are comparable.  AU regions stay on 2-D until NEMED DUID-level
# fuel data is wired in (see IMPROVEMENT_PLAN.md TODO).
_FUEL_SHARES_CACHE: dict = {}
_FUEL_ORDER: list = ["coal", "gas", "nuclear", "petroleum", "hydro", "solar", "wind"]
_FUEL_EFS: dict = {}


def _load_fuel_shares():
    """Lazily load fuel-share JSONs from disk (idempotent, cached)."""
    if _FUEL_SHARES_CACHE:
        return
    global _FUEL_ORDER
    seen_orders = []
    for name in ("fuel_shares_us.json", "fuel_shares_uk.json",
                 "fuel_shares_au.json"):
        path = FUEL_DIR / name
        if not path.exists():
            continue
        with open(path) as f:
            doc = json.load(f)
        order = doc.get("_fuel_order")
        if order:
            seen_orders.append(order)
        for ef_key, ef_val in doc.get("_emission_factors", {}).items():
            _FUEL_EFS.setdefault(ef_key, ef_val)
        for region_key, shares in doc.get("regions", {}).items():
            _FUEL_SHARES_CACHE[region_key] = shares
    # Canonical order = stable union of all fuels seen across jurisdictions,
    # so no per-fuel dimension is silently dropped when US and UK differ
    # (e.g. US has petroleum; UK has biomass/imports/other).  Fuels present in
    # the seed order keep their position; new fuels are appended.
    if seen_orders:
        union = []
        for order in seen_orders:
            for f in order:
                if f not in union:
                    union.append(f)
        _FUEL_ORDER = union


def get_fuel_shares(region_name: str):
    """Return the fuel-share dict for a region, or None if unavailable."""
    _load_fuel_shares()
    return _FUEL_SHARES_CACHE.get(region_name)


def get_fuel_order() -> list:
    """Canonical fuel key order for the multi-fuel config vector."""
    _load_fuel_shares()
    return list(_FUEL_ORDER)


def get_fuel_emission_factors() -> dict:
    """Per-fuel emission factors (gCO2/kWh)."""
    _load_fuel_shares()
    return dict(_FUEL_EFS)


def get_region_config(region_name: str) -> dict:
    """Return the config dict for a known region, raising on unknown names."""
    if region_name in ALL_REGION_CONFIGS:
        return ALL_REGION_CONFIGS[region_name]
    if region_name in UK_REGIONS:
        return UK_REGIONS[region_name]
    raise KeyError(f"Unknown region: {region_name}")
