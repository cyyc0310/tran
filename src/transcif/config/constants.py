"""TransCIF global constants and region configurations.

Extracted from the legacy ``scripts/transcif_pipeline.py`` so that all
experiment scripts share a single source of truth instead of hard-coding
sequence lengths, strides, and per-region emission factors.
"""

from pathlib import Path

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
# Data directory lives at the repository root (excluded from git via .gitignore).
DATA_DIR = Path(__file__).resolve().parent.parent.parent.parent / "data_2023"
RESULTS_DIR = Path(__file__).resolve().parent.parent.parent.parent / "results"

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


def get_region_config(region_name: str) -> dict:
    """Return the config dict for a known region, raising on unknown names."""
    if region_name in ALL_REGION_CONFIGS:
        return ALL_REGION_CONFIGS[region_name]
    if region_name in UK_REGIONS:
        return UK_REGIONS[region_name]
    raise KeyError(f"Unknown region: {region_name}")
