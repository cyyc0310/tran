"""V386: quantify Shanghai's observed annual gap between local generation and consumption boundary."""

from __future__ import annotations

import csv
import hashlib
import json
from decimal import Decimal
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
INPUT = ROOT / "data/china/shanghai_2024_power_balance_anchors.csv"
OUT = ROOT / "results/v386-shanghai-consumption-boundary-r1"
PROTOCOL = ROOT / "docs/V386_SHANGHAI_CONSUMPTION_BOUNDARY_PROTOCOL.md"
SCRIPT = Path(__file__)


def sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main() -> None:
    if OUT.exists():
        raise FileExistsError(f"Refusing to overwrite frozen output: {OUT}")
    with INPUT.open("r", encoding="utf-8", newline="") as f:
        rows = list(csv.DictReader(f))
    values = {r["metric"]: Decimal(r["value"]) for r in rows}
    use = values["total_electricity_use"]
    local = values["local_generation"]
    imports = values["outside_imports"]
    local_re = values["local_new_energy_generation"]
    consumed_re = values["renewable_consumption"]
    if local + imports != use:
        raise ValueError("Published 2024 balance anchors do not close exactly")

    import_share = imports / use
    local_share = local / use
    renewable_consumption_share = consumed_re / use
    local_re_share_of_use = local_re / use
    local_re_share_of_local_generation = local_re / local
    renewable_accounting_gap = consumed_re - local_re
    # Conditional sensitivity only: if the average imported factor is unknown in [0, 1] kg/kWh,
    # its contribution to the annual consumption-average factor spans import_share kg/kWh.
    import_factor_support_width_g_kwh = import_share * Decimal(1000)

    summary = {
        "experiment": "V386 Shanghai annual consumption boundary audit",
        "status": "SCOPE_AND_INFORMATION_BUDGET_AUDIT_NOT_FORECAST_VALIDATION",
        "input_units": "100 million kWh",
        "balance_closes": True,
        "local_generation_share_of_use": float(local_share),
        "outside_import_share_of_use": float(import_share),
        "reported_renewable_consumption_share": float(renewable_consumption_share),
        "local_new_energy_generation_share_of_use": float(local_re_share_of_use),
        "local_new_energy_share_of_local_generation": float(local_re_share_of_local_generation),
        "renewable_consumption_minus_local_new_energy_generation_100m_kwh": float(renewable_accounting_gap),
        "illustrative_import_factor_support_width_g_kwh_if_import_ef_in_0_to_1000": float(import_factor_support_width_g_kwh),
        "illustrative_assumption": "Holding local annual factor fixed and varying only average imported direct EF over [0,1000] g/kWh; not an empirical EF interval.",
        "limits": [
            "Annual values do not identify monthly or hourly import volumes, sending regions, or imported generation mix.",
            "Renewable consumption accounting and local new-energy generation are related but not asserted to be an exact source attribution identity.",
            "NBS user exports are local generation statistics and do not supply the consumption-side cross-boundary carbon flow.",
            "No hourly CIF labels, forecast accuracy, coverage, or interval efficiency are tested.",
        ],
        "sha256": {str(p.relative_to(ROOT)): sha(p) for p in (INPUT, PROTOCOL, SCRIPT)},
    }
    OUT.mkdir(parents=True)
    (OUT / "summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2, allow_nan=False) + "\n")
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
