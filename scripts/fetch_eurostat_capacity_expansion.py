"""Fetch Eurostat annual net capacity slices for four added LOCO countries."""
from __future__ import annotations

import hashlib
import json
from datetime import datetime, timezone
from pathlib import Path
from urllib.request import Request, urlopen

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "data/eurostat-capacity-energy-charts-r2"
COUNTRIES = ("CZ", "SK", "HU", "SI")
BASE = "https://ec.europa.eu/eurostat/api/dissemination/statistics/1.0/data/nrg_inf_epc"


def digest(blob: bytes) -> str:
    return hashlib.sha256(blob).hexdigest()


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    records = []
    for country in COUNTRIES:
        url = (f"{BASE}?geo={country}&freq=A&siec=CF&siec=TOTAL"
               "&plant_tec=CAP_NET_ELC&operator=TOTAL&unit=MW"
               "&time=2020&time=2021&time=2022")
        request = Request(url, headers={"User-Agent": "transcif-research/1.0"})
        with urlopen(request, timeout=90) as response:
            payload = response.read()
            record = {
                "country": country, "url": url, "retrieved_utc": datetime.now(timezone.utc).isoformat(),
                "http_status": response.status, "headers": dict(response.headers.items()),
                "file": f"{country}.json", "bytes": len(payload), "sha256": digest(payload),
                "license": "Eurostat data; source and dataset metadata retained in raw JSON",
            }
        obj = json.loads(payload)
        if obj.get("id") != ["freq", "siec", "plant_tec", "operator", "unit", "geo", "time"]:
            raise ValueError(f"unexpected Eurostat response schema for {country}")
        if obj.get("dimension", {}).get("geo", {}).get("category", {}).get("index", {}).keys() != {country}:
            raise ValueError(f"response country mismatch for {country}")
        (OUT / record["file"]).write_bytes(payload)
        records.append(record)
        print(country, record["bytes"], record["sha256"], flush=True)
    manifest = {
        "schema": "transcif-eurostat-capacity-expansion-r1",
        "dataset": "nrg_inf_epc", "retrieved_utc": datetime.now(timezone.utc).isoformat(),
        "filters": {"freq": "A", "siec": ["CF", "TOTAL"], "plant_tec": "CAP_NET_ELC",
                    "operator": "TOTAL", "unit": "MW", "years": [2020, 2021, 2022]},
        "countries": records,
    }
    (OUT / "acquisition_manifest.json").write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )


if __name__ == "__main__":
    main()
