"""Acquire and hash the public Energy-Charts v2 hourly Portugal series."""
from __future__ import annotations

import hashlib
import json
from datetime import datetime, timezone
from pathlib import Path
from urllib.request import Request, urlopen

ROOT = Path(__file__).resolve().parents[1]
URL = "https://api.energy-charts.info/v2/public_power?country=pt&start=2022-01-01&end=2025-01-01"
OUT = ROOT / "data/energy-charts-pt-cn5-r1"


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=False)
    request = Request(URL, headers={"User-Agent": "TransCIF-research/1.0 (CC-BY-4.0 attribution: energy-charts.info)"})
    with urlopen(request, timeout=60) as response:
        payload = response.read()
        headers = dict(response.headers.items())
    raw = OUT / "pt_2022_2024.json"
    raw.write_bytes(payload)
    parsed = json.loads(payload)
    manifest = {
        "schema": "energy-charts-v2-public-power-acquisition-r1",
        "retrieved_utc": datetime.now(timezone.utc).isoformat(), "url": URL,
        "http_headers": headers, "file": raw.name,
        "sha256": hashlib.sha256(payload).hexdigest(), "bytes": len(payload),
        "country": parsed.get("country"), "timezone": parsed.get("timezone"),
        "resolution": parsed.get("resolution"), "unit": parsed.get("unit"),
        "generated_at": parsed.get("generated_at"),
        "available_from": parsed.get("available_from"),
        "available_until": parsed.get("available_until"),
        "records": len(parsed.get("data", [])), "series": parsed.get("series", []),
        "license": "CC BY 4.0; attribution: energy-charts.info",
    }
    (OUT / "acquisition_manifest.json").write_text(json.dumps(manifest, ensure_ascii=False,
        indent=2) + "\n", encoding="utf-8")
    print(json.dumps({k: manifest[k] for k in (
        "file", "sha256", "bytes", "timezone", "resolution", "available_from",
        "available_until", "records", "series", "license")}, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
