#!/usr/bin/env python3
"""Acquire the 2026 CN5 response snapshot only if it matches metadata preflight."""
from __future__ import annotations

from datetime import datetime, timezone
import hashlib
from http.client import HTTPException
import json
from pathlib import Path
import time
from urllib.error import HTTPError, URLError
from urllib.request import Request, urlopen

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "data/energy-charts-cn5-2026-holdout-r1"
PREFLIGHT = ROOT / "data/energy-charts-cn5-2026-preflight-r1/metadata_preflight.json"


def fetch(url: str):
    request = Request(url, headers={"User-Agent": "TransCIF research frozen holdout/1.0"})
    for attempt in range(5):
        try:
            with urlopen(request, timeout=180) as response:
                return response.read(), dict(response.headers.items())
        except HTTPError as exc:
            if exc.code != 429 or attempt == 4:
                raise
            time.sleep(max(1, int(exc.headers.get("Retry-After", "5"))))
        except (URLError, TimeoutError, ConnectionError, OSError, HTTPException):
            if attempt == 4:
                raise
            time.sleep(2 ** attempt)
    raise RuntimeError("unreachable")


def stable_payload_hash(doc: dict) -> str:
    stable = dict(doc)
    stable.pop("generated_at", None)
    canonical = json.dumps(stable, sort_keys=True, ensure_ascii=False,
                           separators=(",", ":")).encode("utf-8")
    return hashlib.sha256(canonical).hexdigest()


def main() -> None:
    preflight = json.loads(PREFLIGHT.read_text())
    DATA.mkdir(parents=True, exist_ok=True)
    rows = []
    for item in preflight["countries"]:
        path = DATA / f"{item['country']}_2026.json"
        if path.exists():
            body = path.read_bytes()
            headers = {}
        else:
            body, headers = fetch(item["url"])
        digest = hashlib.sha256(body).hexdigest()
        doc = json.loads(body)
        stable_digest = stable_payload_hash(doc)
        if stable_digest != item["payload_sha256_excluding_generated_at"]:
            raise ValueError(f"API response changed after metadata preflight for {item['country']}; "
                             "stop before model scoring and repeat preflight")
        if doc.get("country") != item["country"] or doc.get("unit") != "MW":
            raise ValueError(f"country or unit mismatch for {item['country']}")
        if not path.exists():
            path.write_bytes(body)
        rows.append({"country": item["country"], "url": item["url"], "http_status": 200,
            "http_date": headers.get("Date"), "file": path.name, "bytes": len(body),
            "sha256": digest, "timezone": doc.get("timezone"), "resolution": doc.get("resolution"),
            "generated_at": doc.get("generated_at"), "available_from": doc.get("available_from"),
            "available_until": doc.get("available_until"),
            "response_sha256": digest,
            "metadata_preflight_content_sha256": item["payload_sha256_excluding_generated_at"]})
        print(f"acquired frozen response {item['country'].upper()}: {len(body)} bytes", flush=True)
        time.sleep(1)
    manifest = {"schema": "transcif-cn5-energy-charts-2026-holdout-r1",
        "retrieved_utc": datetime.now(timezone.utc).isoformat(),
        "endpoint": "https://api.energy-charts.info/v2/public_power",
        "period": {"start": "2026-01-01", "end_exclusive": "2026-11-01"},
        "license": "CC BY 4.0; attribute Energy-Charts.info",
        "metadata_preflight": str(PREFLIGHT.relative_to(ROOT)),
        "countries": rows}
    (DATA / "acquisition_manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
    print(f"wrote {DATA / 'acquisition_manifest.json'}", flush=True)


if __name__ == "__main__":
    main()
