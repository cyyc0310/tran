#!/usr/bin/env python3
"""Fetch and audit a small first-party NBS provincial monthly generation panel."""
from __future__ import annotations

import csv
import argparse
import hashlib
import json
import subprocess
import time
from datetime import datetime, timezone
from pathlib import Path
from urllib.parse import urlencode


ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "results/v391-nbs-official-provincial-snapshot-r1"
BASE = "https://data.stats.gov.cn/dg/website/publicrelease/web/external"
START, STOP = "202509MM", "202609MM"
REGIONS = {
    "Shanghai": ("上海市", "310000000000"),
    "Sichuan": ("四川省", "510000000000"),
    "Jilin": ("吉林省", "220000000000"),
    "Fujian": ("福建省", "350000000000"),
    "Qinghai": ("青海省", "630000000000"),
}
TECH_LABELS = {
    "total": "发电量",
    "thermal": "火力发电量",
    "hydro": "水力发电量",
    "nuclear": "核能发电量",
    "wind": "风力发电量",
    "solar": "太阳能发电量",
}
FS_MONTH_ROOT = "f4c6cd795fea436c807163397dd36b98"
COOKIE_JAR = Path("/tmp/transcif-v391-nbs-session.cookie")
USER_AGENT = "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 Chrome/131.0.0.0 Safari/537.36"
PAGE_REFERER = "https://data.stats.gov.cn/dg/website/page.html#/pc/national/fsMonthData"


def bootstrap_session() -> None:
    # The portal's front door can return a WAF 302, but it also establishes the
    # session cookie used by the read-only JSON endpoints. Do not follow the WAF
    # redirect loop; let the API response determine whether the session works.
    subprocess.run(
        ["curl", "--max-time", "20", "-sS", "-A", USER_AGENT, "-c", str(COOKIE_JAR), "-b", str(COOKIE_JAR),
         "-e", PAGE_REFERER, f"{BASE.rsplit('/dg/', 1)[0]}/dg/website/page.html"],
        capture_output=True,
        check=True,
    )


def request_json(path: str, *, params: dict | None = None, payload: dict | None = None) -> tuple[dict, bytes]:
    url = f"{BASE}/{path}"
    if params:
        url += "?" + urlencode(params)
    command = ["curl", "--max-time", "30", "-sS", "-A", USER_AGENT, "-b", str(COOKIE_JAR), "-c", str(COOKIE_JAR),
               "-e", PAGE_REFERER, "-H", "Origin: https://data.stats.gov.cn", "-w", "\n__V391_HTTP__%{http_code}"]
    if payload is not None:
        command.extend(["-H", "Content-Type: application/json", "--data-binary", "@-", url])
    else:
        command.append(url)
    last_error = None
    for attempt in range(6):
        try:
            result = subprocess.run(command, input=json.dumps(payload, ensure_ascii=False).encode() if payload is not None else None,
                                    capture_output=True, check=True)
            marker = b"\n__V391_HTTP__"
            body, status_bytes = result.stdout.rsplit(marker, 1)
            status = int(status_bytes.strip())
            if status in (301, 302, 303, 307, 308, 429, 500, 502, 503, 504):
                raise RuntimeError(f"transient HTTP {status}")
            response = json.loads(body)
            if response.get("success") is not True:
                raise RuntimeError(f"API state {response.get('state')}: {response.get('message')}")
            time.sleep(0.3)
            return response, body
        except Exception as exc:
            last_error = exc
            time.sleep(min(2 ** attempt, 12))
    raise RuntimeError(f"NBS API request failed after retrying {path}: {last_error}")


def children(pid: str) -> list[dict]:
    response, _ = request_json("new/queryIndexTreeAsync", params={"pid": pid, "code": 4})
    return response.get("data", [])


def find_named(nodes: list[dict], name: str) -> dict:
    matches = [node for node in nodes if node.get("name") == name]
    if len(matches) != 1:
        raise RuntimeError(f"Expected one catalog node {name!r}, found {len(matches)}")
    return matches[0]


def find_catalogs() -> tuple[dict, dict]:
    fs_month = find_named(children(""), "分省月度数据")
    energy = find_named(children(fs_month["_id"]), "能源")
    products = find_named(children(energy["_id"]), "能源主要产品产量")
    leaves = children(products["_id"])
    catalog_map = {}
    metadata = {}
    for tech, label in TECH_LABELS.items():
        node = find_named(leaves, label)
        if not node.get("isLeaf"):
            raise RuntimeError(f"Expected leaf catalog for {label}")
        catalog_map[tech] = node
        response, raw = request_json("new/queryIndicatorsByCid", params={"cid": node["_id"]})
        metadata[tech] = {"catalog": node, "indicators": response.get("data", {}).get("list", [])}
        metadata[tech]["metadata_response_sha256"] = hashlib.sha256(raw).hexdigest()
    return catalog_map, metadata


def month_codes(start: str, stop: str) -> list[str]:
    if len(start) != 8 or len(stop) != 8 or not start.endswith("MM") or not stop.endswith("MM"):
        raise ValueError("period codes must look like YYYYMMmm, for example 202601MM")
    start_year, start_month = int(start[:4]), int(start[4:6])
    stop_year, stop_month = int(stop[:4]), int(stop[4:6])
    if start_month not in range(1, 13) or stop_month not in range(1, 13):
        raise ValueError("month code must use YYYYMMMM format")
    values = []
    year, month = start_year, start_month
    while (year, month) <= (stop_year, stop_month):
        values.append(f"{year}{month:02d}MM")
        month += 1
        if month == 13:
            year, month = year + 1, 1
    return values


def main(start: str = START, stop: str = STOP, out: Path = OUT) -> None:
    global OUT
    OUT = out
    months = month_codes(start, stop)
    OUT.mkdir(parents=True, exist_ok=False)
    raw_dir = OUT / "raw"
    raw_dir.mkdir()
    bootstrap_session()
    catalog_map, metadata = find_catalogs()
    current_indicators = {}
    for tech, item in metadata.items():
        matches = [i for i in item["indicators"] if "当期值" in i.get("i_showname", "")]
        if len(matches) != 1:
            raise RuntimeError(f"Expected one current-value indicator for {tech}, got {len(matches)}")
        current_indicators[tech] = matches[0]

    manifest = {
        "source": "National Bureau of Statistics China data portal, first-party JSON API",
        "entry_url": "https://data.stats.gov.cn/dg/website/page.html#/pc/national/fsMonthData",
        "api_base": BASE,
        "retrieved_at_utc": datetime.now(timezone.utc).isoformat(),
        "periods_requested": [start, stop],
        "requested_month_codes": months,
        "region_codes": {key: {"name": value[0], "code": value[1]} for key, value in REGIONS.items()},
        "selection_rule": "one official current-month indicator for each total/thermal/hydro/nuclear/wind/solar catalog; cumulative, growth and cumulative growth indicators excluded",
        "catalogs_and_indicator_metadata": metadata,
        "requests": [],
    }
    observations = []
    for tech, indicator in current_indicators.items():
        cid = catalog_map[tech]["_id"]
        for province, (display_name, region_code) in REGIONS.items():
            payload = {
                "cid": cid,
                "indicatorIds": [indicator["_id"]],
                "das": [{"text": display_name, "value": region_code}],
                "dts": [f"{start}-{stop}"],
                "showType": "1",
                "rootId": FS_MONTH_ROOT,
            }
            response, raw = request_json("stream/esData", payload=payload)
            relpath = Path("raw") / tech / f"{province}.json"
            raw_path = OUT / relpath
            raw_path.parent.mkdir(parents=True, exist_ok=True)
            raw_path.write_bytes(raw)
            manifest["requests"].append({
                "technology": tech,
                "province": province,
                "url": f"{BASE}/stream/esData",
                "payload": payload,
                "response_path": str(relpath),
                "response_sha256": hashlib.sha256(raw).hexdigest(),
                "response_state": response.get("state"),
                "response_message": response.get("message"),
                "period_rows": len(response.get("data", [])),
            })
            for period in response.get("data", []):
                code = period.get("code", "")
                if not code.endswith("MM"):
                    raise RuntimeError(f"Unexpected monthly period code {code!r}")
                year, month = int(code[:4]), int(code[4:6])
                values = period.get("values", [])
                if len(values) != 1:
                    raise RuntimeError(f"Expected one value for {province}/{tech}/{code}, got {len(values)}")
                value = values[0]
                if value.get("da") != region_code or value.get("_id") != indicator["_id"]:
                    raise RuntimeError(f"Region/indicator mismatch in {province}/{tech}/{code}")
                raw_value = value.get("value", "")
                observations.append({
                    "province": province,
                    "province_name": display_name,
                    "region_code": region_code,
                    "technology": tech,
                    "year": year,
                    "month": month,
                    "period_code": code,
                    "value_100m_kwh": float(raw_value) if raw_value not in (None, "") else "",
                    "unit": value.get("du_name", ""),
                    "indicator_name": value.get("i_showname", ""),
                    "observation_kind": "current_value",
                    "source_response": str(relpath),
                })

    expected_rows = len(REGIONS) * len(TECH_LABELS) * len(months)
    if len(observations) != expected_rows:
        raise RuntimeError(f"Expected {expected_rows} province-tech-month rows, got {len(observations)}")
    fields = list(observations[0])
    with (OUT / "official_current_values.csv").open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields)
        writer.writeheader()
        writer.writerows(observations)

    by_key = {(r["province"], r["year"], r["month"], r["technology"]): r["value_100m_kwh"] for r in observations}
    mix_rows = []
    for province in REGIONS:
        for code in months:
            year, month = int(code[:4]), int(code[4:6])
            values = {tech: by_key[(province, year, month, tech)] for tech in TECH_LABELS}
            five_known = [values[t] for t in ("thermal", "hydro", "nuclear", "wind", "solar")]
            has_all = all(v != "" for v in five_known) and values["total"] != ""
            listed_sum = sum(five_known) if has_all else ""
            total = values["total"]
            row = {"province": province, "year": year, "month": month, "period_code": code,
                   "total_100m_kwh": total, "listed_five_tech_sum_100m_kwh": listed_sum,
                   "total_minus_five_tech_100m_kwh": total - listed_sum if has_all else "",
                   "has_all_current_values": int(has_all)}
            for tech in TECH_LABELS:
                row[f"{tech}_100m_kwh"] = values[tech]
                row[f"{tech}_share_of_listed_five"] = values[tech] / listed_sum if has_all and tech != "total" and listed_sum else (1.0 if tech == "total" and has_all and listed_sum else "")
            mix_rows.append(row)
    mix_fields = list(mix_rows[0])
    with (OUT / "provincial_mix_and_closure.csv").open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=mix_fields)
        writer.writeheader()
        writer.writerows(mix_rows)

    manifest["row_count"] = len(observations)
    manifest["provinces"] = list(REGIONS)
    manifest["technologies"] = list(TECH_LABELS)
    manifest["expected_rows"] = expected_rows
    manifest["missing_current_values"] = sum(r["value_100m_kwh"] == "" for r in observations)
    manifest["months_with_full_six_series"] = sum(r["has_all_current_values"] for r in mix_rows)
    manifest["note"] = "Current live NBS snapshot, not a historical as-published vintage. 2026 Jan/Feb may be unavailable; blanks are retained and not imputed from cumulative values. NBS response is the source of record; OREO is used only in a separate comparison."
    (OUT / "manifest.json").write_text(json.dumps(manifest, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")

    full = [r for r in mix_rows if r["has_all_current_values"]]
    closure = [abs(r["total_minus_five_tech_100m_kwh"]) for r in full]
    print(json.dumps({
        "output": str(OUT),
        "provinces": list(REGIONS),
        "technology_indicators": {k: v["i_showname"] for k, v in current_indicators.items()},
        "expected_rows": expected_rows,
        "observations": len(observations),
        "missing_current_values": manifest["missing_current_values"],
        "province_months_with_all_six_series": len(full),
        "max_abs_five_tech_total_closure_100m_kwh": max(closure) if closure else None,
        "retrieved_at_utc": manifest["retrieved_at_utc"],
    }, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--start", default=START, help="first YYYYMMMM month code")
    parser.add_argument("--stop", default=STOP, help="last YYYYMMMM month code")
    parser.add_argument("--output", type=Path, default=OUT, help="new output directory; existing directories are refused")
    args = parser.parse_args()
    main(args.start, args.stop, args.output)
