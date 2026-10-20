#!/usr/bin/env python
"""Archive national NESO actual/forecast fields separately, retaining raw snapshots."""

import argparse
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import sys
import time
import urllib.request

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from transcif.data.asof import utc
from transcif.data.neso import national_intervals, hourly_national


def collect(start, end, output):
    start, end = utc([start, end])
    output = Path(output)
    if start >= end or start != start.floor("h") or end != end.floor("h") or output.exists():
        raise ValueError("ordered whole-hour bounds and fresh output directory required")
    (output / "raw").mkdir(parents=True)
    cursor, snapshots, frames = start, [], []
    while cursor < end:
        stop = min(cursor + pd.Timedelta(days=7), end)
        url = "https://api.carbonintensity.org.uk/intensity/" + cursor.strftime("%Y-%m-%dT%H:%MZ") + "/" + stop.strftime("%Y-%m-%dT%H:%MZ")
        for attempt in range(3):
            try:
                request = urllib.request.Request(url, headers={"User-Agent": "TransCIF-research-data-collector/0.2"})
                with urllib.request.urlopen(request, timeout=30) as response:
                    raw, server_date = response.read(), response.headers.get("Date")
                break
            except OSError:
                if attempt == 2:
                    raise
                time.sleep(2 * (attempt + 1))
        chunk = f"raw/{len(snapshots):04d}.json"
        (output / chunk).write_bytes(raw)
        frames.append(national_intervals(json.loads(raw), cursor, stop))
        snapshots.append({"file": chunk, "url": url, "sha256": hashlib.sha256(raw).hexdigest(),
            "http_server_date": server_date, "downloaded_at": datetime.now(timezone.utc).isoformat(),
            "original_publication_time": None})
        (output / "snapshots.json").write_text(json.dumps(snapshots, indent=2) + "\n")
        cursor = stop
        if len(snapshots) % 10 == 0:
            print(f"Archived {len(snapshots)} weekly snapshots through {cursor}", flush=True)
        time.sleep(.2)
    intervals = pd.concat(frames, ignore_index=True)
    hourly = hourly_national(intervals, start, end)
    intervals.to_csv(output / "native_intervals.csv", index=False)
    hourly.to_csv(output / "hourly.csv", index=False)
    report = {"provider": "NESO Carbon Intensity API national endpoint", "license": "CC BY 4.0",
        "accounting": "provider national estimated intensity; not certified as production_direct_co2",
        "label_kind": "provider estimated actual; forecast fields separate, no fallback",
        "eligible_for_issued_public": False, "vintage": "current downloaded revision; original publication unknown",
        "snapshots": len(snapshots), "half_hour_intervals": len(intervals), "hours": len(hourly),
        "hours_without_complete_actual": int(hourly.provider_actual_g_per_kwh.isna().sum()),
        "missing_evidence": ["original label/forecast publication and revision vintages",
            "compatible accounting, all fuel MW supervision, public configuration and issued input forecasts"]}
    (output / "metadata.json").write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(report), flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--start", required=True)
    parser.add_argument("--end", required=True)
    parser.add_argument("--output", type=Path, required=True)
    collect(**vars(parser.parse_args()))


if __name__ == "__main__":
    main()
