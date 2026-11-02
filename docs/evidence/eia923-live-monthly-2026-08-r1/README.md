# EIA-923 live monthly release audit, August 2026 vintage

The archive is the official EIA-923 monthly file labeled August 2026. The EIA release page recorded an October 23, 2026 release date for August data, and listed the next September-data release for the end of November 2026. The downloaded archive has `Last-Modified: Thu, 22 Oct 2026 13:06:37 GMT`, ETag `"802c8e2b2662dd1:0"`, and SHA-256 `50ec141c118547dc0219356171476319e45b5ea382446bda617752a5635e4e91`.

- `f923_2026.zip`: unmodified official monthly ZIP, 15,657,353 bytes.
- `eia923_release_page.html`: captured official release page.
- `eia923_http_headers.txt`: HTTP headers captured at download.
- `results/ba_month_alignment.csv`: all 56 BA-month candidates with row/hour coverage and pair metrics.
- `results/by_ba_coverage.csv`: per-BA completeness counts.
- `results/summary.json`: protocol/code/input hashes and summary metrics.
- `reproduction/`: independent rerun; both CSV outputs and summary JSON match `results/` byte-for-byte.
- `sha256_manifest.json`: hashes for the original release artifacts, EIA-930 inputs, code/protocol, and both result copies.
- Runner: `scripts/audit_live_eia923_ba_vintage.py`.
- Protocol: `docs/LIVE_EIA923_BA_VINTAGE_AUDIT_PROTOCOL_20261101.md`.

The runner joins Page 1 plant/fuel generation to Page 6 plant-month BA mappings. Missing EIA-923 plant/fuel values are never filled with zero. EIA-930 inputs are the current Jan–Jun and Jul–Dec 2026 source files; they are not historical release snapshots matched to the October 23 EIA-923 publication. Only 18/56 BA-months were complete in both inputs, so this is a data-availability/alignment audit, not a CIF model result.

Reproduce with:

```bash
PYTHONPATH=src python scripts/audit_live_eia923_ba_vintage.py \
  --eia923-zip docs/evidence/eia923-live-monthly-2026-08-r1/f923_2026.zip \
  --eia930 data/native-eia-2026/EIA930_BALANCE_2026_Jan_Jun.csv \
           data/native-eia-2026/EIA930_BALANCE_2026_Jul_Dec.csv \
  --output docs/evidence/eia923-live-monthly-2026-08-r1/reproduction
```
