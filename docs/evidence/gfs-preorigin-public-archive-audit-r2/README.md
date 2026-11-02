# GFS pre-origin public archive audit

- Frozen audit protocol: `docs/GFS_PREORIGIN_PUBLIC_ARCHIVE_AUDIT_PROTOCOL_20261101.md`.
- Auditor: `scripts/audit_expanded_gfs_preorigin_public_archive.py`.
- Source archive: `results/weather-transfer-expanded-r1/` (24 sealed forecast windows).
- Final audit output: `results/gfs-preorigin-public-archive-audit-r2/`.
- r2 validates all 24 completion-list window hashes, 1,176 metadata-file hashes against their window manifests, forecast origins, lead hours 24–72, public NOAA S3 URL prefix, ETags, and object `Last-Modified` times.
- Result: all 1,176 objects were timestamped at least 19.867 hours before their forecast origins. No target labels, predictions, or metric files were read.
- Interpretation is limited to pre-origin presence in the NOAA public S3 archive. The audit does not establish per-client access logs, historical ACL state, or a real-time availability SLA.
- `r1` is retained as a preliminary calculation; cite `r2` as the hash-checked final audit.

Summary SHA-256: `9f41dac54adcfd3053b4ae9019272698d247f03df29595f4a45d64c4e4868cee`.
