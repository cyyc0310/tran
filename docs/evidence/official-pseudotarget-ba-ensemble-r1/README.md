# Official consumption pseudo-target BA expert ensemble evidence

Reproduce from repository root:

```bash
PYTHONPATH=src:scripts python scripts/run_official_pseudotarget_ba_ensemble.py \
  --output results/official-pseudotarget-ba-ensemble-r1
```

This was a development replay on the previously viewed GB 2024 target. Eight target-label filenames were already present at startup; the run recorded them without reading their contents before the 8,784-hour prediction was sealed. Target scoring occurred after sealing. The selected pseudo-target temperature was `inf`, making gated and uniform predictions identical.

`../../../results/official-pseudotarget-ba-ensemble-r1/` contains predictions, source pseudo-target scores, monthly metrics, calibration, input audit, preseal audit, and status. The startup manifest records 10 code/protocol hashes and 25 source-input hashes. The prediction seal is `09c5282a91e2f53c1bcf70451cb774f67db8c9fb273aeb201f3158eeabc73bb7`; the official GB target-label hash is recorded separately in `target_scoring_hashes.json`.

Interpretation and limits: [frozen protocol](../../OFFICIAL_PSEUDOTARGET_BA_ENSEMBLE_PROTOCOL_20261101.md), [results](../../OFFICIAL_PSEUDOTARGET_BA_ENSEMBLE_RESULTS_20261101.md). This is consumption-side official intensity, which includes each data provider's consumption allocation definition. It is not evidence for local generation-side fuel response, nor a new-target confirmation.
