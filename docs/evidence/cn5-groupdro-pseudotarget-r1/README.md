# Strict CN5 GroupDRO evidence

- Frozen design: `../../CN5_GROUPDRO_PSEUDOTARGET_PROTOCOL_20261101.md`
- Runner: `code/run_cn5_groupdro_pseudotarget.py`
- Full output: `results/` and `../../../results/cn5-groupdro-pseudotarget-r1/`
- `startup_hashes.json` records the runner, protocol, data manifests and source files before loading the benchmark.
- `prediction_seal.json` records all nine outer-country prediction files and calibration radii before target-label scoring. The runner verifies both input/code hashes and output hashes.
- `artifact_hashes.json` records the archived run artifacts.

This is a development experiment on nine countries already used in the project. Monthly inputs are reconstructed from revised hourly records to emulate a two-month lag, and hourly labels are fixed-factor generation-side proxies. It is not a historical as-issued replay, blind confirmation, or measured-CIF result. GroupDRO is a standard baseline, not a novel TransCIF method.
