# Strict CN5 nine-country capacity r1

Run from repository root:

```bash
python scripts/run_energy_charts_nine_country_cn5_capacity.py \
  --capacity-csv data/eurostat-capacity-energy-charts-r2/capacity-nine-country.csv \
  --capacity-manifest data/eurostat-capacity-energy-charts-r2/capacity-nine-country.manifest.json \
  --output results/energy-charts-nine-country-cn5-capacity-r1
```

The runner adds lagged annual `CF/TOTAL` capacity to the strict five-group monthly input contract. Capacity is a separately reported budget and uses current Eurostat revised values as a lag simulation. All point forecasts and calibration radii are hashed before scoring.

Primary artifacts under `results/energy-charts-nine-country-cn5-capacity-r1/`: `macro_metrics.csv`, `country_metrics.csv`, `monthly_metrics.csv`, `capacity_increment_pair.csv`, `prediction_seal.json`, and the nine sealed prediction files.

SHA-256:

| File | SHA-256 |
|---|---|
| `startup_hashes.json` | `a522884f08841a196828f29092dec89acb3cd278f0b7bdcf697d62ec5fad6bf7` |
| `input_audit.json` | `d9bff82e170abab186e1f6e87801c0e6f8bbf965fb03faa9588dcda2f8bd2e78` |
| `prediction_seal.json` | `d2bfbb936464b2bed3b4533b772fb28eba9e7692e486130c4c158187dfa19bb5` |
| `monthly_metrics.csv` | `174ab2092a6161e28ff79ee77bb32b420573748f61a822b2c079dee70c5b9730` |
| `country_metrics.csv` | `bc7d83505dc48c1a88f0b67b5e24e293f7e6f633e7dfb68e4158f920e7cdc34f` |
| `macro_metrics.csv` | `a56880a5f05b504c85cee6f4c0de6072f47f58f61b9c10882ca8869f60f19c92` |
| `capacity_increment_pair.csv` | `41e16543d85344ca43c63860db951266d61c5afc56871d2e24ca6b3ffed0baf9` |
| `status.json` | `fb0573aa552586968e491071ea69af4fee7e22a42f32378fb9b578c4f7f20a4b` |
