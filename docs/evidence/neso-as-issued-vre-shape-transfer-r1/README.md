# NESO签发风光形状迁移：封存预测与评分证据

本目录保留GB 2024五类月量强重组及目标日前一日已签发嵌入式风光预测形状的配对结果。目标固定因子发电代理标签已在封存预测后用于评分，且此前已用于项目开发；不得称作盲测、新国家确认或官方小时CIF。

## 文件

- `prediction_seal.json`：冻结协议、源模型、滞后月量合同、NESO特征和预测文件的SHA-256。
- `predictions_common_support.csv`：8,688个共同小时的原始点预测、签发风光值和发布时间字段。预测已按协议封存后才评分。
- `monthly_metrics.csv`：各方法逐月MAE与使用同一源区半径的经验覆盖。
- `scored_summary.json`：主指标、配对日bootstrap及oracle诊断摘要。
- `posthoc_component_ablation.json`：预测封存后单独替换风/光预测形状的机制诊断。

## 哈希

| 文件 | SHA-256 |
|---|---|
| `prediction_seal.json` | `9e47c8800640353241e972aac19c1b14b95c2e4e73f2497cb1d11c00c177ed49` |
| `predictions_common_support.csv` | `1d2443626fcd7c0bfc84a2f66e706939279ab79a6d64c1c2aea34d3c9a37bd54` |
| `monthly_metrics.csv` | `67f170f0e0e72d2faf3a639f0f8f40b31d5b157bf31eb249466f345f1e2e4cc4` |
| `scored_summary.json` | `6a46ea25e9b9c1f77c6bef961ddb219aaed8cd988e09f77953eb0081981ca3a0` |
| `posthoc_component_ablation.json` | `fc2766475a713a0cf645a7657379847205028efa6d60e1e112b36ad6980a27bd` |

封存引用哈希：冻结协议 `d643defa643a6832ff2ea376f7cbd2eb231afd2077813f70267733cdcdbdd133`；模型 `src/transcif/research/cross_country.py` `92e4d0109459ac759514401569a7b5e269785e141106ab3afc4339db47991c63`；月量合同 `docs/CROSS_COUNTRY_PILOT_PROTOCOL.md` `a325d1806eb129ed43608af2f5dc4fa0f5335601ee766488e786d534c563ce3d`；NESO特征 `forecast_origin_features_2024.csv` `15893c1837395d51317bf8970f868624e8083646a545c6d562f95617a3eced06`。NESO原始档案URL、大小和SHA见[特征审计包](../neso-embedded-forecast-vintage-r1/manifest.json)。

当前证据足以重评分已封存预测，但未保存生成这些预测的独立runner；不可声称从原始输入到结果可一条命令完整复现。
