# CN5燃料组成头与火电因子头嵌套混合证据

主运行：`results/cn5-fuel-factor-head-blend-r1/`。最近邻条件门控敏感性：`results/cn5-fuel-factor-head-profile-gate-r2/`。

## 复现

```bash
python scripts/run_cn5_fuel_factor_head_blend.py \
  --output results/cn5-fuel-factor-head-blend-r1

python scripts/analyze_cn5_fuel_factor_head_profile_gate.py \
  --output results/cn5-fuel-factor-head-profile-gate-r2
```

主实验代码、冻结协议、源输入清单哈希保存在`results/cn5-fuel-factor-head-blend-r1/startup_hashes.json`。36组双重排除源模型为72个伪目标各自生成H1/H2预测；这些文件哈希见`source_pseudotarget_seal.json`（144/144复核通过）。九份2024外层预测和校准哈希见`outer_prediction_seal.json`（9/9复核通过）。逐月/逐国/宏指标和alpha选择见`monthly_metrics.csv`、`country_metrics.csv`、`macro_metrics.csv`及各国`*_selection_calibration.json`。

条件门控复核的代码、协议和输入哈希在`results/cn5-fuel-factor-head-profile-gate-r2/startup_hashes.json`；输出包括`source_nested_gate_validation.csv`、`outer_gate_routing.csv`、`outer_gate_country_metrics.csv`和`summary.json`。第一次分析运行因SK早期六个月历史存在缺口而停止，未生成结果；r2按可见历史行数计算目标签名并完成。只有r2作为有效敏感性结果。

## 解释边界

全部小时标签是Energy-Charts燃料量按项目固定因子重建的发电侧代理；t−2…t−7月量是当前修订小时序列回算，不是历史as-issued统计。九个国家都曾用于研究开发。主结果用于排除全局头间混合算法增量；最近邻路由只作已看目标上的探索，不构成外部确认。详见[主结果](../../CN5_FUEL_FACTOR_HEAD_BLEND_RESULTS_20261102.md)、[路由结果](../../CN5_FUEL_FACTOR_HEAD_PROFILE_GATE_RESULTS_20261102.md)及[冻结协议](../../CN5_FUEL_FACTOR_HEAD_BLEND_PROTOCOL_20261102.md)。
