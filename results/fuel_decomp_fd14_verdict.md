# Phase FD-14 Verdict — NEMED DUID 数据工程(AU 逐燃料遥测)+ 两个深层 bug 修复

日期:2026-08-15 · 交付:`scripts/data/extract_au_fuel_breakdown.py` + `fuel_shares_au.json` + 4 区逐燃料 CSV · 消融:8×2(`fuel_decomp_eval_au_fuel.json` vs `fuel_decomp_eval_datatracks.json`)

## 一句话结论

**PASS(经两次调试)**:AU 四区获得逐燃料遥测(提取器验证:可再生份额与真值 rs 相关 **1.000**,物理重建 MAE 6-14)。接入暴露并修复了**两个深层 bug**——①AU 时间线为 NEM 本地时而天文/天气按 UTC 假设(solar↔clearsky 相关 **−0.697**,修复后 **+0.976**);②DST 夏令时 +1h(VIC/SA/NSW 测试期恰在夏令时季)。加**确定性风电路由**(wind ≥ 0.25 走聚合路径,零学习参数)后:**I_cfg 快速集中位 71.57→67.77(win 75%,p=0.074)**,I_0 Spearman 0.277→**0.333**。

## 数据工程细节

**提取器**(AEMO 注册表被 Cloudflare 拦,改帧内自分类):
- 名字连接(nemed bundled duid_mapping + existing_gen_data_summary,主源)
- 强度带:>0.8 煤、(0, 0.8] 气(CIF 安全:AU 煤 ≥0.85,气 ≤0.7)
- 夜间测试:零强度且夜间能量 <2% → solar(发现并修复 Time 已是本地时的 +10 错移)
- 默认 wind(可再生 EF 全 0,内混淆 CIF 中性;AU 水电 ~6% 并入)

**最终年度份额**:QLD1 coal 0.742/solar 0.102/wind 0.073;NSW1 coal 0.700/solar 0.096;VIC1 coal 0.634(褐煤)/wind 0.320;SA1 wind 0.603/gas 0.296/solar 0.081。

## 调试历程(研究记录)

| 步骤 | 现象 | 根因 | 修复 |
|---|---|---|---|
| 初版接入 | VIC1 +61、SA1 +38,solar↔clearsky **−0.697** | AU 数据本地时,天文按 UTC 算(错 10h)、天气错位 10h | `attach_fuel_and_exog` AU 时间线转 UTC |
| 太阳分类 | solar 份额偏低 ~半 | 夜间测试误 +10 时移(太阳能全进 wind) | Time 本就是本地时,去掉平移 |
| 修复后 | 煤主区改善但 SA1/VIC1 仍 +39/+53 | 风头依赖 wcf,AU 风场远离质心,代表性差 | **确定性路由**:wind≥0.25 → 聚合路径 |
| VIC1 残差 | rho −0.30 | DST(VIC/SA/NSW 测试期 +1h) | 逐时戳 DST 修正(VIC1 solar↔clearsky 0.837) |

## 最终数字(8×2,vs FD-13 基线)

| 区域 | I_cfg | 备注 |
|---|---|---|
| NSW1 | 84.6→**73.8**(−10.8) | 煤主,燃料路径 |
| SA1 | 78.7→**71.9**(−6.8) | 路由回聚合路径 |
| UK_02 | 27.0→22.8(−4.2) | 时区修复外溢收益 |
| QLD1/PJM/BPAT | −0.8/−0.5/−0.7 | — |
| UK_08/VIC1 | +2.1/+2.6 | 残差(记录) |
| **中位** | **71.57→67.77(win 75%,p=0.074)** | I_0 Spearman 0.277→0.333 |

## 决策

1. AU 燃料遥测 + 时区/DST 修复 + 风电路由全部保留为默认
2. 全量 145 对正式运行(`fuel_decomp_eval_full_final.json`)后台进行
3. 遗留:VIC1 rho 负值(风场代表性/DST 边界),后续可加风场坐标加权天气
