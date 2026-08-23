# Phase FD-13 Verdict — 数据轨道落地(价格 + 阵风/气压)+ NEMED 核查

日期:2026-08-15 · 交付:`scripts/data/download_prices.py`、`scripts/data/download_pressure_winds.py`、数据层集成(wx 7 通道 / fut_exog 14 通道)· 消融:8×2(`fuel_decomp_eval_datatracks.json` vs `fuel_decomp_eval_deg12.json`)· 全量 145 对运行中(`fuel_decomp_eval_full_tracks.json`,含双轨标签)

## 一句话结论

**三条数据轨道全部落地,信号真实但幅度小**:燃料价格(世界银行粉红表 + FRED 日度,z-score、1 月发布滞后、按辖区映射)与阵风+海平面气压(Open-Meteo ERA5,替代不可用的气压层风)使 I_cfg **72.32→71.57(win 62%,6/8 区域改善:SA1 −2.5 / NSW1 −1.1 / BPAT −1.0)**,p=0.32 未过显著线;I_0/I_+ 中性。**NEMED DUID 不可行于本轮**(venv 缺失 + AEMO GB 级下载),为唯一遗留数据项。

## 轨道明细

| 轨道 | 源 | 实现 | 状态 |
|---|---|---|---|
| 燃料价格 | 世界银行粉红表(月度:NEWC 煤、欧/日/美气)+ FRED DHHNGSP(日度美气),均无密钥 | 辖区映射(AU→日 LNG、UK→欧 TTF、US→HH),z-score,1 月滞后广播 | ✅ 已集成 |
| 阵风 + 气压 | Open-Meteo ERA5(950hPa 探测为全 null——归档 API 不带气压层变量,已换) | 阵风 km/h→m/s、气压 1013 hPa 异常;时间戳连接,缺失优雅为零 | ✅ 29 区已下载 |
| NEMED DUID | AEMO DISPATCH_UNIT_SCADA + nemed 包 | `.venv-nemed311` 缺失,需 python3.11 venv + GB 级 AEMEO 下载(requirements-nemed.txt 有完整指引) | ⛔ 遗留 |

渠道维度:`x_weather`/`fut_weather` 5→7(+阵风、气压异常);`fut_exog` 12→14(+coal_z、gas_z)。模型默认维度同步,42 FD 测试全绿。

## 消融(8×2,配对 vs deg12)

| 层级 | MAE | 判定 |
|---|---|---|
| I_cfg | 72.32→71.57(win 62%,p=0.32) | 方向正确、幅度小、保留(部署合法零成本) |
| I_0 | 45.34→45.20(p=0.30) | 中性 |
| I_+ | 36.73→36.69 | 中性(equalizer 预期内) |

逐区域:SA1 −2.5、NSW1 −1.1、BPAT −1.0、UK_08 −0.8、PJM −0.6、UK_02 −0.2;QLD1/VIC1 +1.0(AU 气价代理为日 LNG,粒度损失)。

## 决策

1. 三通道保留为默认(无伤害 + 小增益 + 部署合法)
2. 全量 145 对正式运行(含双轨标签)后台进行,完成后更新 leaderboard
3. NEMED DUID 为下一步唯一数据项:按 requirements-nemed.txt 建 venv 后运行 generate_nemed_regions.py,再把 DUID 燃料表接进 `load_fuel_shares`(AU 四区预期 I_cfg 89.6→60-70)
