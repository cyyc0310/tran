# FD-50m/n Verdict：破 20 三外部数据通道全部定案

日期：2026-09-11。承接 FD-50k/l 终局（I_lag 层 20 数学不可达，形状地板 23.14），用户下令
"都可以做一下 你都试试吧"后的三条外部数据通道全部收口。结论先行：三条通道无一能把
29 区 J5 late MAE（30.76-31.01）推向 20；I_lag 层最优解仍是 J5，破 20 需更换问题设定。

## 通道 1：NWP 集合预报（FD-50m）— 证负

对 FD-47 四臂（era5 / gfs / icon / ensemble，Open-Meteo 业务预报归档 vs ERA5 再分析代理）
各叠 J5 配方（late 半段，诚实口径）：

| 臂 | base I_cfg | +R | +J5 |
|---|---|---|---|
| era5（再分析代理） | 43.58 | 40.32 | 31.01 |
| gfs | 50.23 | 44.61 | 34.15 |
| icon | 50.01 | 45.24 | 33.86 |
| ensemble | 49.15 | 45.21 | 33.66 |

混合臂（NWP×ERA5 early 拟合权重）：gfs 30.75 / icon 30.64 / ensemble 30.66 vs 纯 ERA5 31.01，
残余 <0.4；逐区 early 选臂（7 候选）30.73 vs 31.01。残余增益全部被 early 半段吸收，late 无实增益。

结论：恢复完整天气资产后（9/10 FULL 质心 farmblend），Open-Meteo 业务预报归档相对
ERA5 再分析在所有层级全面劣化。FD-47 原版"NWP +12 收益"结论基于 9/9 降级资产
（9 区天气全零），已被 FULL 资产重跑推翻——零天气基线上任何天气都是增益，真实对照下
业务预报的额外误差超过其前瞻信息价值。证负定案。

## 通道 2：多年数据（FD-50n）— 证负

数据侧（全部下载完成，诚实标注）：
- UK CI API 2022/2024：18 区 × 2 年负荷 + 燃料全量可得
- EIA-930 2022/2024：8 BA × 2 年负荷 + 燃料全量可得
- AU 2022：NEMED 5 区负荷 + 4 区燃料全年完整
- AU 2024：NEMWEB DISPATCH_UNIT_SCADA 仅保留 24 个月归档，9 月块起 404，
  两条链路（NEMED 负荷 + AU 燃料提取）均 NoDataToReturn——外部数据边界，不可得
- 天气 2022/2024：29 目标区质心 2022 31/31、2024 31/31 全量补齐（UK_12 超时重试成功）

协议（诚实双保证，fd50n_multiyear.py）：
1. 仅 SOURCE 区拼接 2022+2023(2024)（multi_year=True）；TARGET 区固定 multi_year=False，
   测试窗口保持官方 2023 holdout，目标区多年真值零加载（零遥测不破）。
2. monkey-patch dk.DUMP_DIR → results/dunkelflaute_my/，官方 base dump 不被覆盖。
   冒烟测试（UK_02，60 ep）验证：npz 落位正确、72/72 窗口与 origin_hours 逐位一致。

四臂对比（late 半段，seed 0，FD-41 官方口径训练）：

| 臂 | 29 区均值 |
|---|---|
| base I_cfg | 43.58 |
| multi-year I_cfg | 43.72 |
| base + J5 | 31.01 |
| multi-year + J5 | 31.03 |

逐区 Δ(my−base)：改善最大 US_NYIS −2.6、UK_16 −1.4；劣化最大 UK_05_Yorkshire +4.9、
VIC1 +4.2、UK_03 +1.9；其余 ±1 以内。均值 +0.14（base 层）/ +0.02（J5 层），正负互抵。

结论：源域翻倍（多两个季节的跨域燃料-天气动态）对该 zero-shot 路由架构无系统性增益。
训练信号已经饱和——28 源 2023 一年的跨域动态足以支撑路由，瓶颈不在训练数据量而在
目标区非平稳水平漂移（这正是 J5 在线跟踪器已经吃掉的误差源）。证负定案。

## 通道 3：标签工程（FD-28 继承）— 已隐式生效

FD-28 DUID 注册表修正（REGISTRY_OVERRIDES：GUTHEGA/SHGEN/MURRAY→hydro、
W/HOE#1/#2→hydro、WALGRVG1 等 BESS→other）已于 9/10 生效：
- NSW1_fuel_2023_hourly.csv 9/10 11:24 重生成：hydro 0.423%（≈Guthega 121GWh+SHGEN
  109GWh=230GWh，wind 桶 9108GWh 的 2.5%）、wind 16.753%，抽蓄晚峰（17 时 65.8 /
  18 时 96.8 / 19 时 91.1 MW）形态正确。
- 9/10 13:02 的 dunkelflaute 全量 dump（FD-50 系列全部成绩）已继承修正后燃料输入。
- hydro/wind 均 EF=0 → 对 CIF 真值中性，只影响模型输入份额；本通道对 MAE 的贡献
  已隐式包含在 base 43.58 / J5 31.01 中，无独立增量可报。UK API 记账误差
  （forecast vs actual）在 2023 数据内无法分离重算，超出本轮边界。

## 附：J5 5-seed 稳定性（fd50j_5seed_stability）

5-seed 重跑（seed 0-4 × 29 区，FD-41 官方口径）后对每个 dump 各跑 J5：
- 29 区 J5 late 均值 31.07（跨区 std 14.38）；raw I_cfg 43.56（std 18.27）
- 区内 seed std：中位 0.37 / 最大 1.99（UK_09 East Midlands）
- seed 0 单跑 30.76 落在稳定带内，无 seed cherry-pick 风险

J5 配方的增益（43.56→31.07，−12.5）远超 seed 噪声（<2.0），可复现性成立。

## 总结论

| 通道 | 结论 | 关键数字 |
|---|---|---|
| NWP 集合 | 证负 | gfs/icon/ensemble 全层级劣于 era5；混合/选臂残余 <0.4 |
| 多年数据 | 证负 | base +0.14 / J5 +0.02，逐区正负互抵 |
| 标签工程 | 已隐式生效 | FD-28 修正包含在 9/10 数据与全部 FD-50 成绩中 |

三通道穷尽后，I_lag 层可达边界维持 FD-50l 判定：J5 ≈ 31（5-seed 稳定带 31.07±0.4），
20 在当前信息层不可达（水平作弊完美地板 23.14）。破 20 的剩余路径只有更换问题设定：
I_online 层（真实延迟标签接口接入 predict_fuel_windows 后处理）或更长时效的
自回归目标重定义。论文素材不损反增：三条外部通道的证负本身构成 C5 边界链的
"数据通道穷尽"段落——零训练配方 J5 在 NWP/多年/标签三条外部增强全部无效的前提下
依然是最优，进一步凸显在线水平跟踪的不可替代性。

产出文件：
- results/fd50m_nwp_j5.{json,md}（通道 1 四臂 J5）
- results/fd50m_mix_earlyfit.json（通道 1 混合/选臂）
- results/fd50n_multiyear.json + results/dunkelflaute_my/（通道 2 训练 dump 29 区）
- results/fd50n_analysis.{json,md}（通道 2 四臂对比）
- results/fd50j_5seed_stability.{json,md}（5-seed 稳定性）
- scripts/experiments/fd50m_nwp_j5.py / fd50n_multiyear.py / fd50n_analysis.py /
  fd50j_5seed_stability.py / fd50n_smoke.py（全链路可复现）
