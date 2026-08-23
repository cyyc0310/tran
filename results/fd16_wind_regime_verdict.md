# FD-16 verdict:风 regime 通道 + 干旱锚定 —— 诚实负结果(MAE 中性,保留修正)

**日期**:2026-08-16
**动机**:`results/extreme_weather_analysis.md` —— CIF 极端波动集中在风电份额过渡带(wind-lull),风暴日本身反而稳定;风头以"过去一周均值"归一化未来风速,一周干旱时参考值同步塌陷,干旱信号被归一化吸收。
**假设**:给模型显式 wind-regime(24h 滞后均值 + 6h 倾向,纯天气派生、因果、所有信息层级合法)+ 干旱锚定(参考值向年气候回归),能降低 lull 日误差。

## 实现(已合入)

1. `data/fuel.py`:wx 8→10 通道(idx 8 `wind_regime24`,idx 9 `wind_tend6`,因果);fut_exog 15→17(同两通道,聚合头 rs_exog 与 thermal 头可见);`build_fd_windows` 去除硬编码 8。
2. `models/fuel_decomp.py`:n_weather=10 / n_exog=17;`_WX_*` 通道常量;风参考值干旱锚定 `wcf_ref ← (1−0.6·lull)·blend + 0.6·lull·annual`,`lull = σ(8(0.75 − regime/annual))`;8 通道输入向后兼容。
3. `models/hypernet.py`:GENERATED_HEADS 维度同步。
4. **真 bug 修复(独立于 FD-16,保留价值所在)**:`apply_day_ahead_weather_error`(_wn 轨道)此前只重算 fut_exog 的派生通道,而风头直接读取的 `fut_weather[:,:,3]`(wcf)未被降级 —— 天气噪声轨道系统性低估了风速误差的影响;现在 wcf/csi/regime/tend 全部从扰动后的风速/短波自洽重算。

## 评估(三组,全部中性)

| 评估 | 对比 | 结果 |
|---|---|---|
| smoke 8 区 × 2 seed(16 对) | full_final 同对 | I_+ 46.01→45.99(win 9/16, p=0.49);I_cfg 71.86→72.28(p=0.35);I_0 −0.4(p=0.46) |
| 机制诊断 4 风电区(燃料路径锚定) | 同 seed 8 通道旧模型 | lull 三分位:+1.1/−1.0/−0.9/+0.3(噪声) |
| 机制诊断 4 风电区(+聚合头 fut_exog 通道) | 同上 | lull 三分位:+0.6/−1.7/+0.5/+0.5;VIC1/SA1 全程 −1.2/−1.5 |

诊断脚本:`scripts/analysis/diag_wind_lull.py`(按 origin 处 regime24 三分位:lull/mid/normal)。
冒烟输出:`results/fd16_smoke.json`;比较器:`scripts/analysis/compare_eval_runs.py`。

## 为什么无效(机制解释)

1. **信息冗余**:聚合头的 DLinear 直接消费 336 h rs 遥测 —— 干旱在输入序列里一目了然,regime 通道不携带新信息。I_cfg 冷模式下旱情也已被年际锚定部分覆盖。
2. **路由结构**:最需要干旱处理的区域(SA1/VIC1,wind_cfg≥0.25)被结构路由器送往聚合路径,燃料路径的锚定对它们不生效;而走燃料路径的区域风电占比小,CIF 对风头参考值不敏感。
3. **剩余误差的本质**:lull 过渡日的失败模式是"干旱退出的时机",由未来风速的不可预测性决定 —— 这是**天气预报技巧问题**(与 MAE-floor 分析的波动性天花板一致),架构层面的条件化无法弥补。真正可行的杠杆是 NWP 集合/多成员预报输入,而非单值 regime 特征。

## 决定

- **保留代码**:干旱锚定消除了归一化的真实瑕疵(数学上更诚实);_wn 通道一致性是正确性修复;成本中性(冒烟 p=0.49,无回归)。
- **不宣称 MAE 增益**,不触发全量 29×5(无信号可确认,遵循既往负结果记录惯例)。
- 后续方向(若要再攻 lull 日):NWP 集合预报输入 / 概率轨道(CRPS);VIC1 风电场坐标加权天气(shelved 项)。
