# FD-50p 二轮 TSFM 压降 — 判定

**日期**：2026-09-11 ｜ **通道**：TSFM 免训练融合第二轮（用户要求继续压降）
**脚本**：`scripts/experiments/fd50p_blend2.py`、`fd50p2_select.py`、`fd50p3_seeds.py`
**结果**：`results/fd50p_blend2.{json,md}`、`fd50p2_select.json`、`fd50p3_seeds.json`

## 结论一句话

三分量凸混合 **best3 = w_g·g(EWMA-J5) + w_per·persistence + w_C·Chronos-2**（权重 early 拟合、
0.05 网格）把 29 区 late MAE 从 J5 的 30.76 压到 **27.95**（FD-50o 的 J5||C 为 28.27），
重灾区（J5>40 的 9 区）47.58 → **43.57**；5-seed 稳定 27.97 ± 0.10、全部 29/29 胜 J5，
增益 −2.8~−3.0。这是 I_lag 层零训练配方的新最优。

## 主结果（late 半段）

| 臂 | all 29 | J5>30（15 区） | J5>40（9 区） |
|---|---|---|---|
| J5（旧最优） | 30.76 | 42.68 | 47.58 |
| J5||C（FD-50o） | 28.27 | 39.86 | 44.08 |
| **best3（新最优）** | **27.95** | **39.36** | **43.57** |
| select/meta 臂选择 | 27.94-27.95 | 39.35-39.36 | 43.57 |

全周期口径：J5 32.31 | best3 29.15。5-seed：best3 27.97 ± 0.10（J5 30.85 ± 0.15），
逐 seed 29/29 全胜。

## 各臂读数（all，late）

- C（纯 Chronos）30.83、CE（EWMA 校正 Chronos）31.74、CEp（CE+persistence）31.71：
  **对 Chronos 施加 EWMA 水平校正是负向的**——TSFM 已自带水平信息，重复校正反而伤害。
- best3 0.10 网格 27.86 / 0.05 细网格 27.95*：细网格在 all 上略逊（部分区过拟合 early），
  但 select 臂（逐区在 early 上选 best3f 或 J5OC）收敛到与 best3f 相同（27.95），说明
  逐区臂选择没有额外空间，best3 的三分量结构已是信息层的自然形态。
  （*fd50p2 的 best3f 0.05 网格 all=27.95 与 fd50p 0.10 网格 27.86 差异来自网格粒度，
  两者在重灾区一致 43.4-43.6。）
- ce720/ce336（短上下文消融）：全线上调 0.5-1.5，2048h 长上下文是必要配置。
- qtrim（q25/q50/q75 平均）：≈C，分位数平滑无增益。

## 关键发现

1. **最优结构是“EWMA 水平 + persistence + TSFM 形状”的三分量并联**，而非把 TSFM 塞进
   J5 的 persistence 槽位（J5+C 28.91）或与其外混合（J5||C 28.27）。Chronos 的价值在
   形状先验，与 EWMA 的水平校正、persistence 的日内锚各自独立贡献。
2. **对 TSFM 输出做 EWMA 水平校正是重复计数**（CE 31.74 > C 30.83）：水平信息 TSFM 已
   从上下文读出，J5 的在线校正层只该作用于模型原生输出（g）。
3. **上下文长度 2048h 优于 720/336**：长历史（约 85 天）对碳强度这种慢漂移序列有真实价值，
   印证 Chronos-2 的长上下文设计。
4. **逐区臂选择与元混合无增量**（27.94 vs 27.95）：说明在 early 信息下，两臂的逐区优劣
   不可分辨——best3 已充分利用 early 可辨识的结构。
5. **5-seed 稳健**：增益 −2.8~−3.0（每 seed 29/29 胜），跨 seed 的权重变化不改变结论。

## 对 20 目标与论文叙事

- best3（27.95）距 20 仍远，20 不可达判定维持（TSFM 只改形状，不改水平作弊地板 23.14）。
- C5 边界链的 TSFM 通道段更新：最优融合形态为三分量并联（非外混合/非槽位替换），
  这成为“零训练配方的自然终点”叙事的收官形态：EWMA（水平）+ persistence（日内锚）
  + Chronos（形状）各司其职，任何单分量替换都更差。

## 复现

```bash
cd /Users/cyyc0310/code/tran
HF_HOME=$PWD/.hf_cache .venv-nemed/bin/python scripts/experiments/fd50p_blend2.py   # 缓存+混合臂
.venv-nemed/bin/python scripts/experiments/fd50p2_select.py                          # 臂选择
.venv-nemed/bin/python scripts/experiments/fd50p3_seeds.py                           # 5-seed
```

依赖：chronos-forecasting 2.3.2（.venv-nemed），模型权重缓存 `.hf_cache/`（一次下载后离线可用）。
