# Cascaded Tanks 二状态 Python N4SID 稀疏预测结果

日期：2026-11-03。状态：**公开测试记录上的开发期基线；测试标签以前已看过，不是盲测或独立确认。**本结果补上一个可复算的状态空间基线，不能当作作者 MATLAB 分数的重现。

## 方法和评测合同

Forgione–Piga 的公开脚本使用 MATLAB `n4sid(data_id, 2)` 拟合二状态模型，再用估计记录获得的初始状态模拟验证记录。[作者脚本](https://github.com/forgi86/sysid-neural-continuous/blob/956aa7ecd73061b0321c6f6098d74c9e4a7ce67a/examples/CTS/CTS_subspace_comparison.m) 本机没有 MATLAB/Octave，因此这里采用独立 Python 包 NFourSID 1.0.2，固定上游提交 `1e306879f9e697cd275f17d01dc96182b74ac97d`。NFourSID 文档说明它提供 N4SID 子空间辨识、状态空间模型和 Kalman 滤波；其滤波器支持缺测输出时只按输入推进状态。[NFourSID 方法说明](https://nfoursid.readthedocs.io/en/latest/index.html) · [稀疏输出 Kalman 接口](https://nfoursid.readthedocs.io/en/latest/source/kalman.html) · [NFourSID 1.0.2 与 MIT 许可](https://pypi.org/project/nfoursid/1.0.2/)

本结果称为**二状态 Python N4SID**，不称为 MATLAB 原样复现。Python 包的子空间实现、候选 Hankel 块行、初态估计、随机状态空间噪声模型和稀疏评测都与作者 MATLAB 默认脚本不同。两者也不是同一个评测合同：作者脚本对验证记录连续模拟；本项目在每个预测块起点重新读取允许的下水箱观测。

源端只使用 `uEst/yEst`。时间末段 20% 作为源端选型片段；固定状态阶数为 2，在 8 个 Hankel 块行候选中，以 4、12、20 秒跨度的验证 NRMSE 宏平均选择。选定后用完整估计记录重拟合模型。目标端不调参数、不拟合校准量；每个非重叠块起点读取一条真实输出，块内不读中间真实输出，并使用该块记录的完整泵输入路径。测试标签已看过，因此以上约定不能追溯称作测试前冻结。

## 源端选型

| Hankel 块行 | 4 秒 NRMSE | 12 秒 NRMSE | 20 秒 NRMSE | 宏平均 |
|---:|---:|---:|---:|---:|
| 2 | 1.54692 | 1.32594 | 1.32833 | 1.40040 |
| **3** | **0.03635** | **0.06875** | **0.13744** | **0.08084** |
| 4 | 0.18587 | 0.37688 | 0.81215 | 0.45830 |
| 6 | 1.38664 | 1.38512 | 1.38470 | 1.38549 |
| 8 | 1.37904 | 1.37935 | 1.37714 | 1.37851 |
| 10 | 1.37563 | 1.37585 | 1.37364 | 1.37504 |
| 12 | 1.37059 | 1.36948 | 1.36701 | 1.36903 |
| 16 | 1.36884 | 1.36668 | 1.36402 | 1.36651 |

源端选中块行数 3。最终用完整估计记录重新拟合的状态矩阵特征值是 `−1.02529` 和 `0.98922`，谱半径 `1.02529`，因此这个线性模型含一个不稳定极点。目标长跨度误差与不稳定极点相伴出现；现有结果无法单独证明它是误差爆发的唯一原因。

## 目标测试结果

归一化误差分母为完整估计记录输出标准差。每种跨度独立以块起点输出更新状态，块内使用模型预测推进。

| 冻结方法 | 4 秒 NRMSE | 12 秒 NRMSE | 20 秒 NRMSE |
|---|---:|---:|---:|
| 一阶 ARX | 0.03532 | 0.06774 | 0.09592 |
| SINDYc 风格二次稀疏模型 | 0.03448 | 0.06554 | 0.09241 |
| 正则二阶 ARX(2,2) | 0.02590 | 0.05603 | 0.10550 |
| Forgione–Piga 作者混合结构 | 0.02332 ± 0.00019 | 0.03564 ± 0.00015 | 0.04327 ± 0.00035 |
| **二状态 Python N4SID** | **0.04627** | **0.73396** | **16,122.50932** |

在 4 秒跨度，Python N4SID 已弱于表中所有冻结模型；到 12 秒误差大幅增加；20 秒误差失控。20 秒预测全部仍是有限数值，但 45.69% 为负，93.92% 超出估计记录输出范围，预测最小/最大值约为 `−203,850`/`193,006`。所以“有限预测”不能理解为数值稳定或物理可信。线性插值使用了区间终点观测，是离线 oracle，不列在因果模型排名中。

本结果支持的结论是：**在这一条已查看水箱记录和此处固定实现/稀疏观测合同下，源端验证选出的普通二状态 Python N4SID 没有稳定迁移到目标长跨度。**它没有证明 N4SID 一般无效，也没有验证稳定约束 N4SID、作者 MATLAB N4SID 或其他辨识选项。该结果使作者混合结构与 Python N4SID 有了一次同数据条件下的比较，但不能推广成其胜过所有状态空间方法。

## 数据、复算与核验收据

- 原始 ZIP SHA-256：`eb0fa05851e8a7136846c2e3b61fbef87def78d0852c86ab91b02ac5db541b51`；MAT 成员 SHA-256：`cb2f88d4388be4d3f2a24c6402fba804976aac5f2e1f26cda59ea0a38d016eab`。
- 协议 SHA-256：`7a95a2e5669fe0ace5edf007a944c6f3b255daadae7a23b593923c5487a0f57b`。
- 主脚本 SHA-256：`8c70346cb9bbe5dd9bcbfd0c5d57e59b5773859a6534f05d05a88e79b0b2bcb7`；独立核验脚本 SHA-256：`646024ee92b68780473d2c59ec3a45c67c72f7efbf00b3fd92cf95ea55a8ceb2`。
- 四个输出文件的完整尺寸与 SHA-256 保存在 [`python_n4sid_metadata.json`](results/cascaded_tanks_python_n4sid_sparse_v1/python_n4sid_metadata.json)；模型矩阵和极点在 [`python_n4sid_model.json`](results/cascaded_tanks_python_n4sid_sparse_v1/python_n4sid_model.json)。逐块分数和完整源端候选表分别在 [`python_n4sid_sparse_metrics.csv`](results/cascaded_tanks_python_n4sid_sparse_v1/python_n4sid_sparse_metrics.csv)、[`python_n4sid_sparse_blocks.csv`](results/cascaded_tanks_python_n4sid_sparse_v1/python_n4sid_sparse_blocks.csv) 与 [`python_n4sid_source_validation.csv`](results/cascaded_tanks_python_n4sid_sparse_v1/python_n4sid_source_validation.csv)。
- NFourSID 源码和许可按固定提交原样保存在 [`external-source/nfoursid/`](external-source/nfoursid/)；追加运行依赖见 [`requirements-cascaded-tanks-n4sid-v1.lock.txt`](../../requirements-cascaded-tanks-n4sid-v1.lock.txt)，需与项目基础锁共同使用。

复算与独立重放命令：

```bash
python3.12 scripts/cascaded_tanks_python_n4sid_sparse_benchmark_v1.py
python3.12 scripts/verify_cascaded_tanks_python_n4sid_sparse_benchmark_v1.py
```

核验器重新拟合 8 个源端候选，并用保存的模型矩阵和独立手写的 Kalman 更新方程重放 3,066 条目标预测、重算各跨度误差；当前通过。它只证明当前文件、实现和该数据记录之间可复核，不构成新物理记录或独立确认。

本结果已加入公开版 `control-benchmark-v1.1.0`，并由 v1.2.0 保留。后续加入的 Forgione–Piga 作者结构/训练配方适配和正则 ARX(2,2) 对照也在 v1.2.0 中提供脚本、协议、结果与核验器。N4SID 结果仍只表示这条已查看记录上的固定 Python 配置；若继续补线性基线，稳定性处理必须作为独立方法并预先写定源端选型规则，不能在同一目标记录上调参后称为独立确认。算法主张仍需新整条运行或新物理系统上的冻结后验证。
