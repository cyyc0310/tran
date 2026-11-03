# Cascaded Tanks 作者神经混合模型：稀疏输出结果与边界

日期：2026-11-03。状态：**公开数据开发期比较，标签已查看，不是盲测或独立确认**。本报告扩展了[水箱稀疏输入结果](CASCADED_TANKS_SPARSE_INPUT_RESULTS_20261103.md)，并加入正则二阶 ARX 与二状态 Python N4SID 对照。

## 数据、模型和评测任务

数据来自实体级联双水箱，原生采样间隔 4 秒；泵电压是输入，下水箱传感器读数是输出。官方估计/测试记录各 1,024 点。测试标签在数据准入和早期开发中已看过，以下都只用于提出假设。[官方基准说明](https://www.nonlinearbenchmark.org/benchmarks/cascaded-tanks)

Forgione–Piga 的公开代码包含级联水箱连续时间神经状态空间模型和训练样例，使用 100 单元 ReLU 网络、4 秒前向 Euler、128 步片段和截断模拟误差训练。本地固定上游提交 `956aa7ecd73061b0321c6f6098d74c9e4a7ce67a`，保留了原模型、训练脚本和 Apache-2.0 许可。本实验采用其主要结构与训练配方，但把输出接口改成稀疏块预测，故称**作者结构/配方的协议适配复现**，不是原论文分数原样复算，也不能代替 InTaSet 作者 NODE。[论文](https://doi.org/10.1016/j.ejcon.2021.01.008) · [固定版本源码](https://github.com/forgi86/sysid-neural-continuous/tree/956aa7ecd73061b0321c6f6098d74c9e4a7ce67a)

训练只用 `uEst/yEst`，优化器随机种子 0、1、2 各训练 10,000 次。作者代码使用其发布的 CSV；正则 ARX(2,2) 使用官方 MAT 高精度数据。官方 CSV 与 MAT 的对应数组最大差约 `5.0×10⁻⁵`，估计输出标准差相对差约 `2.1×10⁻⁶%`，因此表中归一化分母实质相同。测试块起点读取下水箱当前输出；块内只用该块记录的完整泵输入路径，不读块内真实中间水位。上水箱潜状态由源端训练后继续传播，下水箱状态在稀疏块边界重置。预测跨度和输出间隔均为 1、3、5 个转移（4、12、20 秒）。这测的是**离线、已知记录输入的分块条件预测**，不是未知未来输入预测、闭环控制或整条记录不重置的滚动预测。

## 冻结模型的比较

下表报告零目标校准预算的 NRMSE，分母为估计记录输出标准差。模型都在同一条公开测试记录上评分；ARX 与 SINDYc 数字来自[既有冻结基线](CASCADED_TANKS_SPARSE_INPUT_RESULTS_20261103.md)。

| 方法 | 4 秒 | 12 秒 | 20 秒 |
|---|---:|---:|---:|
| 一阶 ARX，冻结 | 0.03532 | 0.06774 | 0.09592 |
| SINDYc 风格二次稀疏模型，冻结 | 0.03448 | 0.06554 | 0.09241 |
| 正则二阶 ARX(2,2)，冻结 | 0.02590 | 0.05603 | 0.10550 |
| 二状态 Python N4SID，冻结 | 0.04627 | 0.73396 | 16,122.50932 |
| 作者神经混合模型，冻结权重 | **0.02332 ± 0.00019** | **0.03564 ± 0.00015** | **0.04327 ± 0.00035** |

正则二阶 ARX 的岭强度只用估计记录时间顺序末段选取：候选为 0 至 1,000，比较 1、3、5 步预测平均 NRMSE，选出 `α=0.1`，再用全估计记录重估。它在源端验证拒绝了导致多步误差急剧放大的弱正则值；目标 h=5 下的误差仍高于原一阶 ARX，说明源端选型并未消除转移不确定性。目标块起点只观测一个输出，故 ARX 上一输出状态由前一块预测延续；初始时将其设为首个观测值。详细收据见[ARX 协议](CASCADED_TANKS_ARX2_REGULARIZED_PROTOCOL_V1_20261103.md)与结果目录。

在这条记录上，作者混合模型比正则 ARX(2,2) 的 NRMSE 低约 9.9%、36.4%、59.0%（对应 4、12、20 秒），也低于本次 Python N4SID。但 Python 模型在 20 秒跨度失控，含谱半径 1.025 的不稳定极点；不能把结果写成“混合模型优于所有状态空间模型”。作者 MATLAB N4SID 的稀疏观测版本仍未运行，本机没有 MATLAB/Octave；Python N4SID 不是其原样复现。[Python N4SID 结果与核验收据](CASCADED_TANKS_PYTHON_N4SID_SPARSE_RESULTS_20261103.md) · [上游 MATLAB N4SID 脚本](https://github.com/forgi86/sysid-neural-continuous/blob/956aa7ecd73061b0321c6f6098d74c9e4a7ce67a/examples/CTS/CTS_subspace_comparison.m)

## 短前缀倍率校准

作者模型权重冻结，只从目标前缀拟合一个输出路径倍率 `gamma`。对 1、3、5 个前缀转移 × 1、3、5 个预测转移的全部 9 个组合，校准 NRMSE 都高于不校准模型。以最长预测为例：零前缀模型为 0.04327；用 20 秒目标前缀拟合 `gamma` 后为 0.14507，误差约增加 235%。一至两个点就能被倍率拟合得很好，但该调整没有转化为后续块的收益。校准表和所有目标前缀收据在结果目录中。

## 物理诊断和解释限制

- 三个种子的预测均有限，预测下水箱输出没有负值。
- 学得的“上水箱”潜状态约 51% 时间为负，范围约 −1.007 至 1.108。它不是被物理非负约束的水位轨迹，故这项低输出误差不能解读为恢复了真实水箱内部状态。
- 冻结模型约 5.4%–5.9% 的预测超出估计记录输出范围；20 秒跨度预测最高约 10.117，而测试读数最高为 10.0。水箱观测传感器未校准且存在饱和，因此这些只是数据范围诊断，不是设备安全限值。
- Python N4SID 在 20 秒跨度仍输出有限浮点数，但 45.7% 预测为负、93.9% 超出源端范围，预测约达 `−203,850` 至 `193,006`。有限数值不等于稳定，更不代表物理安全。
- 3 个随机种子描述模型初始化差异，不是 3 台设备或 3 条物理重复。73,428 个逐步结果行来自同一条物理测试记录，也不是独立样本。
- 目标测试未来泵输入来自事后记录，评测属于离线条件预测。它没有证明模型能在现场提前获得未来泵信号，也没有证明闭环控制效果。

## 复算和收据

数据来源 ZIP SHA-256：`eb0fa05851e8a7136846c2e3b61fbef87def78d0852c86ab91b02ac5db541b51`。作者模型 CSV 成员 SHA-256：`ef2388ed822f3aef4aa80d6b0f2b466dd80b361786b3eafc7a2957c31ea323a7`。作者代码、模型源和训练脚本哈希见 [`external-source/forgione_piga_sysid_neural_continuous/README.md`](external-source/forgione_piga_sysid_neural_continuous/README.md)。

作者混合模型主要收据：指标 CSV SHA-256 `4218e91e512fe3416ed74c9d74a1d27541b66f6728560033c66d7fadece9594c`；逐块 CSV SHA-256 `6c2679807e56f8b85186050567dd1cf1256daf68d01bd9bd93f303fdd366bc9e`；协议 SHA-256 `9f9d479cf55e1d27fa7391e4ea00ff90332dbb3fa7d990ef5db7b91c2a8edd49`。正则 ARX 结果和逐块 SHA-256 记录于 `results/cascaded_tanks_arx2_regularized_v1/arx2_ridge_metadata.json`。

在 Python 3.12.13 下复算：

```bash
python scripts/cascaded_tanks_author_hybrid_sparse_benchmark_v1.py --iterations 10000 --seeds 0 1 2
python scripts/verify_cascaded_tanks_author_hybrid_sparse_benchmark_v1.py
python scripts/cascaded_tanks_arx2_regularized_ablation_v1.py
python scripts/verify_cascaded_tanks_arx2_regularized_ablation_v1.py
python3.12 scripts/cascaded_tanks_python_n4sid_sparse_benchmark_v1.py
python3.12 scripts/verify_cascaded_tanks_python_n4sid_sparse_benchmark_v1.py
```

三个只读核验器均通过：作者混合模型重放 73,428 行预测且不用重新训练；两个线性基线各重放 3,066 行，N4SID 核验器还重新拟合 8 个源端候选并重算目标跨度指标。最初报告形成时，这些文件仅在本地研究工作树；作者模型、正则 ARX(2,2) 和结果收据现纳入公开候选版本 v1.2.0。公开包复核结果及其逐文件哈希记录在 `CLEAN_COPY_REPLAY_RECEIPT_v1.2.0.md` 和 `RELEASE_MANIFEST.json`。

## 下一步

此结果支持一个谨慎的实验观察：神经混合模型在一条外部水箱记录的 12、20 秒条件预测中强于正则 ARX(2,2) 和本次 Python N4SID；但它的潜状态违反非负性，Python N4SID 也不等同于作者 MATLAB 实现。稳定性约束模型只能作为另一个预先定义的开发方法；最终算法判断仍需新的未查看整条运行或新设备确认。所有已看测试曲线都只能作开发证据。
