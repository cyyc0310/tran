# 级联水箱外部系统数据准入审计

审计日期：2026-11-03。此次审计只确认来源、许可、文件完整性和数据合同；没有使用数据训练模型或产生 benchmark 分数。

## 来源与完整性

数据集由 Schoukens 与 Noël 发布在 4TU.ResearchData，DOI 为 [10.4121/12960104](https://doi.org/10.4121/12960104)，官方记录标注 **CC BY-SA 4.0**。[TU/e 数据记录](https://research.tue.nl/en/datasets/cascaded-tanks-benchmark-combining-soft-and-hard-nonlinearities/) 和 [NonlinearBenchmark 任务说明](https://www.nonlinearbenchmark.org/benchmarks/cascaded-tanks) 均确认这是带泵输入的实体级联水箱数据。

本轮 4TU 主下载端点 TLS 握手超时；随后使用官方 `nonlinear_benchmarks` 加载器中列出的 Google Drive 镜像获取原始 ZIP。归档字节数为 7,520,592，与官方加载器登记大小相同；MD5 `f832776ed1efdca4572338f467b7baec` 与 4TU 文件记录相符；独立计算 SHA-256 为 `eb0fa05851e8a7136846c2e3b61fbef87def78d0852c86ab91b02ac5db541b51`。ZIP CRC 检查通过，CSV/MAT/PDF 的内部 SHA-256 收据见 [`acquisition_manifest.json`](external-source/cascaded_tanks/acquisition_manifest.json)。

## 数据合同

- 两个长度均为 1,024 的记录：估计集 `uEst/yEst` 与测试集 `uVal/yVal`。CSV 数值按约四位小数保存，MAT 保留更高精度；四个数组逐项对应，最大绝对差约 `5.0×10⁻⁵`，符合 CSV 四舍五入精度。MAT 数组无缺失值；CSV 采样间隔 `Ts=4 s`。
- `u` 是泵电压，样值间零阶保持；`y` 是下水箱水位传感器读数。原始输入是多正弦信号，基准论文报告频率范围 0–0.0144 Hz。[官方基准说明 PDF](https://cris.vub.be/ws/portalfiles/portal/24190624/CascadedTanks_Maarten.pdf)
- 基准说明指出初始状态未知，估计与测试记录的起点状态相同；测试记录原本不得用于模型估计。任何对测试前缀做目标校准的扩展，都必须另立任务名和协议，不能称作原基准测试分数。
- 官方设备说明确认：下水箱使用电容式水位传感器，传感器未校准且可能带来非线性；模型中的水箱状态是实际水位，但观测 `y=x₂+e` 含测量误差。该说明未给出水位物理单位、传感器校准曲线、罐体工程上限或泵/执行器安全电压范围。因此只能把潜在水位非负作为模型物理结构，不能把读数 `y` 当成严格非负或把 `y=10.0` 当成已认证的高度/安全边界。泵输入可以确认是电压，但数据记录中的取值范围不是设备额定范围。
- 同一说明描述上水箱溢流、溢流进入下水箱以及由此产生的输入相关过程扰动；并报告测量信噪比约 40 dB。数据中 `y=10.0` 的样本在估计集有 47 个、测试集有 37 个。该值只作为观测饱和诊断。

## 对当前研究目标的价值与限制

这是一个独立于 InTaSet 和 Nano-Drone 的实体水箱基准，提供明确的泵电压输入、稀疏采样和真实溢流非线性。它只有一个估计记录和一个测试记录，且两者起始状态相同；不是独立重复或新的双向工况迁移。其单输入、单输出、4 秒采样也与 InTaSet 的四水箱/多通道和 Nano-Drone 的 100 Hz 多状态数据不同，必须分任务报告，不能将原始 RMSE 合并排名。

本文件记录的是**数据准入阶段**，当时未训练模型。之后另行冻结了稀疏输入扩展协议并生成开发期结果，详见[任务协议](CASCADED_TANKS_SPARSE_INPUT_PROTOCOL_V1_20261103.md)和[结果报告](CASCADED_TANKS_SPARSE_INPUT_RESULTS_20261103.md)。测试前缀校准版与原始官方测试协议分开；该扩展不属于官方未触碰测试分数，也不是盲测确认。

## 复核命令与产物

- 原始数据：[`external-source/cascaded_tanks/CascadedTanksFiles.zip`](external-source/cascaded_tanks/CascadedTanksFiles.zip)
- 来源清单：[`external-source/cascaded_tanks/acquisition_manifest.json`](external-source/cascaded_tanks/acquisition_manifest.json)
- 只读核验器：[`verify_cascaded_tanks_source_v1.py`](../../scripts/verify_cascaded_tanks_source_v1.py)

```bash
python scripts/verify_cascaded_tanks_source_v1.py
```
