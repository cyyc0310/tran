# 控制系统辨识研究

[English](README.md) | 简体中文

本仓库当前聚焦控制工程中的系统辨识与模型迁移。公开基准在稀疏状态观测和已记录控制输入条件下比较预测与辨识方法。

## 当前基准

**跨工况控制系统辨识基准 v1.2.0** 包含 InTaSet 的 M1↔M2 工况迁移，以及独立的 Cascaded Tanks 物理系统。仓库提供评测协议、数据来源凭据、脚本、归档结果和可独立复现的核心实验包。

- [基准简介与复现说明](system-identification-benchmark/README.md)
- [下载 v1.2.0 发布包](https://github.com/cyyc0310/tran/releases/tag/control-benchmark-v1.2.0)
- [发布说明与证据边界](system-identification-benchmark/RELEASE_NOTES_v1.2.0.md)
- [逐文件哈希清单](system-identification-benchmark/RELEASE_MANIFEST.json)

比较的方法包括 ARX/线性状态空间、SINDYc 风格模型、独立实现的受控 Neural ODE/混合模型、按论文结构与主要训练配方适配的作者神经混合模型、二状态 Python N4SID，以及有限目标前缀校准。端点插值作为离线参照单独报告。作者混合模型和 N4SID 都在一条已查看的 Cascaded Tanks 测试记录上评估，不是盲测；N4SID 也不是 MATLAB 原样复现。

## 证据状态

现有证据支持发布可复现基准并分析方法适用边界；尚未证明某种方法能在不同迁移方向和物理系统上稳定胜出，也不支持提出已验证的新算法。开发期间已查看测试记录，因此结果不属于盲测。InTaSet 作者的 Neural ODE 无法从已检查的公开材料复现；包中的 InTaSet Neural ODE 是独立实现。双水箱作者混合模型是按已发表结构和主要训练配方改写到稀疏输出协议，不是原论文分数的复算。Nano-Drone 仅作补充评估，其上游许可范围尚未核实。

## 许可范围

MIT 许可仅覆盖 `system-identification-benchmark/` 中项目编写的代码和说明文字。第三方数据集与实验结果有各自的复用条件，详见[许可范围说明](system-identification-benchmark/LICENSE_SCOPE.md)和[第三方声明](system-identification-benchmark/THIRD_PARTY_NOTICES.md)。

## 仓库历史资料

旧项目材料继续保存在[历史项目说明](README_transcif_legacy.md)及[简体中文版](README_transcif_legacy_zh.md)中，与当前控制系统基准分开。
