# 上海公开月表：无小时标签推理接口已接通

2026-10-31。本轮是实际公开输入的接口验证，不是方法晋级、中国精度验证或可推荐的生产预测。

## 实际执行

外国模型只读取DE/GB/US数据：2022训练53,286小时，2023H1校准36,809小时；保存8个研究参考，未根据中国标签选择模型或再训练。独立推理进程仅读取模型包、NBS月表、来源清单及绑定凭证，没有目标小时CIF文件参数。

起点2026-11-01 00:00+08，预测24小时。按精确t−2读取2026-09公开组成，历史特征仅用此前六个历月窗口。上海当月火电78.5、水电0、核电0、风电2.1、太阳能1.5，原单位亿kWh，总量82.1。没有用累计量差分补月或把缺失填0。共输出192行（8参考×24小时）。

NBS是规上工业发电口径，本地生成侧代理不等于含外来电的消费侧CIF。总量与五类差在六个四舍五入误差合计0.3原单位以内时，使用五类归一化；其余项0仅是模型参考占位，不能称测得无其他可再生。差值超限或精确t−2缺失会拒绝该输入。

## 输入时点凭证修复

初版沿用来源清单的`retrieved_at_utc`。审查发现它是开始抓取时间，不能独立证明整张表已完成。原结果和代码保留。现新增seal：读完月表及清单、逐一验证186个原响应SHA并复核文件未变化后，记录当前保守可用时间；panel和来源清单都绑定SHA。

本次seal可用时间为2026-10-31T07:35:21.614367Z，早于预测起点。CLI必须校验凭证，拒绝错配表、错配清单、无时区或起点早于凭证的输入。不声称该seal证明更早的发布版本。修复版r2与r1的192行预测数值逐元素完全相同，是时间凭证完善，不是新实验确认。

## 支持诊断揭示真实缺口

外国训练火电份额支持约[.0123,.7957]，上海为.9562；源核电份额约[.0583,.3392]，上海为0。火电份额历史均值也越界。最初7个美国区和2022年的德英源都没有提供与上海相同的零核电、高火电组合。

模型训练终点距该预测起点47个月，风险参考也来自2023H1。模型包显式标记`innovation_gate_passed=false`和`current_production_accuracy_certified=false`。这些参数用于验证可运行性，不能据此宣称当前准确性。

示例的研究参考均值从直接历史HGB约295到Ridge约676 g/kWh不等，macro重组约534；没有本地真值时不能据输出本身挑选“最好”的模型。输出列名为`source_reference_lower90/upper90`，明确是源校准参考，不是已验证的中国90%覆盖。

## 可复现入口

```bash
python scripts/export_public_transfer.py --output results/foreign-package-new
python scripts/seal_public_snapshot.py \
  --panel results/v394-nbs-all-province-long-panel-r1/provincial_mix_and_closure.csv \
  --manifest results/v394-nbs-all-province-long-panel-r1/manifest.json \
  --output results/nbs-input-seal-new.json
python scripts/predict_public_monthly.py \
  --model results/foreign-package-new/foreign_models.joblib \
  --panel results/v394-nbs-all-province-long-panel-r1/provincial_mix_and_closure.csv \
  --manifest results/v394-nbs-all-province-long-panel-r1/manifest.json \
  --seal results/nbs-input-seal-new.json --province Shanghai \
  --origin 2026-11-01T00:00:00+08:00 --output results/shanghai-new
```

新的seal只支持其实际验证时间之后的起点。若在示例日期之后运行，应选择更晚且有合法月报支持的起点，不能倒签时间。模型包留本地`results/all-foreign-public-package-r1/`；[192行研究预测](evidence/shanghai-public-inference-r2/predictions.csv)、[输入与支持诊断](evidence/shanghai-public-inference-r2/inference.json)、[证据清单](evidence/shanghai-public-inference-r2/manifest.json)随仓库保存。

新增测试覆盖来源可用时间、双文件哈希绑定、精确t−2、缺月/闭合拒绝和无标签接口。与供体支持、三国及既有适配器合计40项相关测试通过，保留一条加权常量合成样本的StandardScaler数值警告；真实推理输出有限。独立Python审查确认单位、索引、标签隔离和seal修复。未运行全仓库测试或未安装的静态工具。

下一步是补足真实源支持并验证方法，不能把部署接口完成当作可信预测已经成立。年度煤/气装机预算仍待用户选择，不影响本轮月报接口的完成。
