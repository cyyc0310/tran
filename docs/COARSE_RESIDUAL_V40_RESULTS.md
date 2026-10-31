# V40：低维粗残差校准没有迁移成功

2026-10-27，Sol。固定五臂比较完成。ResidualDS的CIF与内部EF误差均劣于ResidualS及未校准ensemble；V39的历史偏差持续性没有转化为本轮可部署纠偏增益。不继续在这六个源级样本上增加交互或挑回退门槛，下一步先检查开发供体覆盖。

## 同预算比较

按[合同](COARSE_RESIDUAL_V40_PROTOCOL.md)，首次actual原session24503退出0，prepare/predict/score为0/0/0；33项合成测试和两份独立预审通过。复用42个已存五源Month profile，没有重拟合曲线；每outer六个源级标签，四个alpha=1小Ridge校准器，共28个新fit。未把源月份当独立地区。

五臂共享同一六component ensemble、同样源late12平均EF偏差和合法粗历史。源pseudo-origin及目标query都按end/release取12条，历史月各用自己的year−2登记库存；目标窗口仍可能陈旧或缺月。None、Mean、D、S、DS分别不校准、只用均值、只用粗残差、只用曲线推得结构、联合两输入。无early真实EF或late实现总量输入，响应true−pred/1000，截距未罚，先component clip再平均，原MG不变。

所有42查询预测封存后才读35个评分月。原1330个参考case数值不变，新增350行，共1680 case、336 region、48 macro、336去单源敏感性、42 support。ISNE仍只一个评分月。源曲线/标签是2025截止时点的回溯构造，并非2023 as-of预测；五源规模匹配也不消除operator变化、窗口及年度均值到单月的迁移差异。

源端同键绝对发电量、登记库存仍属于V38/V39的B1附加表示预算；不宣称中国仅B0月表即可实施全部模型。CIF/EF均为固定燃料因子的发电侧代理，不是实测排放或含进口的消费侧碳强度。

| 模型 | Mean CIF MAE | Median CIF MAE | EF MAE |
|---|---:|---:|---:|
| V36 CoalOilUpper参考 | 16.396137 | 15.495588 | 25.708227（mean） |
| EnsembleNone | 31.745721 | 32.965980 | 55.808309 |
| ResidualMean | 30.808701 | 32.041329 | 59.330059 |
| ResidualD | 38.340809 | 39.803895 | 63.904191 |
| ResidualS | 39.412804 | 40.720383 | 73.064202 |
| ResidualDS | 48.013856 | 48.960487 | 79.864856 |

单位gCO2/kWh，先源内平均再七源等权。新臂的EF不随MG粒子变化，mean/median相同；CIF两个摘要均保留，不据此挑摘要。已有参考不是都与本轮源响应表示相同，仅作外部强参考；五个新臂之间的信息预算一致。

| 源 | None CIF／EF MAE | S CIF／EF MAE | DS CIF／EF MAE |
|---|---:|---:|---:|
| BPAT | 6.8227／88.5274 | 8.8800／131.7284 | 7.8484／112.4267 |
| CISO | 9.2207／1.2557 | 9.2392／1.5878 | 9.2744／1.8967 |
| ERCO | 55.1689／111.9561 | 43.1093／87.9813 | 25.4277／44.4848 |
| FPL | 12.5588／1.1520 | 12.0732／0.0001 | 110.9423／148.5790 |
| ISNE | 26.4116／17.3879 | 23.4854／12.1592 | 31.3107／26.1414 |
| MISO | 91.5144／160.6369 | 158.4058／266.9287 | 131.4280／224.3331 |
| NYIS | 20.5229／9.7422 | 20.6966／11.0638 | 19.8656／1.1924 |

DS相对S仅4/7源同时改善两指标，相对None仅ERCO和NYIS改善。去MISO后，DS CIF/EF为34.1115/55.7868，仍差于S19.5806/40.7534和None21.7843/38.3369。不能以ERCO/NYIS改善代替整体迁移证据。

![五臂逐源误差与简单门控参考](../results/coarse-residual-v40-figures/coarse-residual-controls.png)

图由正式region表生成，两面板各42个点，包含五个新臂及V36参考；PNG/PDF与输入/输出hash收据保存在同目录。图为事后展示，不是模型输入。

## 已见的外推缺口

事后另存252个query/component的D/S及六校准源各列min/max，不改原方法或反馈选择：FPL的D有36/36超出校准范围；MISO的S有30/36超出，BPAT的D有30/36超出。FPL校准D最大约.0952，query为.1247至.1448；MISO校准S最大约.1150，query可达.2548。

这些是边际范围描述，不是联合支持认证、因果失败解释或风险界。ISNE虽两列都在范围内，DS仍退化，说明落在范围内也不保证条件关系可迁移。五臂分别有0/0/34/63/61个component被clip（每臂252个）；clip保证因子范围，没有修复错误的纠偏方向。

补充CSV位于`results/coarse-residual-v40-feature-support.csv`及`results/coarse-residual-v40-clip-diagnostics.csv`，明确不属于原24个runner输出。未用这些标记临时设计gate再回填评分。

## 下一步先改变供体覆盖，不把数据增益当算法创新

六个源级样本既少，又缺少部分结构类型。本轮更适合检验“已查看开发供体是否覆盖不足”，而非继续增加同一小样本回归的复杂度。

仅做元数据盘点，已找到原V4六区SWPP/AVA/BANC/JEA/SC/NWMT和V5四区SCL/WACM/SEC/SCEG的既有月表。[V5结果](RISK_TRANSFER_V5_RESULTS.md)明确十区均已查看，不能重新称未见测试。这十区均有2022–2025日期记录，四年cached库存Plant表中也有同名BA code；这不证明数值质量或库存口径可直接用。尚未读取它们新的发电/份额/容量数值，也未训练扩展池。

拟先核新十区的库存/报告membership，再比较原七源池与扩展十七源池；原七outer及35评分月保持，PACE/LGEE和真正未查看域继续封存。候选与强基线获得相同新增历史监督；固定总训练权重质量或等价平均损失正则，避免把SUM样本变多后的正则变弱当覆盖增益。

首轮只能判断新增供体整包干预是否有用，源数量、类型、质量和样本量同时变化，不能单独归因为数量，更不能称方法原创。若有效，再研究可观测支持与不可识别碳方向的风险机制；若无效，也只是否定这次供体扩展，不能推广成所有迁移无效。

## 新颖性与可信风险边界

新查到的[UAI2026证书驱动校准](https://proceedings.mlr.press/v337/huang26b.html)已使用门控残差Bayesian head。其原文Eq25–26使用预测后目标响应反馈；完整界仍有依赖目标标签机制的余项，输入侧可计算surrogate不等于该余项可被识别。我们只有粗GPOS反馈，不能替换其监督损失后继承目标CIF保证。已读范围为[原文§3.2、§4.2–4.3及附录A.5](https://raw.githubusercontent.com/mlresearch/v337/main/assets/huang26b/huang26b.pdf)，不冒称全面综述。

[CLeaR2026 Proxy-Guided Measurement Calibration](https://proceedings.mlr.press/v323/vishnubhatla26a.html)也已研究代理辅助的聚合测量纠偏；本轮只核摘要，其代理与偏差机制的独立结构并未在本项目验证。普通残差校准、gate、回退或proxy命名都不足以构成创新。

独立公式重建使用正式NumPy1.26.4环境，不导入V40 producer、不重跑stage或旧profile。源/查询特征、28个正规方程、全部component/clip/ensemble、EF/CIF粒子及1680 case数值一致，1330参考numeric exact，支持字节相同。聚合最大差3.637978807091713e−12，低于预锁1e−9/rtol0；normal残差L2最大2.7755575615628914e−17，46个本轮直接绑定文件首尾hash不变。

本轮未找到方法突破，没有评价CIF区间覆盖、宽度或小时风险，未打开旧目标或未知地区。长期目标保持active。
