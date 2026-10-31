# V294 冻结协议：中国省级月度燃料分项数据的观测/估算血缘审计

日期：2026-10-30。目标是核验一份当前公开的中国31省月度按燃料发电表，区分直接发布字段、统计调整和模型估算字段，判断它是否可作为 TransCIF 的输入或独立验证标签。

## 冻结来源

- Ember China monthly CSV：`https://files.ember-energy.org/public-downloads/generation/outputs/release_chn_subnational_generation_monthly.csv`；官方数据页面标注2026-10-28更新。
- 同站点方法 PDF：`https://storage.googleapis.com/emb-prod-bkt-publicdata/public-downloads/ember_china_generation_methodology.pdf`；封面标注2026年8月更新。

脚本保存下载的CSV与方法PDF快照并记录SHA-256。报告只审计CSV字段、覆盖和发布方法，不将排放列作为小时标签，也不拟合/评分预测器。若CSV含全国汇总行，则将其与31个省级地区分开统计。

## 审计项目

1. CSV列名、燃料类别、聚合类别标记、单位、日期范围、地区数、月度行数和重复键。
2. 对每类燃料检查最近日期上的省份覆盖；分开逐燃料原始行与总计/组合行。
3. 在方法PDF中定位来源、燃料拆分、地区拟合、再生能源修正、排放/因子生成及年/月差异。
4. 对适用的原始类目验证`1000 × Emissions (MtCO2e) / Generation (TWh)`与`Emissions intensity (gCO2e/kWh)`的算术关系，确认强度字段是否为排放列的派生量。

## 判读规则

- 来源方法明确为校正、分摊或模型构造的字段必须标为估算/再分析产品，而非地方逐月独立观测。
- 与生成量同时给出的排放量/因子若由共享上游EF或构造过程生成，不可作为独立实测CIF验证源。
- 审计结果只适用于该Ember版本；不推断用户手头数据一定来自Ember，用户原始表仍应按其来源另立数据合同。
