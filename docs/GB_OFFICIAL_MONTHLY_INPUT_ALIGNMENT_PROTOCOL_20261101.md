# 英国官方月度能源统计与小时燃料组合对齐审计

冻结日期：2026-11-01。此审计针对四项研究优先级1的输入合同，回答能否用英国官方、按月发布且滞后两个月的发电统计替代从NESO修订小时数据回算的目标月输入。它不训练或选择小时预测模型，不读取CIF响应。

## 预测任务与数据口径

- 目标响应暂固定为现有NESO GB小时发电燃料序列，2024自然月，仅读取燃料发电量，不读取`CARBON_INTENSITY`。
- 当前官方来源采用DESNZ Energy Trends ET 5.4（2026-10-29发布的Excel）。该版明确包含2026年8月数据，历史2024月值是该文件中的当前修订值，不是2024预测起点可用的as-issued快照。
- ET 5.4按官方说明在月末发布、数据滞后两个月，但月度表仅覆盖Major Power Producers；全体发电商仅在季度表中汇总。统计地理是UK，NESO目标序列为GB。这两种覆盖边界必须作为独立口径差异披露。
- 统一五组映射：thermal = coal + oil + gas + bioenergy + other fuels；hydro、nuclear、wind、solar分别对应同名列。储能净量、净进口和其他来源采购不视为本地发电燃料，均不放入五类分母。缺失/`[x]`不得当作零。
- NESO对照亦按五组聚合：thermal = GAS + COAL + BIOMASS + OTHER；hydro = HYDRO；nuclear = NUCLEAR；wind = WIND + WIND_EMB；solar = SOLAR。月份先按燃料电量求和再归一化。

## 主要比较

逐2024月比较两种数据源五类组成：按组绝对份额差、总变差距离（L1/2）、thermal份额差和五类组成相关。只有双方该月组成完整、非负且总量为正时纳入主审计。输出按月表及宏平均表，并保留源工作簿、脚本和NESO输入SHA-256。

## 判定边界

若月度官方ET 5.4与NESO同月五类份额差异较大，则其不能无校正地代表完整GB小时目标的同口径粗组成；即使相近，也只能说明当前修订值的统计口径对齐，不证明历史as-issued输入可恢复或预测效果相同。由于英国官方ET 5.4只覆盖MPP、UK且目标是GB、当前工作簿含历史修订，本审计不升级为真实发布版模型回测，也不替代2026前瞻预测封存。

## 来源

- DESNZ [Energy Trends: UK electricity](https://www.gov.uk/government/statistics/electricity-section-5-energy-trends)：官方说明ET 5.4为月度发电统计、月末发布且滞后两个月；月度数据覆盖MPP，完整发电商数据在季度统计中。
- 工作簿取自该页2026-10-29当前ET 5.4附件。工作簿封面称该版新增2026年8月，且修订期覆盖2026年7月。
