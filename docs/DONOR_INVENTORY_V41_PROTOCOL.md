# V41：已查看开发供体的登记库存整理

2026-10-27，Sol，新增容量数值转换前固定。本阶段仅整理库存与核旧七源规则一致性，不读取新供体发电/份额数值，不拟合模型，不比较2025误差。

## 固定名单与来源

旧七源顺序BPAT/CISO/ERCO/FPL/ISNE/MISO/NYIS；新增开发供体顺序SWPP/AVA/BANC/JEA/SC/NWMT/SCL/WACM/SEC/SCEG。总17域，只有这些BA可转换容量。新增十域已查看的角色由`docs/RISK_TRANSFER_V5_RESULTS.md:72`确认，不重新称未知区。PACE/LGEE、其他保留旧目标及真正未查看域继续关闭。

父V40清单SHA `e4684d1c4f749ca3ccb79ab79058aa133be926597ebd66ef26e11335b47f8e7d`，旧库存V33清单SHA `f3649828c812a19bcc81371c3f1960239fc6c8572962afd61c6bf7be127aebe8`。使用原cached四ZIP `results/fleet-schema-v33/raw/eia8602020.zip`至`eia8602023.zip`，不新下载或覆盖。原V33实现/数据/证据保持不变。

四年沿用已核公开日期2021-09-09、2022-09-22、2023-09-19、2024-09-23；后续每月m选m.year−2且release≤m。ZIP仍为检索版本，未认证首发历史字节。登记BA库存不等于EIA930报告membership，不是物理可用出力上界。

## 与V33完全相同的整理规则

复用V33的字符串XLSX reader、码表、标识解析、容量转换及单区域汇总函数。新的纯curate函数只将源白名单参数化为旧7或固定17两种合法集合，使用相同处理顺序；不修改或monkeypatch旧模块SOURCES，不通过BA假名绕过源过滤。

Plant只取Plant Code/BA Code，重复plant失败。Generator仍只取V33白名单字段；先Plant→BA连接，非17域/空BA/未匹配plant只计metadata，不读取其容量、status或fuel。选中机组先查Plant Code+Generator ID唯一，再查status/prime mover/primary fuel。OP主口径，OP/SB/OA/OS为Operable敏感性。

储能主能源MWH或PM在BA/CE/CP/FW/ES/PS先排除，不转换容量。其他PM、年度能源码、H2年份限制、WH/other归类、主燃料只计一次、次燃料/多燃料flag仅审计，全部继承V33固定码表。选中非储能容量须有限非负，空白/未知/负/inf/nan失败，真实0保留。不得因为某新域不便而改fuel mapping或默认other。

每年每域须有非储能OP且总容量>0；若没有，整次失败并保留attempt，不能生成假零表。POS库存可为0，明确anchor_eligible=false；不填epsilon、不把它解释为当地无火电。此阶段不读取发电值，因此不能宣称已检查零库存与正发电的矛盾。

旧七源/固定17模式的纯函数应在合成样本上验证与V33同规则。实际17域结果中，旧七源的unit/excluded/op/operable子集按原schema和顺序序列化后，必须分别与旧四个CSV字节精确相同；若不一致则整次失败，不改旧结果迁就。

## 输出、一次运行与验证

五个新代码/测试/合同、V33复用实现、输入与两份独立code/Python-math预审固定后prepare。新目录`results/donor-inventory-v41`exclusive。curate在任何新容量转换前exclusive写started收据；其存在即拒绝重跑，进程状态须看真实handle。四年全部成功并通过旧七源字节比较后，才写数值表。

输出capacity_op/operable各68行、17域units.csv、无容量字段的储能excluded.csv、audit.json、legacy_match.json，加freeze/started/completed共9根文件。audit按年记全部metadata排除、选中/status/fuel/storage计数和范围限制；zero/multifuel计数保留在两policy表的各源年行，与V33 schema一致。legacy_match绑定四旧CSV和重建子集hash。选域/删域不能依赖CIF表现，除上述预锁有效性失败外不进行筛选。

独立复核以固定码表、原ZIP字符串表重建新十域和旧七域，校验全部机组/排除/68行两口径/账目及四个旧CSV子集字节；独立聚合可用Decimal并固定float比较容差atol1e−9/rtol1e−12，计数/分类/标识/旧子集必须exact。只为本阶段写短结果报告与薄清单，不新增大型auditor或重跑旧阶段。实际输出和本轮输入首尾hash不变。

本阶段通过只说明登记库存整理可信，不证明发电报告可比、预测改善、源数量因果效应或方法创新。后续另锁源监督队列、固定总权重和同预算强基线，比较旧池与扩展池整包价值。原七outer/35评分不变，未知区资格仍关闭，长期目标active。
