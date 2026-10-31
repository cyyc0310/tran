# V33：七源公开装机库存的数值整理合同

2026-10-26，Sol，真实容量转换前固定。此阶段只整理独立预算B1的库存代理，不拟合CIF、不打开旧目标或真正未知地区。B0仍为原公开月报；B1增加库存信息，数据增益与方法创新须分开判断。

## 来源与版本

绑定V32研究清单 `docs/evidence/history-signature-v32-verification.json` SHA `8e3cbfe354deab91e4f6f42e518f8340677c075339a88a55558791fe1d081a4a`，以及字段阶段清单 `docs/evidence/fleet-schema-v33-acquisition.json` SHA `bfd20570f0b4c58e05108c93a1aaa41b0a95874574fd8e32e116626596ee3398`。两者完整现有闭包、四个原ZIP及字段审计记录保持绑定，不重复下载或改文件。

读取2020—2023各档内Plant和Generator的Operable工作表。只选BPAT、CISO、ERCO、FPL、ISNE、MISO、NYIS；不得转换其他BA、空BA或无法连接plant的容量。Proposed／Retired工作表不作库存，多燃料表不另累加容量。

四年的已核公开日期分别为2021-09-09、2022-09-22、2023-09-19、2024-09-23。后续按origin年份减2取唯一年度，同时验证该年度存在且发布日期≤origin；不能取最近可用年补缺。2022／2023／2024／2025起点分别对应2020／2021／2022／2023。日期型origin按UTC日零点，带时间的origin须有显式时区；这些固定选择与发布日期至少相隔数月，不声称认证发布日的具体小时。当前ZIP是检索版本，尚未认证为首发历史字节；表头一致也不等于历史语义完全一致。

## 连接与访问顺序

XLSX读取只解码白名单字段为字符串，不执行公式、宏或外部链接，不直接把numeric cell转float。Plant只取Plant Code与Balancing Authority Code。Generator只取Plant Code、Generator ID、Prime Mover、Status、Nameplate Capacity (MW)、Energy Source 1至6、Multiple Fuels?、Cofire Fuels?、Switch Between Oil and Natural Gas?。不读取发电量、排放、热耗、最低出力、未来退役／改造容量等数值。

先建立Plant Code→BA的一对一metadata映射，重复Plant Code即失败，不能first去重。Plant Code作为正整数标识规范化，Generator ID仅去首尾空白、保留前导零和大小写。按BA筛到七源后，再检查该年内Plant Code＋Generator ID唯一；重复在状态／储能排除前拒绝。不可按州、所有者或位置代替BA，不可把CA／CT／CS等联合循环部件按名称合并。

非七源记录只计排除条数，不转换其容量。无法连接plant或空BA的generator计metadata不确定性，不分配给任何源，不读取其容量，也不把它们默认为当地零容量。整理范围是已登记BA的库存代理，不能认证其覆盖完整发电辖区。

## 预锁归类与质量规则

主口径为Status=OP；Operable敏感性明确为OP／SB／OA／OS。Operable表内遇RE或其它状态失败。先按储能排除：Energy Source 1=MWH，或Prime Mover在BA／CE／CP／FW／ES／PS中；只记七源排除metadata，不转换这些行的容量。CP指太阳能储热，PS指抽蓄；WAT或SUN不能绕过该排除。

其余Prime Mover只允许ST／GT／IC／CA／CT／CS／HA／HB／HK／HY／BT／PV／WT／WS／FC／OT。CC官方用于尚无细分机组信息的规划总单元，若在本Operable口径遇到则失败；未知码也失败。

每台机组按Energy Source 1一次分配，不按多燃料能力重复算容量。四年Layout的Reference Table 28支持以下研究聚合，年度限制也固定：

| 组 | 允许主能源码 |
|---|---|
| coal | ANT BIT LIG RC SGC SUB WC |
| gas | NG |
| oil | DFO JF KER PC PG RFO SGP WO |
| nuclear | NUC |
| hydro | WAT，已先排PS |
| wind | WND |
| solar | SUN，已先排储能PM |
| other | AB BFG BLQ GEO LFG MSW OBG OBL OBS OG OTH PUR SLW TDF WDL WDS WH；H2仅2021—2023 |

未知／空主能源码失败，不能默认other。WH包括部分来源不明或无补燃联合循环蒸汽量，不能无证据重分给NG；与B0同名组也不证明口径完全一致。次能源码与多燃料／混烧原flag仅保留审计，不增加容量或进入模型。multifuel_flagged_count按“至少一个非空且不同于主能源的次能源码，或三个原flag之一为Y”计机组数，不能解释成实际混烧量；其它原flag值保留原文而不猜测。

筛选后才将非储能Nameplate Capacity转float，须有限非负；空白、非法、负值或NaN／inf失败，不用skipna、零或epsilon填补。真实零值保留，并报告zero-capacity计数。每年七源均须有至少一台合法非储能OP、OP总容量>0，否则失败，不捏造缺区零表。

在已登记范围内，某燃料没有合格机组可记observed容量0，但必须随行保留机组数与范围说明；这不证明该地区实际无该燃料。若POS库存总量为0，标记anchor_eligible=false，不能把分母补epsilon。只整理nameplate，不读取夏冬容量或最低出力来增加隐含预算；nameplate不是未来认证出力上限。

## 输出与运行

五个producer／tests／protocol文件及三最终独立预审固定后prepare。`results/fleet-inventory-v33/`使用exclusive写；失败保留，不覆盖。curate在读取容量前exclusive创建curation_started收据，存在该收据即拒绝再次curate，不能因还没有数值表就重跑失败attempt。它不代表进程仍活跃，恢复须另查真实handle。curate读取四档，不碰月报隐藏标签或任何CIF值；全四年成功后再写完整输出与completion。

输出：`capacity_op.csv`、`capacity_operable.csv`各28行、七源合法非储能`units.csv`、七源储能metadata的`excluded.csv`、`audit.json`，再加freeze／curation_started／completion共八个根文件。unit表在两个口径共用，不重复机组；excluded无已转换容量字段。八组总量、POS总量、机组数及零值／多燃料计数可审查，但尚不产生预测。

audit记录每年plant／generator行数、非七源／空BA／无法连接plant数量、七源选中和储能排除数量、各状态／燃料的已整理计数，以及所有归类／范围标志。记录不完整边界，不能将未知成员隐入任何源的零容量。

独立核验先检查固定收据、代码、父闭包、四ZIP路径／hash、当前环境与八文件库存，再用独立ZIP/XML读取和既定码表重建七源机组、两套28行容量及所有账目。核验不得依赖producer的分类或聚合函数，首尾hash不变。当前阶段状态只可称数据整理已核，不称预测或覆盖成功。

## 后续方法边界

库存只供软特征、简单装机比例锚及同B1普通模型比较。多燃料、停机、容量评级、未匹配BA及与发电代理的范围差异均保留，不能据此硬截小时CIF。仅煤气分母对应POS月量时只能称部分容量归一化代理，不能将大于1自动clip。整理通过后另锁模型实验、同队列与强参考规则；未知地区仍关闭，长期目标active。
