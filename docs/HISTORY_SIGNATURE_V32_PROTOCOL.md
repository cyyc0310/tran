# V32：十二条合法月报的分层信息检验

2026-10-26，Sol，运行前合同。依据 [设计](HISTORY_SIGNATURE_V32_DESIGN.md)，不再增加候选。这里只检验公开历史特征的信息价值，不预设原创性或概率覆盖。

## 数据与时点

父证据清单为 `docs/evidence/forward-context-v31-verification.json`，SHA256 `f6544892fc7bf24256cadd5809b3b9fcdfde60c4e42bde9bfa13db41ea9b46df`。完整闭包、原月表及七个 MG 预报档案保持绑定。七源仍为 BPAT、CISO、ERCO、FPL、ISNE、MISO、NYIS。

训练标签窗口仍为 2022Jan 至 2024Jun，月末与发布时间均不迟于部署截止 2025Jan1；每个 outer 排除整区隐藏标签。训练当前 G5 是已实现的源监督变量，不称历史前瞻 hindcast。2025Jan1 至 Jun1 每区六个月初查询，当前 G5 完全使用原 MG 128 粒子；不读取当前真实组成、总量或隐藏燃料作为特征。日期、整数时间戳与档案统一 UTC 纳秒。

训练 case 和 query 都取同区自 2022Jan 起、`start < origin`、`end <= origin`、`available_at <= origin` 的最新十二行完整月报，latest-first。选定后才转换公开数值；不足十二条的训练 case 记跳过，query 不足则整个 attempt 失败。选中数值非法直接失败，不能改选旧行。源训练当前 POS≤1e−12 的 case 统一跳过。所有十八臂共享最终训练队列；潜在 117 个 case 只是运行前元数据计数。

只读取历史四个 ZERO 份额及总发电量，POS 为其补集。总量须有限且正，`total_MW=total_mwh/calendar_hours`；`thermal_MW=total_MW*POS`，允许为零。这里的 thermal 指源数据 coal／gas／oil／other 代理组，不等于已认证的中国火电类别。当前总量不作当前 case 特征，但日后合法成为过去月报时可以使用。

## 固定变换

十二行可以不连续，不填补缺月。`month_age=origin_calendar_month−past_calendar_month`；`span=max(age)−min(age)+1`，`maxgap=max(diff(sorted(age)))`，单位均为月。三条旧 context 是这十二行的最新三条，G5／Trend／Size 的列顺序和定义沿用 V30／V31。

对 total MW 和 thermal MW 各计算六个描述量，按以下顺序保存：

1. `log1p(mean(v))`。
2. `quantile(v,0.1,method='linear')/mean(v)`。
3. `std(v,ddof=0)/mean(v)`。
4. 每日历月的归一化线性趋势 β，新近月份上升时为正。
5. 一阶谐波 cos 系数。
6. 一阶谐波 sin 系数。

令 `t=-month_age`、`tc=t−mean(t)`、`z=v/mean(v)−1`，`β=(tc·z)/(tc·tc)`。先得到 residual=`z−β*tc`，再用 `[1,cos(2π(month−1)/12),sin(2π(month−1)/12)]` 对 residual 作 `np.linalg.lstsq(...,rcond=None)`，取后两系数。若 mean(v)=0，后五项定义为 0，不人为增加 epsilon。该两阶段变换是描述性统计，不声称独立的趋势／季节分解。

保存谐波设计的 rank 与 condition；rank 不足允许最小范数解，但不能解释相位。诊断不隐式进入预测特征。PhaseScramble 在同一 cos／sin 系数对上保幅旋转：`[c',s']=[c cosθ−s sinθ,c sinθ+s cosθ]`。θ 由 `sha256('history-signature-v32-phase/'+region+'/'+origin.strftime('%Y-%m'))` 的前八字节、big-endian uint64 除以 `2**64` 后乘 `2π` 固定。逐 case 旋转，JointTrend 前缀逐位相同、系数范数相同；不增加显式地区 ID。该单次随机控制不是因果证明。

## 九组特征与模型

| 组 | 定义 | 维度 |
|---|---|---:|
| G5 | 原当前五组 | 5 |
| Trend | 原三条历史及趋势 | 30 |
| Size | Trend 加原最新 log MW | 31 |
| LongTotal | Size 加 span／maxgap、total 六描述量 | 39 |
| JointLevel | LongTotal 加 thermal 前三描述量 | 42 |
| JointTrend | JointLevel 加 thermal β | 43 |
| JointAmplitude | JointTrend 加 sqrt(cos²+sin²) | 44 |
| JointShape | JointTrend 加 thermal cos／sin | 45 |
| PhaseScramble | JointTrend 加保幅旋转 cos／sin | 45 |

每组分别训练 Ridge／ET，共十八臂；七 outer 共 126 次新拟合。Ridge alpha=1、fit_intercept，六个训练源等权，mean／std 只在训练源加权计算，std≤1e−12 置 1。ET 原始 X，128 棵、min_samples_leaf=2、max_features=1、seed17023、n_jobs=1。所有组同队列、同 y、同样本权重，不能复用 V31 的旧头冒充匹配对照。

预测隐藏 r4 后只作一次 simplex 投影，EF4 固定 `[1000,400,800,400]`，每粒子 `CIF=g_POS*EF`；CIF 点预测为 mean(product)。保存 raw／projected r、EF／CIF 粒子和点、MG、九组 query features、十二条 context 来源／数值、六描述量、相位角、rank／condition及十八个完整 fit metadata。

## 封存、支持与强参考

五个 producer／tests／protocol 文件及三份独立最终预审固定后 prepare。输出 exclusive，不覆盖失败。所有七区 42 个起点先预测封存；十四个普通非 symlink 预测文件与完整 seal 验证后，才读取评分真值及 Legacy 分值。

评分沿用 V31 `2025Sep1` 截止、whole-month metadata、coverage≥0.9、正 total、同完整八类代理目标；零 POS 保留 primary，内部 EF 为 NaN。42 行支持全部保留。当前十八臂共同评分；实际标签支持还须与父 V31 的具体 keys／status／有效 EF 及质量字段一致。不得为本轮放宽或重选。按既有数据预期是 35 月／630 case／126 region／18 macro／42 support，但最终以合法支持为准。

原 V31 Size-Ridge／Size-ET 是全三条历史训练队列的额外强参考，称 LegacySize。prepare 绑定父 `region_metrics.csv` 和 `label_support.csv`，预测全封后才读其数值；比较前核对相同月表、forecast paths、全部 origin 和精确标签支持，不能只凭地区均值判断同样本。它不与新共同队列的特征消融混同。

## 固定二十三对与判定

保留旧九对：每学习器 Trend−G5、Size−G5、Size−Trend 六对，以及这三组各 ET−Ridge 三对。每学习器另比较 JointShape 对 Size、LongTotal、JointLevel、JointTrend、JointAmplitude、PhaseScramble，共十二对。最后 JointShape 对同学习器 LegacySize 两对。合计二十三对、161 条地区 contrast，不补全两两比较或事后选赢家。

六个旧 feature 对比、十二个 JointShape 对比及两个 Legacy 对比均按主 CIF MAE 使用原规则：宏至少下降 5%、至少四区严格改善、任一区不超过参考 110%（容差1e−12）、MISO／ERCO均不恶化，参考宏须>1e−12。三个 ET−Ridge 只作描述，不判信号。

同学习器六个 JointShape 匹配比较全部通过，才记 phase_association；再通过其 LegacySize 比较，才记 practical_forward_signal。十八臂的其余分数完整报告，但不能临时改用其它组作新主方法。即使全部通过，也只说明预测关联值得研究；`mechanism_breakthrough`、`hourly_stage_authorized`、`new_holdout_authorized` 始终 false。

原 35 月已查看，ISNE 只有一个合格月。此轮是源开发，不称盲确认、煤电基荷识别或小时概率校准。后续仍须解释粗统计如何约束跨域持续风险，并超过简单收缩、标准动态／异方差等同预算参考。

## 独立核验

独立审计不导入 V32 producer。先核固定收据、完整父闭包、消费路径、正式环境、九根文件／十四预测文件，再读取数值；重建共同训练队列、十二条上下文、所有描述量与相位控制、126 个模型、逐粒子输出、支持、所有表及二十三对。raw 独立重现后若用已核 stored raw／point 保留端点舍入，必须明示分阶段范围。首尾 hash 不变；最终科学审核和证据清单之后封存本轮。长期目标保持 active。
