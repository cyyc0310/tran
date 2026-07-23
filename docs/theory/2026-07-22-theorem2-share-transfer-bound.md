# Theorem 2:Term① 份额迁移误差的域适应上界(推导草稿)

**日期:** 2026-07-22
**状态:** **推导已完成**(见第 2–4 节,严格证明,只用 L1 损失的三角不等式 + discrepancy 定义)。**数值估计未完成且短期内不打算做**(discrepancy / Wasserstein $W_1$ / 理想联合风险 $\lambda^\*$ 三个量在本项目均未估计,原因见第 6 节)。因此本定理与 Theorem 1 性质完全不同:Theorem 1 是可精确验证到浮点精度的**代数恒等式**,Theorem 2 是一个**结构性上界**,其右端项本项目只做了定性解释、未做数值标定。本文档诚实标注这一区别,供续接会话与论文写作引用。

本文档承接:

- `docs/theory/2026-07-14-theorem1-error-bound-draft.md` 第 3 节("Theorem 2:可选/次要,尚未详细推导")——该节只提出思路;
- `docs/paper/2026-07-17-transcif-workshop-draft.md` 第 5.7 节 / Limitations / Conclusion——三处均写明 "Theorem 2 is not yet derived / sketched but not completed"。

本文档把那个"思路"补成一份可复核的证明,并把它与 Theorem 1、Corollary 1、以及 §5.7 的 Bates-Granger 融合结果挂钩。**沿用 Theorem 1 与论文草稿的全部记号,不引入与之冲突的新符号。**

---

## 0. 为什么写这份推导(决策背景)

Theorem 1 把单步 CI 迁移误差精确分解为

$$CI_{pred,t} - CI_{true,t} = \underbrace{(\hat s_t - s_t)(C_{\text{renew}}-C_{\text{nonrenew}})}_{\text{Term①(迁移放大)}} + \underbrace{(\hat\Delta_t - \varepsilon_t)}_{\text{Term②(残差估计)}},$$

并在四个真实 AU 地区、8/8 组合上验证了 **Term① 主导**(占比 77.8%–91.8%,见汇总文档第 4.4 节)。这留下一个未回答的理论问题:

> Theorem 1 把误差归到了 Term① 的 $|\hat s_t - s_t|$(份额预测误差本身),但**没有说明这个份额误差在"源域训练、目标域部署"的迁移设定下能被什么界住**。

Theorem 1 是"误差长什么样"的**分解**;Theorem 2 要回答"份额误差在域移下能被什么控制"的**上界**。这正是论文草稿反复标注为唯一未完成理论缺口的部分(paper §5.7:"a natural direction for a second theorem ... which we have sketched but not yet derived")。

**诚实前提(全文最高约束):** 本项目**没有**做过任何外部 SOTA 对比、**没有**估计过任何域散度、**没有**目标域全分布标签。因此本定理只能作为**解释性/结构性**结果,不能包装成"数值验证过的紧界"。凡涉及未估计的量,一律显式标注。

---

## 1. 设定与记号(与 Theorem 1、论文草稿一致)

固定一个预测步(与 Theorem 1 的单步设定一致;多步只需对 $h$ 逐分量重复)。

- **输入空间** $\mathcal{X}$:窗口化的历史电网信号(RenewShare / LoadNorm / 可选发电通道 / 可选温度)。
- **份额预测器** $h:\mathcal{X}\to[0,1]$:即 Stage-1 编码器,$\hat s = h(x)$。$h$ 取自假设类 $\mathcal{H}$(编码器可实现的全部函数)。
- **真实份额标注函数** $f:\mathcal{X}\to[0,1]$:$s = f(x)$ 为真实未来可再生占比;源域记 $f_S$、目标域记 $f_T$(允许两域标注函数不同,这是域适应的一般情形)。
- **份额损失** 取 **L1(绝对误差)**:$\ell(a,b) = |a-b|$,$a,b\in[0,1]$。选 L1 是因为本项目全程用 MAE 度量份额误差(见 §5.7 的 calib/eval MAE 均为 RenewShare 单位的绝对误差),且 L1 满足三角不等式,是下面证明的唯一结构性要求。注意 $\ell$ 在 $[0,1]$ 上有界:$0\le\ell\le 1$。
- **源/目标分布** $P_S, P_T$ 为 $\mathcal{X}$ 上分布($P_S$ 是 QLD1/NSW1/VIC1 窗口的混合,$P_T$ 是目标区窗口)。
- **份额风险(期望绝对份额误差):**
$$\varepsilon_S(h) := \mathbb{E}_{x\sim P_S}\,|h(x)-f_S(x)|,\qquad \varepsilon_T(h) := \mathbb{E}_{x\sim P_T}\,|h(x)-f_T(x)|.$$
- **物理常数** $L_T := |C_{\text{renew}}-C_{\text{nonrenew}}|$,精确可查(SA1 490.43 / QLD1 841.59 / NSW1 875.14 / VIC1 1160.12,gCO₂/kWh),来自 `EMISSION_FACTOR_TABLES`,**非估计值**。

---

## 2. Lemma 2.1(把 Theorem 1 的 Term① 桥接到份额风险)

**Lemma 2.1.** 目标域上 Term① 的期望绝对幅度精确等于 $L_T$ 乘以目标份额风险:

$$\mathbb{E}_{x\sim P_T}\big|\text{Term①}\big| \;=\; \mathbb{E}_{x\sim P_T}\big[L_T\cdot|\hat s - s|\big] \;=\; L_T\cdot \varepsilon_T(h).$$

**证明.** $L_T$ 为常数可提出期望;$|\hat s - s| = |h(x)-f_T(x)|$,取期望即 $\varepsilon_T(h)$。$\blacksquare$

这一步是**精确等式**(不是不等式),把 Theorem 1 里那个被实证确认为主导的项,直接翻译成学习理论里的标准量 $\varepsilon_T(h)$。下面只需给 $\varepsilon_T(h)$ 一个上界。

---

## 3. Theorem 2(目标份额风险的域适应上界)

沿用 Ben-David et al. (2010) 与 Mansour et al. (2009) 的经典域适应框架,针对 L1 损失给出。

**定义(discrepancy distance,Mansour et al. 2009).** 对损失 $\ell$ 与假设类 $\mathcal{H}$,
$$\operatorname{disc}_\ell(P_S,P_T) := \sup_{h,h'\in\mathcal{H}}\Big|\,\mathbb{E}_{x\sim P_S}\ell(h(x),h'(x)) - \mathbb{E}_{x\sim P_T}\ell(h(x),h'(x))\,\Big|.$$
它是 Ben-David 的 $\mathcal{H}\Delta\mathcal{H}$-散度对一般损失的推广:只依赖两域在 $\mathcal{X}$ 上的**输入分布**与假设类,**不依赖标签**。

**定义(理想联合假设).** 记 $h^\* := \arg\min_{h\in\mathcal{H}}\big[\varepsilon_S(h)+\varepsilon_T(h)\big]$,并令
$$\lambda^\* := \varepsilon_S(h^\*) + \varepsilon_T(h^\*).$$
$\lambda^\*$ 度量"同一个假设能否在两域同时做好"——它是域适应**不可约的**部分(若两域标注函数根本不相容,$\lambda^\*$ 大,任何迁移都受限)。

**Theorem 2.** 对任意份额预测器 $h\in\mathcal{H}$,在 L1 损失下,
$$\boxed{\;\varepsilon_T(h) \;\le\; \varepsilon_S(h) \;+\; \operatorname{disc}_\ell(P_S,P_T) \;+\; \lambda^\*\;}$$

**证明.** 全程只用 L1 损失的三角不等式与上述定义。

1. 目标域,对份额加减 $h^\*(x)$ 并用三角不等式:
$$\varepsilon_T(h)=\mathbb{E}_{T}|h-f_T|\le \mathbb{E}_{T}|h-h^\*| + \mathbb{E}_{T}|h^\*-f_T| = \mathbb{E}_{T}\,\ell(h,h^\*) + \varepsilon_T(h^\*).$$
2. 把目标域上的 $\mathbb{E}_{T}\,\ell(h,h^\*)$ 换到源域,误差不超过 discrepancy(因 $h,h^\*\in\mathcal{H}$,受 $\sup$ 支配):
$$\mathbb{E}_{T}\,\ell(h,h^\*)\le \mathbb{E}_{S}\,\ell(h,h^\*) + \big|\mathbb{E}_{T}\ell(h,h^\*)-\mathbb{E}_{S}\ell(h,h^\*)\big| \le \mathbb{E}_{S}\,\ell(h,h^\*) + \operatorname{disc}_\ell(P_S,P_T).$$
3. 源域,再次三角不等式:
$$\mathbb{E}_{S}\,\ell(h,h^\*)=\mathbb{E}_{S}|h-h^\*|\le \mathbb{E}_{S}|h-f_S| + \mathbb{E}_{S}|f_S-h^\*| = \varepsilon_S(h)+\varepsilon_S(h^\*).$$
4. 合并 1–3:
$$\varepsilon_T(h)\le \varepsilon_S(h)+\varepsilon_S(h^\*)+\operatorname{disc}_\ell(P_S,P_T)+\varepsilon_T(h^\*) = \varepsilon_S(h)+\operatorname{disc}_\ell(P_S,P_T)+\lambda^\*.\qquad\blacksquare$$

这是标准结果(Ben-David 2010 Thm 中 L1/discrepancy 版本),**证明本身无新意**,新意在下一节与物理常数的耦合。

---

## 4. Corollary 2(物理耦合的迁移误差上界:本文档的实际贡献)

把 Lemma 2.1 与 Theorem 2 相乘,再叠回 Theorem 1 的 Term②,得到目标域整条管线单步误差的期望上界:

**Corollary 2.**
$$\mathbb{E}_{T}\big|\text{Term①}\big| = L_T\,\varepsilon_T(h)\;\le\; L_T\big(\varepsilon_S(h)+\operatorname{disc}_\ell(P_S,P_T)+\lambda^\*\big),$$

$$\boxed{\;\mathbb{E}_{T}\big|CI_{pred,t}-CI_{true,t}\big|\;\le\; L_T\big(\underbrace{\varepsilon_S(h)}_{\text{源域可训练}}+\underbrace{\operatorname{disc}_\ell(P_S,P_T)}_{\text{域对齐(CORAL)}}+\underbrace{\lambda^\*}_{\text{目标微调触及}}\big)\;+\;\underbrace{\mathbb{E}_{T}|\hat\Delta_t-\varepsilon_t|}_{\text{Term②}}\;}$$

(末步用 $\mathbb{E}_T|A+B|\le\mathbb{E}_T|A|+\mathbb{E}_T|B|$,即对 Theorem 1 恒等式取期望+三角不等式。)

**这一步才是本文档相对纯教科书结果的增量:** 标准域适应界给的是 $\varepsilon_T(h)$;这里把它左乘**精确可算的物理常数 $L_T$**,于是:

- 迁移误差的 CI 单位上界**显式正比于 $L_T$**——这正是 Corollary 1 里那个"排放系数差"常数。Theorem 2 因此给 §5.3 观察到的"term1_share 随 $L_T$ 单调递增"提供了一个学习理论读法:草稿(§5.3 发现2)已诚实指出该单调性"接近代数副产物",Corollary 2 进一步说明——只要各地区 $\varepsilon_T(h)$(即 $|\hat s - s|$ 的量级)相近,CI 单位迁移误差就线性受 $L_T$ 支配,这与 8/8 Term① 主导的结构性发现一致。
- 三个右端项对应三种可操作干预:$\varepsilon_S(h)$ 靠源域训练压低;$\operatorname{disc}_\ell$ 靠特征对齐(CORAL,方向 E)压低;$\lambda^\*$ 只能靠**接触目标域标签**(监督微调,方向 D)去逼近 $h^\*$。这与实验结论吻合(见第 5 节)。

---

## 5. 与已有真实结果的定性一致性(不是数值验证,是机制解释)

Theorem 2 的价值在于**解释已观测到的真实现象**,而非产生新数字。下列一致性均为定性对照,数据全部来自仓库 docs:

1. **为什么单独 CORAL(E)无效、单独/配合微调(D)有效(paper §5.5,真实数据):**
   - 方向 D(渐进解冻监督微调)直接用目标校准集标签把 $h$ 推向 $h^\*$,压的是 $\lambda^\*$ 那部分 slack——对应 §5.1 观测到的 **Term① 下降 12.4%**(84.413→73.966)。
   - 方向 E(CORAL)只对齐源/目标特征二阶统计量,压的是 $\operatorname{disc}_\ell$,**完全不碰 $\lambda^\*$**。Theorem 2 说明:若 $\lambda^\*$(两域标注不相容度)才是瓶颈,只压 disc 不足以降 $\varepsilon_T$,甚至可能把 $h$ 拉离对预测有用的方向——这与"E 单独使用有害(+12.2%,比基线更差)、只有与 D 结合才起正则作用"的真实结论方向一致。
   - **诚实边界:** 这是机制层面的**一致性论证**,不是"Theorem 2 预测了 E 无效"。我们没有估计 $\operatorname{disc}_\ell$ 或 $\lambda^\*$ 的实际数值,无法定量断言瓶颈就是 $\lambda^\*$。

2. **与 Bates-Granger 融合(paper §5.7,真实数据)的关系:**
   - §5.7 融合的三个份额预测器,其个体份额风险即本文的 $\varepsilon(h)$,在 RenewShare 单位下的真实 calib/eval MAE 为:persistence 0.1549/0.1396、network(D+E) 0.1644/0.1503、climatology 0.2018/0.1840(汇总文档 §4.1)。
   - 融合的操作意义正是:构造一个组合预测器,使其**有效 $\varepsilon_T$ 低于任一单分量**,从而经 Lemma 2.1 放大后得到更小的 $L_T\varepsilon_T$——这解释了融合何以是全项目跑赢持久性基线幅度最大(−9.4%)的结果。
   - **诚实边界(与草稿完全一致):** 论文草稿把"把 Bates-Granger 权重结构连到闭式 Ben-David 界"列为**未完成的猜想**。本文档**没有**证明拟合出的最小方差权重等于任何界最优权重;Corollary 2 只解释了"降低有效 $\varepsilon_T$ 会降低 CI 迁移误差上界"这个方向,不声称权重层面的对应。这一猜想仍开放。

---

## 6. 未数值估计项(强制诚实标注)

Theorem 2 / Corollary 2 右端三项,在本项目的估计状态如下:

| 项 | 含义 | 本项目是否估计 | 原因 |
|---|---|---|---|
| $\varepsilon_S(h)$ | 源域份额风险 | **部分可得** | 训练过程可测源域份额 MAE,但未作为"界的一项"单列报告 |
| $\operatorname{disc}_\ell(P_S,P_T)$ | 域间 discrepancy | **未估计** | 需在假设类上求 $\sup$ 或做代理 A-distance / OT 计算,本项目无此实现 |
| $\lambda^\*$ | 理想联合风险 | **未估计** | 需目标域**全分布**标签求 $h^\*$;本项目只有目标校准集(70%切分),无法估计 $\lambda^\*$ |
| $W_1(P_S,P_T)$(见附录) | Wasserstein-1 | **未估计** | 无最优传输/Sinkhorn 计算 |

**因此必须如实表述:**

- Theorem 2 是**结构性上界**,不是像 Theorem 1 那样"验证到 5e-5 浮点精度"的恒等式。
- 论文中应写为:"我们**完成了** Theorem 2 的推导(把 §5.7 里 sketched 的方向补成严格证明),但**未对其右端散度项做数值标定**;它作为解释 Theorem 1 迁移项、以及 D/E/融合三类干预机制的理论框架,而非一个经验紧界。"
- **不得**把 $L_T(\varepsilon_S+\operatorname{disc}+\lambda^\*)$ 当作可与实测 MAE 逐点比较的预测值——右端两项未知,界的松紧未知。

---

## 7. 附录:Wasserstein 特化(可选,进一步降低对假设类的依赖)

若"经 $\ell$ 复合后的假设"关于输入是 $L_g$-Lipschitz(即 $x\mapsto\ell(h(x),h'(x))$ 一致 Lipschitz,常数 $L_g$),则由 Kantorovich–Rubinstein 对偶(Redko et al. 2017 域适应版本):

$$\operatorname{disc}_\ell(P_S,P_T)\;\le\; 2L_g\,W_1(P_S,P_T),$$

代入 Corollary 2:

$$\mathbb{E}_{T}\big|\text{Term①}\big|\;\le\; L_T\big(\varepsilon_S(h)+2L_g\,W_1(P_S,P_T)+\lambda^\*\big).$$

**诚实标注:** $W_1$ 与 $L_g$ 同样**未估计**。此特化仅说明界可写成对输入分布 Wasserstein 距离的形式,便于将来若引入 OT 计算时直接套用;当前不构成任何数值结论。

---

## 8. 当前状态与下一步

**已完成:**

- Lemma 2.1(Term① 期望 = $L_T\varepsilon_T(h)$,精确等式)。
- Theorem 2(L1 损失下 $\varepsilon_T\le\varepsilon_S+\operatorname{disc}+\lambda^\*$,严格证明,逐步可复核)。
- Corollary 2(物理耦合上界,含 Term②),及其与 Corollary 1 的 $L_T$、§5.3 单调性、§5.5 的 D/E 机制、§5.7 融合的定性一致性论证。
- 未估计项的强制诚实标注(第 6 节)。

**未完成(后续可选):**

1. 数值标定 $\operatorname{disc}_\ell$ / $W_1$ / $\lambda^\*$ 中任一项(需新增 OT 或代理 A-distance 计算、或目标域更多标签),才能把 Theorem 2 从"结构性上界"升级为"经验界"。优先级:低——Term① 主导(Corollary 1)已是站得住脚的发现,Theorem 2 作为解释框架已足够,不必为凑"验证过的界"而勉强估计。
2. 证明或证伪 §5.7 的猜想(Bates-Granger 最小方差权重 ↔ 界最优权重的对应)。这是真正开放的数学问题,非本次推导范围。

**写入论文的建议改动(供 paper 任务参考,本次不改论文):**

- 把 §5.7 末句、Limitations 第 5 条、Conclusion 里 "Theorem 2 is not yet derived / sketched but not completed" 改为 "Theorem 2 is now derived as a *structural* bound (Appendix / Section X); its divergence terms $\operatorname{disc}_\ell,\lambda^\*$ are **not numerically estimated**, so it functions as an explanatory framework rather than an empirical bound."
- 参考文献需补:Mansour, Mohri & Rostamizadeh (2009), *Domain adaptation: Learning bounds and algorithms* (COLT);Redko, Habrard & Sebban (2017), *Theoretical analysis of domain adaptation with optimal transport*。Ben-David et al. (2010) 已在草稿参考文献中。

---

## 9. 相关文件

| 文件 | 与本推导的关系 |
|---|---|
| `docs/theory/2026-07-14-theorem1-error-bound-draft.md` | Theorem 1 恒等式(本定理的 Term① 来源)+ §3 提出的 Theorem 2 思路(本文档将其补全) |
| `docs/experiments/2026-07-17-all-experiments-summary.md` | 全部真实数字来源:$L_T$ 表、Term① 占比、§4.1 融合与 calib/eval 份额 MAE |
| `docs/paper/2026-07-17-transcif-workshop-draft.md` | §5.7 / Limitations / Conclusion 三处标注 Theorem 2 未完成,本文档对应补全 |
| `src/transcif/physics/cif.py` | $L_T$ 与 CIF 线性性(Lemma 2.1、Corollary 2 的物理常数来源) |
| `src/transcif/training/domain_adaptation.py` | 方向 D(逼近 $h^\*$,压 $\lambda^\*$ slack)/ E(CORAL,压 $\operatorname{disc}_\ell$)的实现,第 5 节机制对照的代码依据 |
| `scripts/theorem2_bayes_fusion.py` | §5.7 融合脚本(第 5 节第 2 点定性关联的对象;权重↔界的对应仍为开放猜想) |
