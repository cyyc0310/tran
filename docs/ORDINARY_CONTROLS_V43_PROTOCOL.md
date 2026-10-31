# V43：固定混合与扩展范围回退对照

2026-10-27，Sol，生成新组合前固定。父清单`docs/evidence/donor-budget-v42-verification.json` SHA256 `059b97ed4cc7013f6634ca053f31560276dda20ca2b55917c099cdd48316bc67`。V42的42个拟合、原MG、7个开发评估地区、42查询/128粒子和35评分月不变。无新拟合、无新fine标签，不读PACE/LGEE、真正未查看地区、原小时或新增供体2025细分。七地区反复用于开发，本轮不具独立确认性。

## 固定三十个混合对照

十对端点按下列顺序取左L、右R，每对固定w=.25/.5/.75，EF=(1−w)L+wR，随后逐粒子CIF=EF*原MG_POS。不能先混合点估计或中位数，不裁切/重投影，也不按成绩选权重。端点已保存在V42/V34。

| 对名 | 左端点L | 右端点R |
|---|---|---|
| Pool-SizeCapET | Legacy-SizeCap-ET | Expanded-SizeCap-ET |
| Pool-DirectEFET | Legacy-DirectEF-ET | Expanded-DirectEF-ET |
| Capacity-SizeCapET | Expanded-Size-ET | Expanded-SizeCap-ET |
| Capacity-DirectEFET | Expanded-Size-ET | Expanded-DirectEF-ET |
| StrongB0-SizeCapET | Legacy-Size-Ridge | Expanded-SizeCap-ET |
| StrongB0-DirectEFET | Legacy-Size-Ridge | Expanded-DirectEF-ET |
| Anchor-SizeCapET | Capacity-OP | Expanded-SizeCap-ET |
| Anchor-DirectEFET | Capacity-OP | Expanded-DirectEF-ET |
| Router-SizeCapET | CoalOilUpper | Expanded-SizeCap-ET |
| Router-DirectEFET | CoalOilUpper | Expanded-DirectEF-ET |

新臂命名`Mix-{对名}-w{w}`，数值字符串0.25/0.5/0.75。旧CoalOilUpper按V36原weights第8行，以旧六源同队列q范围、旧Size-Ridge及Capacity-OP粒子重建；两端点直接复制旧数组后按0/1选择。旧q从V42 training_q中按旧源白名单投影，旧180键保持已在父轮验证；每outer旧154/FPL156条。原CoalOilUpper等已有参考case直接复制，不以重建替换旧评分。

## 固定五个扩展范围回退对照

只改变V36经验范围的计算池：用V42同outer排除后的合格训练q（386/FPL388，15域，非候选17域全部库存），query用V34原SizeCap特征的q后缀。复用封存V36.weights的Range4/CoalUpper/OilUpper/CoalOilUpper/AnchorEFUpper五规则、tol1e−12，以旧Size-Ridge为L、原Capacity-OP为R。新名`ExpandedRange-{规则}`。保留每种规则、训练q范围及每query权重。不能将经验极值叫统计支持或保证，不能将登记容量当硬出力上界。原全局五权重已在父轮参考中，无需重复另加同臂。

三十混合加五回退共35新臂。B0/B1与少供体/扩供体的区别保持；某组合使用较少源信息不构成原创方法。保留原最强B0 Size-Ridge和CoalOilUpper，不能仅用退化的Expanded Size-ET作强参考。

## 数据、程序与执行

固定5个新producer/测试/合同文件、两份最终预审以及父清单和所需预测、cohort、旧case/support、原月表及实际调用helper代码的hash。只读取父链manifest元数据来解析所需别名，校验本轮直接依赖；不重复对旧899数据文件做hash或遍历数值。实际调用旧helper必须绑定到父链的对应代码，不能静默接受改写。

exclusive目录`results/ordinary-controls-v43`。prepare/predict/score各执行一次，日志exclusive。报告文件存在、非空且两最终SHA核对后才启动prepare。prediction_started先于数组消费；七地区全部预测/route/范围先seal并校验完整grid、shape、EF bounds及MG乘法，再exclusive写score_started并读取原35个评分标签。score_started已有时拒绝评分重入，completed绑定其hash。任何失败保留attempt，不因观察超时重启进程。

新增预测7组NPZ/JSON，35×6×128 EF/CIF；route1470行、范围7行。原V42 2520 case numeric exact复制，加35×2×35=2450，合计4970 case、994 region、142 macro、994去单地区和42 support。原各端点与强参考仍在父2520中。mean/median全部报，地区先平均再7地区等权；RMSE仍为各地区RMSE平均。

独立临时公式复算所有权重/粒子/新2450 case和全部聚合，原2520 numeric exact；atol1e−9/rtol0，布尔route和键精确，无新训练。校验本轮绑定首尾不变，禁止重跑旧fit或新增大审计框架。

## 判断与后续

首先判断V42的树模型局部收益是否可由普通全局收缩/锚定解释，及扩大经验范围是否使回退失效。报告逐地区和去地区敏感性，不能只展示最低macro臂，也不能以CIF误差抵消掩盖EF偏差。

即使某组合更好，也只报告普通对照收益。它不提供容量增量的可迁移适用条件、原创方法或区间覆盖证据。若混合仍不能稳定同时改善内部EF和CIF，下一步应优先检验源外B1增益方向能否由合法输入预测；诊断须另锁outer+inner排除设计，不由本轮分数临时造gate。方法冻结、未查看地区及小时/区间保证资格均不改变，长期目标active。
