# V37：同B1预算的状态相关容量解码基线

2026-10-27，Sol，真实拟合前固定。V36的容量范围fallback已有点预测增量，但仍是普通规则、收益集中于已查看源。本轮用三个普通概率回归头检查容量offset、状态信息与目标函数的作用，不预告新方法、风险识别或未知区泛化。

## 固定数据与三头

父清单`docs/evidence/capacity-routing-v36-verification.json` SHA `a7d9466339a40a3db9ffd80747530553e35c3ec2d34d68187041f26789c80aee`，完整闭包绑定。复用V34 SizeCap-Ridge同outer的raw X36、四比例y、训练keys/时点/六源均衡权重和查询特征；不调用旧fit_predict重拟合旧模型。每outer仍154例，FPL outer156，完整源池180。原42起点/128粒子、35评分月、库存origin.year−2及已登记发布日期均不变。

三臂固定为PlainSoftmax36、OffsetSoftmax36、OffsetInterceptOnly。前两臂获得完全相同的36维输入，包括容量份额与压力，以训练权重计算同一standard scaler，再加截距。常数头只加截距，不使用状态。gas logit固定0，coal/oil/other三类为自由参数。

Plain以softmax(η)解码。Offset以softmax(log q+η)解码，q为登记库存的POS四比例；常数头使用同一offset。Plain不是故意缺少容量特征的弱对照。状态logit是相对倾向，不是实际或绝对利用率，q的零也不认证物理缺席。

## 损失与优化

\[
L(\Theta)=-\sum_iw_i\sum_{f:y_{if}>0}y_{if}\log p_{if}(\Theta)
+\tfrac12\|\Theta\|_F^2.
\]

使用加权和、λ=1，惩罚全部自由参数，包括截距；不能改成平均损失仍沿用λ=1。三头均从0初始化，解析梯度为设计矩阵转置乘w(p−y)的自由类列，再加Θ。采用SciPy L-BFGS-B，maxiter=10000、maxls=50、ftol=1e−15、gtol=1e−9，无超参数搜索。

对容差内行和不恰为1的浮点输入，梯度按实际CCE使用p×sum(y)−y，以保持与所算目标的精确导数一致；不重归一化标签。严格simplex时恢复上式。

Offset仅在q>0处计算log q；y=0项不取log、不乘0×负无穷。q=0且y=0合法并保持输出零；q=0且y>0报告具体训练key/fuel并失败，不能补epsilon或删行。Plain无该输出支持限制。两类输入都要检查有限非负simplex，source/日期检查先于训练数值使用。

记录solver成功/状态/message/nit/nfev/njev、Θ、scaler、惩罚/未惩罚损失、完整梯度及L2/∞范数。要求success、全有限且gradient L2≤1e−5，否则失败不fallback。λ=1使全部自由参数上的目标强凸，可记录gradient L2²/2为目标值距最优值的上界；这只是训练优化证书，不是目标预测误差或跨区保证。

## 预测与比较

七outer×三头共21新fit；预测四比例后乘既定EF，再逐粒子乘原MG_POS。均值点为乘积均值，median从粒子计算；不把只有粗组成变化的粒子当完整风险分布。全部七源、42查询的预测和参数seal完成前不读取2025评分标签。

评分原值复制V36的700 case对照行，加入三新头×两摘要×35月的210行，共910 case、182 region、26 macro、182去单源敏感性。基线不重拟合、不改摘要或挑w。所有CIF/EF MAE、平方/有符号误差、各源RMSE均保留，先各源平均再七源等权；去MISO等敏感性只作描述，不反馈模型或选源。

重点是OffsetSoftmax36对同CCE的PlainSoftmax36和OffsetInterceptOnly，以及V36完整点误差权衡。若不胜plain就没有offset增益证据；若不胜常数头就没有状态增益证据；若只靠MISO或CIF错误抵消，就不能据此主张普遍可迁移机制。三个头、两摘要完整展示，不设置新方法冻结或未知区授权。本轮仍不评价CIF区间覆盖、宽度或小时风险。

## 执行与核验

固定五个代码/测试/合同、两份独立code/Python-math预审与环境后prepare，输出exclusive，失败attempt保留。预测开始前写收据，完成七源后seal；评分支持须与V36原42行逐字节相同、35个EF已定义。输出参数足够独立重算目标/梯度和全部粒子，原输入首尾哈希不变。

独立结果复核以相同封存源数据核对cohort、凸目标/梯度证书、已存参数的概率/粒子/评分为主，无需再建大型auditor。固定参数重算与评分数值容差1e−9、rtol=0，原对照数值精确保留。任何独立优化比较须区分收敛容差与固定参数重放，不能按结果放宽容差。零容量掩码、名义库存/运行边界和未认证首发历史字节限制继续保留，旧目标/小时/真正未知地区继续关闭。
