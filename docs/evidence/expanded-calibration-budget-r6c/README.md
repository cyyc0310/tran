# R6c源校准日历反事实证据

对应[R6c结果](../../EXPANDED_CALIBRATION_BUDGET_RESULTS.md)及[事后开发协议](../../EXPANDED_CALIBRATION_BUDGET_PROTOCOL.md)。在同一批已查看的2400目标小时上，只改变两个预定源校准日历；模型和点预测不变，没有新方法或独立国家确认。

`predictions.csv.gz`保存331200条旧日历反事实；`results/`含126条源半径、同点/中位数核对、全部指标、配对差及3510输入/4预测封存。与[R6a证据](../expanded-anchor-no-issued-r6a/README.md)的全4824行带标签队列按country、region、origin、valid_at、label_row绑定即可复算；新旧源校准的完整raw组件/原始成员也在R6a包中，不重复存放。

`independent_review/`保留旧日历精确选行、126项源校准、144旧R5参数回验、全量预测重建和评分复算。两条重组的36条旧半径来自R5封存；新加法/logit和固定锚的旧半径按同旧源行计算，不声称在R5中已有。点/中位数保存CSV读取后与R6a逐值一致。

扩展校准没有普遍可靠缩宽：HGBB与加法CQR缩宽同时降低覆盖，原联合重组提高覆盖伴加宽。该政策同时改变校准样本量和月份/日期组成，不能单独归因于样本量；不是目标历史月量适配收益，也不是算法创新。

manifest绑定本包文件与原始路径/SHA。静态装机代理、固定因子发电侧标签、历史档案时效模拟及未完成的完整天气阶段保持原有边界。
