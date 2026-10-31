# V189空条件评分表修复与r1退役

2026-10-29。原正式r1完成fold0–4后，在fold5写出前因空油电条件任务没有DataFrame列schema而KeyError退出，无成功root summary。冻结V188的61个油电正月query确实全在fold0–4，fold5为0；不伪造0评分或更改条件任务。原预检r1未读取held真值，未覆盖实际条件表为空的归档分支。

保留r1全部五个已提交fold、日志和当时helper/runner/audit/协议快照`results/v189-failed-source-r1`；它们退役，不声称当前修改脚本hash可复现旧r1。r2仅显式固定评分列schema并让空finite检查转float数组，审计同schema处理零行表。新增不读取任何truth的空CSV serialize/readback检查。模型、样本、特征、源删留、概率、任务定义、指标和比较不变。

预检/正式全部重跑，输出分别preflight-r2和source-block-oil-r2，审计audit-r2；不复用或覆盖r1评分。488森林指纹和概率之后再验证与r1原样本块模型相同（只在存在完整旧块处检查），这不把失败r1当成功全量结果。全29场景/5490条件总分母不变，fold5条件CSV必须仅表头零行，预测NPZ仍保存全部条件概率。
