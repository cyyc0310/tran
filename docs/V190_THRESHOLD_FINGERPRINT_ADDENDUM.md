# V190阈值±0指纹修复，r1退役

2026-10-29。r1 smoke和六折无评分preflight成功，但二者fold0 HGB原始nodes bytes指纹不同。执行者的复合shell在比较assert失败后仍短暂启动formal，formal由run_fold入口指纹校验拒绝并exit1，未seal任何query law、未读cal/test评分真值、未提交fold；r1正式目录仅manifest。保留当时源码、日志、smoke/preflight，不能称r1正式成功。

初假设padding未成立：首次两fresh拟合诊断在命名bytes检查中失败；后续三次fresh fold0（27风险模型）逐字段/字节定位为45处num_threshold的+0.0/−0.0表示差异，padding差异0、其余具名字段/位集合均一致。每个7 HGB在全部17262源行与全部34query×48状态上的预测严格tobytes一致，原始差异nodes/预测NPZ与完整记录保留`results/v190-hgb-bit-diagnostic-r1`。原smoke/首次preflight的完整模型未保存，不能声称直接核对了那两个原进程对象；该新诊断复现并定位同类问题。

阈值+0和−0对≤比较及缺失值路由等价，missing_go_to_left等其他字段一致；不修改模型或阈值，仅指纹逐具名field哈希，并且**只对num_threshold字段数值等于0的项规范化为+0**。非零阈值、其他浮点字段及所有其他字段不放宽；保存dtype/node_count。不是近似阈值或模型预测容差比较。

r2六折预检每折两次fresh fit，逐项语义树指纹相等，并对9模型（两ET、7HGB）完整源输入及所有cal/test query×48状态预测严格bytes相等。预检实际108风险模型拟合（54selected+54repeat），正式仍54模型。12完整未来/raw guard和100 toy保留；正式与审计全部r2新归档，不覆盖r1。依赖hash包含本说明；任何后续实质节点/预测差异都必须拒绝，不能继续规范化隐藏。
