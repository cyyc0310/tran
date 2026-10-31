# V42准备阶段失败与r1尝试说明

2026-10-27，在新增供体细分标签和实际拟合前记录。

原session 77171在prepare阶段exit 1，原日志保留为`results/donor-budget-v42-prepare.log`，SHA256 `6489bebaae45ffb06560e657fe778d1608e4cf897fc8e56aae1c3d93b6a20c44`。root在代码预审报告尚未落盘时启动，contract因缺少该输入报`Regular new source input required`。当时`results/donor-budget-v42`目录、freeze、prediction_started均不存在；predict和score没有执行，没有读取新增fine响应或拟合模型。

r1为单独记录的新尝试，待两份预审报告完整落盘并核对后启动。因为原输出目录从未生成，r1仍使用原合同的空输出路径`results/donor-budget-v42`，三个阶段日志改用`donor-budget-v42-prepare-r1.log`、`donor-budget-v42-predict-r1.log`、`donor-budget-v42-score-r1.log`，均exclusive创建。原失败日志不改名、不覆盖、不删除；不把原session写成成功。r1使用同五个代码/测试/合同哈希、数据、队列、参数和阈值，不按结果改变设计。本说明仅修订失败后的执行记账，不改实验设计。

runner SHA256 `884fda72a377f2f944bb573b36cbb31ecd1f604b9c19a0a405a85089fdb0e0de`；其余四个producer pins沿用预审。最终证据同时绑定本说明、原失败日志、r1三日志和真实退出码。方法创新、区间覆盖、小时覆盖和未查看地区资格均未获得。
