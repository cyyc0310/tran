# GFS已证实短读的窄范围恢复

2026-11-01，Sol。保留所有已冻结GFS采集、解码、compact及原24窗代码。新入口只恢复传输，不改模型、日期、输入预算、原始文件、可用性标记或源队列。

实际留证：`results/gfs-expanded-range-diagnostic-r2/failed-20230224-40-shortwave_surface.json`显示初始化2023-02-24、lead40的DSWRF响应。Content-Range与ETag均正确，Content-Length声明856722，GRIB头也声明856722，但只读取454981字节，尾部不完整，原collector已拒绝且未提交该snapshot。此前泛化为Range/版本ValueError，本轮仅将有完整证据的短读归入网络恢复。

分类器必须从原`gfs.collect_gfs`的真实异常traceback读取现场：错误类型/消息准确、响应Range和HEAD总大小准确、ETag准确、Content-Length等于期望、可见GRIB2头及声明长度准确，且16≤实际长度<期望。任何条件不足均保留原失败；普通哈希/索引/时间/GRIB错误不转换，不重试版本或范围错误。绝不接受截断消息。

新driver复用原cohort初始化、完整窗口检查、原子compact及owned cache清理。每窗口最多3次恢复，完整lead按原SHA复用；最终每条消息仍须通过原完整长度、版本、Range、GRIB边界及起点前对象写入检查。helper不改请求，不monkeypatch原函数，不改字节值；仅在原collector已拒绝之后识别异常现场，再调用同一原流程。

新入口、分类器、测试、本文及原source receipt在恢复前另作SHA绑定；每次识别保存事件及部分响应SHA，完成凭证引用恢复receipt和事件SHA。原`pre_model_source_receipt.json`与已完成窗口不覆盖。`available_at=null`、`eligible_for_issued_public=False`保持，完整消息不因此升级为原始公众发布认证。
