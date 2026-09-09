# FD-46: event-day conservative shrinkage — NEGATIVE, archived

**规则**: wind_lull 事件日将 I_cfg 预测向 config-constant 收缩（w 网格 0-1，LORO 从其他 28 区选权，部署合法）。

**结果**: LORO w* 几乎全为 1.0（不收缩）；oracle 上界也仅 VIC1 -5.9（132.1→126.2）、UK_16 -0.4，其余 <±0.2。中位 MAE 52.90 → 52.90（+0.00）。

**解释**: FD-45 的事件日误差不是"水平过度自信"（config-constant 恰好水平稳），而是"日内份额摆动相位缺失"。config-constant 在事件日同样系统性错位（VIC1 config-constant 事件日 MAE 也 >170），向它收缩无法注入相位信息。

**结论**: 事件日改进须引入调度响应先验（P1 merit-order：火电爬坡排序决定 lull 日的份额摆动方向/相位），浅层统计收缩无空间。归档不再投入。
