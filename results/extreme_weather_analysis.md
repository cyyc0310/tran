# 极端天气 → CIF 极端事件:归因分析(2023,29 区域)

**问题**:极端天气(台风/飓风/温带风暴/暴雨/冰冻/热浪/风电 drought)是否造成电网碳强度的极端事件?极端 CIF 波动的成因是什么?

**数据**:`data_2023/weather`(ERA5 温度/短波/100m 风速)+ `weather2`(阵风/海平面气压,ERA5)+ **新增 `weather3`(ERA5 逐小时降水,29 区域)** + 25 区逐小时逐燃料发电遥测。事件日历经 BOM/NWS/Met Office/NOAA 核实。

**脚本**:`scripts/data/download_precipitation.py`(降水)、`scripts/analysis/analyze_extreme_weather.py`(检测+归因)、`scripts/figures/make_extreme_weather_figure.py`(事件图)。明细:`results/extreme_weather_analysis.json`。

---

## 一、结论(先说答案)

1. **极端天气确实全部命中了我们的天气数据** —— 飓风 Idalia 在 FPL 阵风 z=+5.1、Hilary 在 CISO z=+4.5、气旋 Jasper 在 QLD1 z=+2.5、风暴 Debi/Gerrit 在 UK z=+3.4~4.5、冬季风暴 Mara 在 ERCO 温度 z=−2.6、Babet 暴雨 25 mm/6h。天气通道本身是干净的。
2. **但天气→CIF 的传导只有一个通道:天气敏感电源(风/光)的份额摆动**。份额≈0 的纯热力电网(FPL 风 0%/光 5%)在 Cat 3 飓风过境时 CIF 波动反而**下降**(std 31→16)。
3. **真正危险的 regime 不是风暴,而是"平静过渡带"**:风电份额在 0.15–0.3 之间摆动时 CIF 日波动最高(50),份额钉在 0.6+ 时最低(19.5)。UK 全网 CIF 波动与阵风相关为**负**(−0.27):大风天风电满发、CIF 又低又稳;无风天风电在混合中点进出、气电频繁顶上,才产生极端波动。
4. 热浪/冰冻通过**需求端**起作用(峰值气电开机):PJM 2 月寒潮 CIF 波动 ×1.85、7 月热穹 ×1.52,ERCO Mara ×1.35 —— 热力占比高的电网对温度极端更敏感。
5. **暴雨本身与 CIF 几乎无关**(29 区相关中位 −0.14):降水只通过云量→光伏、洪涝→停电影响电网,而停电不改变送出电量的燃料结构。

## 二、事件归因表(节选,完整见 JSON)

| 事件 | 区域 | 天气签名(实测) | CIF 波动比* | 类型 |
|---|---|---|---|---|
| 9 月热浪(平静) | UK_01 北苏格兰 | 阵风 5 m/s(死风),风电份额 0.79→0.20 | **27.4×** | wind-lull |
| Storm Debi | UK_01/UK_03 | gust z +1.7~+4.5,雨 10 mm/6h | 8.6× / 2.0× | cyclone |
| Storm Elin/Fergus | UK_01 | gust z +0.9~+4.1 | 8.0× | wind-lull |
| Storm Gerrit | UK_02/UK_06 | gust z +3.2,雨 20 mm/6h,冻 | 0.4× / 3.3× | cyclone+freeze |
| Storm Babet | UK_03/UK_08 | gust z +3.3,雨 16~25 mm/6h | 2.0× / 2.8× | cyclone+rain |
| 9 月风电 drought | SA1 | w100 z −1.5 以下持续 | **1.6×** | wind-lull |
| 同窗口 VIC1 | VIC1 | gust z +3.3 但 w100 z 低 | 1.4× | wind-lull |
| 2 月寒潮 | US_PJM | temp z −2.7, gust z +3.4 | 1.85× | freeze |
| 冬季风暴 Mara | US_ERCO | temp z −2.6(冰冻) | 1.35× | freeze |
| 7 月热穹 | US_PJM/MISO | temp z +2.4, w100 z −2 | 1.52× / 0.94× | heat+lull |
| **飓风 Idalia (Cat 3)** | US_FPL | **gust z +5.1**,雨 11 mm/6h | **0.64×(更稳!)** | — |
| **飓风 Hilary 残余** | US_CISO | gust z +4.5,雨 15 mm/6h | 0.83× | — |
| 气旋 Jasper (Cat 2) | QLD1 | gust z +2.5,雨 8.7 mm/6h | 1.07× | cyclone+lull |

\* 事件窗 CIF std / 该区域 2023 日波动中位数。UK_01 的 27× 之所以巨大:该区域平时风电稳发、CIF 几乎不动(日 std≈4)。

## 三、机制验证(三个钻取)

1. **UK_01 热浪(9/4–9/11)**:风电份额中位 0.785,热浪期间日均值在 0.20↔0.77 之间逐日摆动(阵风仅 5 m/s);CIF 日 std 从基线 ~4 飙到 50–90。成因 = 反气旋静稳 → 风电出力在中点进出 → 气电循环顶替。
2. **FPL 飓风 Idalia(8/29–9/1)**:阵风 21 m/s(5σ、区域 P99=14.6),但风/光合计份额仅 ~5%,CIF 均值 332、std 16,六个风暴日 std 全部 ≤19(vs 基线 31)。飓风摧毁的是配电网(停电 28k–346k 户),不改变送出电量的燃料结构 → 对 CIF 不可见。
3. **跨区域风电份额区间量化**(25 个有燃料遥测的区域,按日聚合):wind_share∈[0.15,0.30) 日均 CIF std=50.1,∈[0.45,0.60)=33.1,∈[0.60,1.0]=19.5 —— 单调递减,混合中点即波动峰值区。

## 四、对 TransCIF-FD 的含义

- **困难区域(VIC1/SA1/UK 中部)的极端日 = wind-lull 过渡日**,不是风暴日 —— 这解释了 MAE-floor 分析中"波动性 r=+0.85 决定难度":模型要预测的是**份额摆动相位**,天气(风速)是其唯一前导信号,阵风/气压通道(已在 wx 通道 5/6)方向正确。
- 风暴 cut-out(阵风 >25 m/s 风机保护停机)在 UK 数据中未见明显 CIF 爆炸信号(阵风 z+3~4 天多数 ratio <3),ERA5 网格 11 km 对风暴眼分辨率不足 + 风机抗切出保护,优先级低于 wind-lull。
- 降水通道对 CIF 无直接增益(相关 ~−0.14),但可留作**光伏压制指数**(云量代理)与洪涝停电风险特征;不建议直接进模型通道。
- 温度极端(热穹/寒潮)是需求端驱动,degree-hour 通道(hdh/cdh)已覆盖,heat-dome 日 ratio ~1.5 量级,小于 wind-lull 的极端值。

## 五、产出物

- `figures/extreme_weather_events.png/.pdf` — 8 案例 16 面板:CIF×风电份额 | 阵风×降水×气压
- `results/extreme_weather_analysis.json` — 29 区日相关矩阵、逐区 top-5 极端日、全部已知事件归因
- `data_2023/weather3/{REGION}_precip_2023_hourly.csv` — 29 区 ERA5 降水
