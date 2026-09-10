# FD-48 join 取证（FD-47 残余恶化的对齐诊断）

| region | farmblend | gfs holes(test) | icon holes(test) | gfs best lag w100 | icon best lag w100 | swr peak nwp/era5 |
|---|---|---|---|---|---|---|
| NSW1 | False | 0 | 0 | 0 (r=0.8051) | 0 (r=0.838) | 12h/12h |
| QLD1 | False | 0 | 0 | -1 (r=0.6873) | 0 (r=0.7558) | 12h/12h |
| VIC1 | False | 0 | 0 | 0 (r=0.8875) | 0 (r=0.9043) | 13h/13h |
| SA1 | False | 0 | 0 | 0 (r=0.8478) | 1 (r=0.8173) | 13h/14h |
| US_BPAT | False | 9 | 9 | None (r=None) | None (r=None) | 12h/0h |
| US_CISO | False | 9 | 9 | -2 (r=0.3952) | -1 (r=0.4211) | 12h/13h |
| US_NYIS | False | 6 | 6 | -3 (r=0.6453) | -2 (r=0.6906) | 12h/12h |
| UK_16_Scotland | False | 0 | 0 | 0 (r=0.7075) | 0 (r=0.7568) | 13h/12h |
| UK_11_South_West_England | False | 0 | 0 | -1 (r=0.829) | -1 (r=0.8326) | 13h/12h |

- QLD1 DST 月规则 vs 真实日历：0 小时不符，测试区 0 小时（例 []）
