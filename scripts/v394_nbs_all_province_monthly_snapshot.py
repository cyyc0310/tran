#!/usr/bin/env python3
"""Download the six official monthly generation series for all mainland provinces."""

from __future__ import annotations

import argparse
from pathlib import Path

import v391_nbs_official_monthly_snapshot as snapshot


REGIONS = {
    "Beijing": ("北京市", "110000000000"),
    "Tianjin": ("天津市", "120000000000"),
    "Hebei": ("河北省", "130000000000"),
    "Shanxi": ("山西省", "140000000000"),
    "InnerMongolia": ("内蒙古自治区", "150000000000"),
    "Liaoning": ("辽宁省", "210000000000"),
    "Jilin": ("吉林省", "220000000000"),
    "Heilongjiang": ("黑龙江省", "230000000000"),
    "Shanghai": ("上海市", "310000000000"),
    "Jiangsu": ("江苏省", "320000000000"),
    "Zhejiang": ("浙江省", "330000000000"),
    "Anhui": ("安徽省", "340000000000"),
    "Fujian": ("福建省", "350000000000"),
    "Jiangxi": ("江西省", "360000000000"),
    "Shandong": ("山东省", "370000000000"),
    "Henan": ("河南省", "410000000000"),
    "Hubei": ("湖北省", "420000000000"),
    "Hunan": ("湖南省", "430000000000"),
    "Guangdong": ("广东省", "440000000000"),
    "Guangxi": ("广西壮族自治区", "450000000000"),
    "Hainan": ("海南省", "460000000000"),
    "Chongqing": ("重庆市", "500000000000"),
    "Sichuan": ("四川省", "510000000000"),
    "Guizhou": ("贵州省", "520000000000"),
    "Yunnan": ("云南省", "530000000000"),
    "Tibet": ("西藏自治区", "540000000000"),
    "Shaanxi": ("陕西省", "610000000000"),
    "Gansu": ("甘肃省", "620000000000"),
    "Qinghai": ("青海省", "630000000000"),
    "Ningxia": ("宁夏回族自治区", "640000000000"),
    "Xinjiang": ("新疆维吾尔自治区", "650000000000"),
}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--start", default="201801MM")
    parser.add_argument("--stop", default="202609MM")
    parser.add_argument("--output", type=Path, default=Path("results/v394-nbs-all-province-long-panel-r1"))
    args = parser.parse_args()
    snapshot.REGIONS = REGIONS
    snapshot.main(args.start, args.stop, args.output)


if __name__ == "__main__":
    main()
