#!/usr/bin/env python3
"""
================================================================================
 P-Tree pickData API — 葵花8号 SWR 按经纬度+时间批量提取
 ================================================================================

 API: https://www.eorc.jaxa.jp/cgi-bin/ptree/tilemap/pickData_T10m_v2r1.py
 免费, 无需认证, 0.05°(5km)分辨率, 10分钟间隔

 用法:
   修改下方全局变量 → python ptree_api_swr.py

 输出格式:
   observation_time,swr_value,satellite_name,longitude,latitude,resolution
================================================================================
"""

import pandas as pd
import numpy as np
import requests
import time
import os
from datetime import datetime, timedelta
from concurrent.futures import ThreadPoolExecutor, as_completed

# =========================== 全局配置 (在这里改) ===========================

# 目标经纬度
TARGET_LAT = 29.95
TARGET_LON = 115.65

# 时间范围 (BJT 北京时间, YYYYMMDD)
TIME_START = "20250101"
TIME_END   = "20251231"

# 输出 CSV 路径
OUTPUT_CSV = os.path.join(os.path.dirname(__file__),
                          "himawari_SWR_WXJY_BJT_API_20250101_20251231.csv")

# 参考 CSV (可选, 用于偏差对比, 设为 None 跳过)
REFERENCE_CSV = None

# 分辨率标签
RESOLUTION = "5km"

# API 参数
API_URL = "https://www.eorc.jaxa.jp/cgi-bin/ptree/tilemap/pickData_T10m_v2r1.py"
MAX_WORKERS = 80        # 并发线程数
REQUEST_TIMEOUT = 15     # 单次请求超时(秒)
MAX_RETRIES = 3          # 失败重试次数

# 时间配置
EXPECTED_MINUTES = ["00", "10", "20", "30", "40", "50"]
MAINTENANCE_SLOTS = {("02", "40"), ("14", "40")}  # UTC 维护时段
TIME_ZONE_OFFSET = 8     # BJT = UTC + 8

# =========================== API 查询 ===========================

def query_one(bjt_str, sdate, lat, lon):
    """单个时刻查询, 返回 (bjt_str, swr_value) 或 None"""
    params = {
        "lang": "en", "prod": "SWR", "sdate": sdate,
        "ulat": lat, "llon": lon, "dlat": lat, "rlon": lon,
    }
    for _ in range(MAX_RETRIES):
        try:
            resp = requests.get(API_URL, params=params, timeout=REQUEST_TIMEOUT)
            resp.raise_for_status()
            text = resp.text.strip()
            val = float(text.split(",")[0]) if text else 0.0
            if np.isnan(val):
                return None
            return (bjt_str, val)
        except:
            time.sleep(0.3)
    return None


def extract_series(lat, lon, date_start, date_end):
    """
    批量提取 SWR 时间序列

    参数:
        lat, lon:          目标经纬度
        date_start/end:    'YYYYMMDD' (BJT)

    返回:
        list of dict, 按 observation_time 排序
    """
    start = datetime.strptime(date_start, "%Y%m%d")
    end   = datetime.strptime(date_end, "%Y%m%d")

    # 构建所有待查询时刻
    # BJT 一天跨两个 UTC 日期: 00:00-07:50 来自 UTC 前一天, 08:00-23:50 来自 UTC 当天
    tasks = []
    current = start
    while current <= end:
        utc_day1 = current - timedelta(days=1)
        utc_day2 = current
        for utc_day in [utc_day1, utc_day2]:
            for h in range(24):
                hh = f"{h:02d}"
                for mm in EXPECTED_MINUTES:
                    if (hh, mm) in MAINTENANCE_SLOTS:
                        continue
                    dt_utc = datetime(utc_day.year, utc_day.month, utc_day.day,
                                     h, int(mm))
                    dt_bjt = dt_utc + timedelta(hours=TIME_ZONE_OFFSET)
                    if dt_bjt.date() != current.date():
                        continue
                    bjt_str = dt_bjt.strftime("%Y-%m-%d %H:%M:%S")
                    sdate = dt_utc.strftime("%Y%m%d%H%M")
                    tasks.append((bjt_str, sdate, lat, lon))
        current += timedelta(days=1)

    print(f"待查询: {len(tasks)} 个时刻")
    print(f"时间范围: {date_start} ~ {date_end} (BJT)")
    print(f"经纬度: ({lat}, {lon})")

    # 80 线程并发
    records = {}
    t0 = time.time()
    with ThreadPoolExecutor(MAX_WORKERS) as ex:
        futures = {ex.submit(query_one, *t): t for t in tasks}
        for i, f in enumerate(as_completed(futures)):
            r = f.result()
            if r:
                records[r[0]] = r[1]
            if (i + 1) % 500 == 0:
                print(f"  进度: {i+1}/{len(tasks)}")

    elapsed = time.time() - t0
    print(f"完成: {len(records)}/{len(tasks)} 条, 耗时 {elapsed:.0f}s "
          f"({elapsed/60:.1f}min)")

    # 转为有序列表
    result = []
    for t in sorted(records.keys()):
        result.append({
            "observation_time": t,
            "swr_value": records[t],
            "satellite_name": "葵花08",
            "longitude": lon,
            "latitude": lat,
            "resolution": RESOLUTION,
        })
    return result


# =========================== 偏差计算 ===========================

def compare_with_reference(records, ref_csv_path):
    """
    与参考 CSV 对比偏差

    参考 CSV 使用双线性插值, API 使用最近邻。
    预期偏差 10~20 W/m² 属正常插值方法差异。
    """
    ref = pd.read_csv(ref_csv_path)
    api = pd.DataFrame(records)

    lat = records[0]["latitude"] if records else None
    lon = records[0]["longitude"] if records else None
    if lat and lon:
        ref = ref[(ref["latitude"] == lat) & (ref["longitude"] == lon)]

    merged = ref.merge(api, on="observation_time", suffixes=("_ref", "_api"))
    merged["diff"] = merged["swr_value_api"] - merged["swr_value_ref"]
    merged["abs_diff"] = merged["diff"].abs()

    day = merged[merged["swr_value_ref"] > 1]
    night = merged[merged["swr_value_ref"] <= 1]

    print(f"\n{'='*55}")
    print(f"  API vs 参考CSV 偏差统计")
    print(f"{'='*55}")
    print(f"  匹配: {len(merged)}/{len(ref)} 条")
    print(f"  白天: {len(day)} 条  |  夜间: {len(night)} 条")
    print()
    print(f"  白天 MAE:      {day['abs_diff'].mean():.2f} W/m²")
    print(f"  白天 RMSE:     {np.sqrt((day['diff']**2).mean()):.2f} W/m²")
    print(f"  相关系数 r:     {day['swr_value_ref'].corr(day['swr_value_api']):.4f}")
    print(f"  平均偏差:      {day['diff'].mean():+.2f} W/m²")
    print(f"  最大偏差:      {day['abs_diff'].max():.2f} W/m²")
    print(f"  夜间 MAE:      {night['abs_diff'].mean():.2f} W/m²")
    print(f"{'='*55}")


# =========================== 主流程 ===========================

if __name__ == "__main__":
    records = extract_series(TARGET_LAT, TARGET_LON, TIME_START, TIME_END)

    df = pd.DataFrame(records)
    os.makedirs(os.path.dirname(OUTPUT_CSV), exist_ok=True)
    df.to_csv(OUTPUT_CSV, index=False, float_format="%.6f")
    print(f"\n已保存: {OUTPUT_CSV} ({len(df)} 条)")

    if REFERENCE_CSV and os.path.exists(REFERENCE_CSV):
        compare_with_reference(records, REFERENCE_CSV)
