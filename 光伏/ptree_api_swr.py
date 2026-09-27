#!/usr/bin/env python3
"""
================================================================================
 P-Tree pickData API — 葵花 SWR 多站点(多点)批量提取
 ================================================================================

 API: https://www.eorc.jaxa.jp/cgi-bin/ptree/tilemap/pickData_T10m_v2r1.py
 免费、无需认证、0.05°(5km)、10 分钟间隔。
 注意：**一次请求 = 一个点的一个时刻**。实测给经纬度框(ulat/llon/dlat/rlon 拉成
 一个小区域)也只返回一个数，所以没法"一次取一片格点"，多点就是请求数成倍增加。

 相比上一版（写死单点 29.95/115.65）补齐 4 件事：

   1) 多站点     STATIONS 里想加几个站加几行，三个场站一起爬
   2) 断点续爬   每爬到一个时刻立刻落进 sqlite；重跑自动跳过已完成的。
                 3 站 × 1 年 ≈ 15 万次请求，跑一半崩了不该从头再来。
   3) 连接复用   线程内复用 requests.Session（省掉每次请求的 TLS 握手），
      + 分类重试 超时/连接错误/5xx 才重试；4xx 与空响应记"无数据"不重试
   4) 命令行    站名/起止日期/并发/输出目录都不用改源码

 用法:
   python ptree_api_swr.py                                  # 默认三站, 2025 全年(BJT)
   python ptree_api_swr.py --start 20240101 --end 20251231  # 两年一起
   python ptree_api_swr.py --stations WXJY --start 20250101 --end 20250131
   python ptree_api_swr.py --workers 32 --out-dir ./swr_out
   python ptree_api_swr.py --dry-run                        # 只数要发多少请求，不联网
   python ptree_api_swr.py --status                         # 看缓存里爬了多少
   python ptree_api_swr.py --ref-csv 场站对照/xxx.csv       # 顺带跑偏差对比

 输出（<out-dir>/）:
   himawari_SWR_BJT_<站名>.csv   每站一个，列与原脚本一致
   himawari_SWR_BJT_all.csv      合并长表，多一列 station，便于多站对比
================================================================================
"""

import argparse
import csv
import math
import os
import random
import sqlite3
import sys
import threading
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime, timedelta

import requests

# =========================== 配置（也可以在命令行覆盖） ===========================

API_URL = "https://www.eorc.jaxa.jp/cgi-bin/ptree/tilemap/pickData_T10m_v2r1.py"

# 站点清单：站名 -> (纬度, 经度)。加站点就在这里加一行。
# 坐标取自 场站对照/卫星csv测试/himawari_SWR_data_BJT_<站>.csv 的 longitude/latitude 列
STATIONS = {
    "WXJY": (29.93, 115.65),   # 武穴
    "HHFK": (30.09, 113.32),
    "SSNK": (29.74, 112.34),
}

TIME_START = "20250101"        # 北京时间 YYYYMMDD
TIME_END   = "20251231"
OUT_DIR    = os.path.dirname(os.path.abspath(__file__))
STATE_DB   = os.path.join(OUT_DIR, "state", "ptree_api_cache.sqlite")

MAX_WORKERS     = 32           # 并发线程数。旧版用 80，太容易被 JAXA 限流/掐断，建议 16~32
REQUEST_TIMEOUT = 20           # 单次请求超时(秒)
MAX_RETRIES     = 4            # 超时/连接错误/5xx 的重试次数
RESOLUTION      = "5km"
SATELLITE_NAME  = "葵花09"      # 2022-12-13 起主力星是 Himawari-9；旧脚本硬编码"葵花08"
                               # 只影响标签列，要沿用旧标签就 --satellite 葵花08

EXPECTED_MINUTES = ["00", "10", "20", "30", "40", "50"]
MAINTENANCE_SLOTS = {("02", "40"), ("14", "40")}   # UTC 维护时段，必然缺
TIME_ZONE_OFFSET = 8           # BJT = UTC + 8

# =========================== 断点续爬缓存 ===========================

class Cache:
    """sqlite 缓存：记录每个 (站点, 时刻) 的爬取结果，重跑自动跳过"""

    def __init__(self, path):
        d = os.path.dirname(os.path.abspath(path))
        if d:
            os.makedirs(d, exist_ok=True)
        self._lock = threading.Lock()
        self.conn = sqlite3.connect(path, check_same_thread=False, timeout=30)
        self.conn.execute("""CREATE TABLE IF NOT EXISTS series (
            station TEXT NOT NULL,
            sdate   TEXT NOT NULL,      -- UTC YYYYMMDDHHMM，请求用的键
            bjt     TEXT NOT NULL,      -- 北京时间字符串，输出用
            value   REAL,               -- NULL = 无数据（也缓存，免得反复重试）
            updated_at TEXT,
            PRIMARY KEY (station, sdate))""")
        self.conn.commit()

    def done_keys(self, station):
        with self._lock:
            rows = self.conn.execute(
                "SELECT sdate FROM series WHERE station=?", (station,)).fetchall()
        return {r[0] for r in rows}

    def put_many(self, rows):
        if not rows:
            return
        with self._lock:
            self.conn.executemany(
                "INSERT OR REPLACE INTO series(station,sdate,bjt,value,updated_at) "
                "VALUES(?,?,?,?,?)", rows)
            self.conn.commit()

    def rows(self, station):
        with self._lock:
            return self.conn.execute(
                "SELECT bjt, value FROM series WHERE station=? ORDER BY bjt",
                (station,)).fetchall()

    def counts(self):
        with self._lock:
            return self.conn.execute(
                "SELECT station, COUNT(*), SUM(value IS NOT NULL) FROM series "
                "GROUP BY station ORDER BY station").fetchall()


# =========================== API 查询 ===========================

_tls = threading.local()


def _session():
    """每个线程一条 Session，复用 TCP/TLS 连接（旧版每次 requests.get 都重握手）"""
    s = getattr(_tls, "sess", None)
    if s is None:
        s = requests.Session()
        s.headers.update({"User-Agent": "ptree-swr-fetch/2.0 (research use)"})
        _tls.sess = s
    return s


def query_one(sdate, lat, lon, timeout):
    """
    查一个时刻的一个点。
    返回 (status, value): status = "ok"(有值) / "nodata"(确实没数据) / "fail"(网络问题, 下次重试)
    只有 ok 和 nodata 会被缓存；fail 不缓存，所以下次运行会再试。
    """
    params = {"lang": "en", "prod": "SWR", "sdate": sdate,
              "ulat": lat, "llon": lon, "dlat": lat, "rlon": lon}
    sess = _session()
    for attempt in range(MAX_RETRIES):
        try:
            r = sess.get(API_URL, params=params, timeout=timeout)
            if r.status_code >= 500:
                raise requests.HTTPError(f"HTTP {r.status_code}")
            if r.status_code >= 400:
                return ("nodata", None)          # 客户端错误，重试无益
            txt = r.text.strip()
            if not txt:
                return ("nodata", None)
            try:
                val = float(txt.split(",")[0])
            except ValueError:
                return ("nodata", None)          # 返回了非数字（错误页/空槽）
            if math.isnan(val):
                return ("nodata", None)
            return ("ok", val)
        except (requests.Timeout, requests.ConnectionError, requests.HTTPError):
            if attempt == MAX_RETRIES - 1:
                return ("fail", None)
            time.sleep(min(2 ** attempt, 8) + random.random())
    return ("fail", None)


# =========================== 时刻表 ===========================

def build_tasks(date_start, date_end):
    """
    构建 (bjt_str, sdate_utc) 列表。
    BJT 一天跨两个 UTC 日期：00:00-07:50 来自 UTC 前一天，08:00-23:50 来自 UTC 当天。
    """
    tasks = []
    start = datetime.strptime(date_start, "%Y%m%d")
    end = datetime.strptime(date_end, "%Y%m%d")
    cur = start
    while cur <= end:
        for utc_day in (cur - timedelta(days=1), cur):
            for h in range(24):
                hh = f"{h:02d}"
                for mm in EXPECTED_MINUTES:
                    if (hh, mm) in MAINTENANCE_SLOTS:
                        continue
                    dt_utc = datetime(utc_day.year, utc_day.month, utc_day.day, h, int(mm))
                    dt_bjt = dt_utc + timedelta(hours=TIME_ZONE_OFFSET)
                    if dt_bjt.date() != cur.date():
                        continue
                    tasks.append((dt_bjt.strftime("%Y-%m-%d %H:%M:%S"),
                                  dt_utc.strftime("%Y%m%d%H%M")))
        cur += timedelta(days=1)
    return tasks


# =========================== 单站爬取 ===========================

def crawl_station(name, lat, lon, tasks, cache, workers, timeout, refresh=False):
    """爬一个站；已缓存的跳过。返回 (ok, nodata, fail) 计数"""
    have = set() if refresh else cache.done_keys(name)
    todo = [t for t in tasks if t[1] not in have]
    print(f"[{name}] 时刻总数 {len(tasks)} | 缓存已有 {len(tasks) - len(todo)} | "
          f"本次待爬 {len(todo)}")
    if not todo:
        return (0, 0, 0)

    rows, lock = [], threading.Lock()
    stat = {"ok": 0, "nodata": 0, "fail": 0}

    def flush():
        with lock:
            batch, rows[:] = list(rows), []
        cache.put_many(batch)

    t0 = time.time()
    done = 0
    total = len(todo)
    with ThreadPoolExecutor(workers) as ex:
        futs = {ex.submit(query_one, sd, lat, lon, timeout): (bjt, sd)
                for bjt, sd in todo}
        for f in as_completed(futs):
            bjt, sd = futs[f]
            status, val = f.result()
            done += 1
            with lock:
                stat[status] += 1
                if status != "fail":
                    rows.append((name, sd, bjt, val,
                                 datetime.now().strftime("%Y-%m-%d %H:%M:%S")))
            if len(rows) >= 200:
                flush()
            if done % 500 == 0 or done == total:
                el = max(time.time() - t0, 1e-6)
                rate = done / el
                eta_min = (total - done) / rate / 60 if rate else float("nan")
                print(f"  [{name}] {done}/{total}  {rate:.1f} 请求/秒  "
                      f"已用 {el/60:.1f} 分钟  预计剩余 {eta_min:.1f} 分钟")
    flush()
    return (stat["ok"], stat["nodata"], stat["fail"])


# =========================== 输出 ===========================

def write_outputs(cache, out_dir, stations, satellite):
    os.makedirs(out_dir, exist_ok=True)
    all_path = os.path.join(out_dir, "himawari_SWR_BJT_all.csv")
    header = ["observation_time", "swr_value", "satellite_name",
              "longitude", "latitude", "resolution"]
    with open(all_path, "w", newline="", encoding="utf-8-sig") as fa:
        wa = csv.writer(fa)
        wa.writerow(["station"] + header)
        for name in stations:
            lat, lon = STATIONS[name]
            recs = cache.rows(name)
            per_path = os.path.join(out_dir, f"himawari_SWR_BJT_{name}.csv")
            with open(per_path, "w", newline="", encoding="utf-8-sig") as fp:
                wp = csv.writer(fp)
                wp.writerow(header)
                for bjt, val in recs:
                    v = "" if val is None else f"{val:.6f}"
                    row = [bjt, v, satellite, lon, lat, RESOLUTION]
                    wp.writerow(row)
                    wa.writerow([name] + row)
            n_ok = sum(1 for _, v in recs if v is not None)
            print(f"  写出 {per_path}  ({len(recs)} 行, 其中有值 {n_ok} 行)")
    print(f"  写出 {all_path}（多站合并长表）")


# =========================== 偏差对比（可选） ===========================

def compare_with_reference(station, ref_csv_path, out_dir):
    """
    与参考 CSV 对比偏差。参考一般来自 NC 裁剪（双线性），API 是最近邻，
    预期有 10~20 W/m² 的方法差异。需要 pandas，所以放在这里懒加载。
    """
    import numpy as np
    import pandas as pd

    api_path = os.path.join(out_dir, f"himawari_SWR_BJT_{station}.csv")
    if not os.path.exists(ref_csv_path) or not os.path.exists(api_path):
        print(f"  跳过对比：缺少 {ref_csv_path} 或 {api_path}")
        return
    ref = pd.read_csv(ref_csv_path)
    api = pd.read_csv(api_path)
    if "latitude" in ref.columns and "latitude" in api.columns:
        ref = ref[(ref["latitude"] == api["latitude"].iloc[0]) &
                  (ref["longitude"] == api["longitude"].iloc[0])]
    merged = ref.merge(api, on="observation_time", suffixes=("_ref", "_api"))
    if merged.empty:
        print("  跳过对比：两边没有可对齐的时刻")
        return
    merged["diff"] = merged["swr_value_api"] - merged["swr_value_ref"]
    merged["abs_diff"] = merged["diff"].abs()
    day = merged[merged["swr_value_ref"] > 1]
    night = merged[merged["swr_value_ref"] <= 1]

    print(f"\n{'=' * 58}")
    print(f"  [{station}] API vs 参考CSV 偏差统计")
    print(f"{'=' * 58}")
    print(f"  匹配: {len(merged)} 条  |  白天 {len(day)} / 夜间 {len(night)}")
    if len(day):
        print(f"  白天 MAE : {day['abs_diff'].mean():8.2f} W/m²")
        print(f"  白天 RMSE: {np.sqrt((day['diff'] ** 2).mean()):8.2f} W/m²")
        print(f"  相关系数 r: {day['swr_value_ref'].corr(day['swr_value_api']):8.4f}")
        print(f"  平均偏差 : {day['diff'].mean():+8.2f} W/m²")
        print(f"  最大偏差 : {day['abs_diff'].max():8.2f} W/m²")
    if len(night):
        print(f"  夜间 MAE : {night['abs_diff'].mean():8.2f} W/m²")
    print(f"{'=' * 58}")


# =========================== 主流程 ===========================

def main():
    ap = argparse.ArgumentParser(
        description="P-Tree pickData API 多站点 SWR 批量提取（支持断点续爬）",
        formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--stations", default=",".join(STATIONS),
                    help=f"站名，逗号分隔（可选: {','.join(STATIONS)}）")
    ap.add_argument("--start", default=TIME_START, help="开始日期 YYYYMMDD（北京时间）")
    ap.add_argument("--end", default=TIME_END, help="结束日期 YYYYMMDD（北京时间）")
    ap.add_argument("--workers", type=int, default=MAX_WORKERS, help="并发线程数")
    ap.add_argument("--timeout", type=int, default=REQUEST_TIMEOUT, help="单请求超时(秒)")
    ap.add_argument("--out-dir", default=OUT_DIR, help="输出目录")
    ap.add_argument("--state-db", default=STATE_DB, help="断点续爬缓存(sqlite)")
    ap.add_argument("--satellite", default=SATELLITE_NAME, help="satellite_name 标签列")
    ap.add_argument("--ref-csv", default=None, help="可选：偏差对比用的参考 CSV")
    ap.add_argument("--refresh", action="store_true", help="忽略缓存，整段重爬")
    ap.add_argument("--dry-run", action="store_true", help="只统计请求数，不联网")
    ap.add_argument("--status", action="store_true", help="只看缓存进度，不联网")
    args = ap.parse_args()

    names = [s.strip() for s in args.stations.split(",") if s.strip()]
    unknown = [s for s in names if s not in STATIONS]
    if unknown:
        ap.error(f"未知站名 {unknown}；已定义的站: {list(STATIONS)}")

    tasks = build_tasks(args.start, args.end)
    cache = Cache(args.state_db)

    if args.status:
        print(f"缓存 {args.state_db}")
        print(f"{'站点':<8}{'已爬时刻':>10}{'其中有值':>10}")
        for st, n, n_ok in cache.counts():
            print(f"{st:<8}{n:>10}{n_ok or 0:>10}")
        return 0

    if args.dry_run:
        per_station = len(tasks)
        print(f"时间范围 {args.start} ~ {args.end} (BJT)，每站 {per_station} 个时刻")
        for name in names:
            have = len(cache.done_keys(name))
            todo = max(0, per_station - have)
            print(f"  {name}: 待爬 {todo} 次请求（缓存已有 {have}）")
        print(f"合计约 {sum(max(0, per_station - len(cache.done_keys(n))) for n in names)} 次请求")
        return 0

    print(f"时间范围 {args.start} ~ {args.end} (BJT) | 站点 {names} | 并发 {args.workers}")
    t0 = time.time()
    total_ok = total_nodata = total_fail = 0
    for name in names:
        lat, lon = STATIONS[name]
        ok, nodata, fail = crawl_station(name, lat, lon, tasks, cache,
                                         args.workers, args.timeout, args.refresh)
        total_ok += ok
        total_nodata += nodata
        total_fail += fail
    print(f"\n爬取结束，用时 {(time.time() - t0)/60:.1f} 分钟："
          f"有值 {total_ok} | 无数据 {total_nodata} | 失败(未缓存, 重跑会再试) {total_fail}")

    write_outputs(cache, args.out_dir, names, args.satellite)
    if args.ref_csv:
        for name in names:
            compare_with_reference(name, args.ref_csv, args.out_dir)
    return 0


if __name__ == "__main__":
    sys.exit(main())
