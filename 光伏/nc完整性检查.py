"""
================================================================================
 NC文件时间完整性检查工具
================================================================================
 检查指定时间窗口内葵花8号SWR数据的NC文件是否完整。

 目录结构要求:
   {DATA_ROOT}/{YYYYMM}/              ← 月份文件夹
     {DD}/                            ← 日期文件夹 (01-31)
       {HH}/                          ← 小时文件夹 (00-23)
         H09_{YYYYMMDD}_{HHMM}_...nc  ← 10分钟间隔NC文件

 输出:
   - 按小时汇总的缺失统计
   - 逐日详细缺失列表（区分维护/异常）
   - 完整性百分比
================================================================================
"""

import os, re, sys
from datetime import datetime
from collections import defaultdict

# ==================== 配置参数 ====================
DATA_ROOT = "/work/zfshu/24learn/hyx-test/大创/葵花卫星/SWR" # NC文件根目录
YEAR_MONTH = "202602" # 要检查的年月 (YYYYMM)
DAY_START = 3 # 起始日期 (含)
DAY_END = 22 # 结束日期 (含)
OUTPUT_FILE = "/work/zfshu/24learn/hyx-test/大创/葵花卫星/SWR/完整性报告_202602.txt" # 报告输出路径，None则自动生成 完整性报告_{YYYYMM}.txt

# 文件名正则: H09_YYYYMMDD_HHMM_xxx.nc
FILE_PATTERN = re.compile(r"H09_(\d{8})_(\d{4})_.*\.nc")

# 每小时应有的分钟时刻 (10分钟间隔)
EXPECTED_MINUTES = ["00", "10", "20", "30", "40", "50"]

# 卫星例行维护时段 (UTC小时, UTC分钟) → 说明
MAINTENANCE_SLOTS = {
    ("02", "40"): "北京10:40 葵花8号例行维护",
    ("14", "40"): "北京22:40 葵花8号例行维护",
}

# ==================== 工具函数 ====================
def utc_to_bjt(utc_hour, utc_min):
    """
    UTC时间转北京时间
    输入:
        utc_hour: int (0-23)
        utc_min: str ("00", "10", "20", "30", "40", "50")
    输出:
        str: 北京时间 "HH:MM"
    """
    total_min = (utc_hour * 60 + int(utc_min)) + 8 * 60
    bjt_h = (total_min // 60) % 24
    bjt_m = total_min % 60
    return f"{bjt_h:02d}:{bjt_m:02d}"


def scan_directory(root, year_month, day_start, day_end):
    """
    扫描目录，返回每个时次的文件存在情况
    输入：
        root: str, 根目录
        year_month: str, 年月 "YYYYMM"
        day_start: int, 起始日 (含)
        day_end: int, 结束日 (含)
    输出：
        records: dict
            key: (day_str, hour_str, minute_str)
            value: 文件路径
    """
    base = os.path.join(root, year_month)
    if not os.path.isdir(base):
        print(f"错误: 目录不存在 - {base}")
        sys.exit(1)

    records = {}

    for day in range(day_start, day_end + 1):
        day_str = f"{day:02d}"
        day_path = os.path.join(base, day_str)
        if not os.path.isdir(day_path):
            continue

        for hour in sorted(os.listdir(day_path)):
            hour_path = os.path.join(day_path, hour)
            if not os.path.isdir(hour_path):
                continue

            # 记录该小时实际存在的分钟
            for f in os.listdir(hour_path):
                m = FILE_PATTERN.match(f)
                if m:
                    file_date = m.group(1)   # YYYYMMDD
                    file_time = m.group(2)   # HHMM
                    file_mm = file_time[2:]  # 分钟部分
                    records[(day_str, hour, file_mm)] = os.path.join(hour_path, f)

    return records


def check_completeness(records, day_start, day_end):
    """
    检查完整性
    输入：
        records: dict, 扫描结果
        day_start: int, 起始日 (含)
        day_end: int, 结束日 (含)
    输出：
        maint_missing: list of (day_str, hour, mm, bjt, reason)
        other_missing: list of (day_str, hour, mm, bjt)
        hour_stats: dict, 每小时的统计信息
    """
    maint_missing = []
    other_missing = []
    hour_stats = defaultdict(lambda: {"present": 0, "total_days": 0, "missing": 0})

    for day in range(day_start, day_end + 1):
        day_str = f"{day:02d}"
        # 扫描该天实际存在的小时
        hours_present = set()
        for (d, h, mm) in records:
            if d == day_str:
                hours_present.add(h)

        for hour in hours_present:
            hour_stats[hour]["total_days"] += 1
            for mm in EXPECTED_MINUTES:
                key = (day_str, hour, mm)
                if key in records:
                    hour_stats[hour]["present"] += 1
                else:
                    hour_stats[hour]["missing"] += 1
                    slot_key = (hour, mm)
                    bjt = utc_to_bjt(int(hour), mm)
                    if slot_key in MAINTENANCE_SLOTS:
                        maint_missing.append((day_str, hour, mm, bjt, MAINTENANCE_SLOTS[slot_key]))
                    else:
                        other_missing.append((day_str, hour, mm, bjt))

    return maint_missing, other_missing, hour_stats


def generate_report(root, year_month, day_start, day_end,
                    maint_missing, other_missing, hour_stats):
    """
    生成完整性报告
    输入：
        root: str, 根目录
        year_month: str, 年月 "YYYYMM"
        day_start: int, 起始日 (含)
        day_end: int, 结束日 (含)
        maint_missing: list of (day_str, hour, mm, bjt, reason)
        other_missing: list of (day_str, hour, mm, bjt)
        hour_stats: dict, 每小时的统计信息
    输出：
        str, 完整性报告文本
    """
    lines = []
    w = 70

    total_days = day_end - day_start + 1
    total_hours_in_data = sum(s["total_days"] for s in hour_stats.values())
    total_expected = total_hours_in_data * 6
    total_found = sum(s["present"] for s in hour_stats.values())
    total_missing = total_expected - total_found

    lines.append("=" * w)
    lines.append(" 葵花8号NC文件时间完整性报告")
    lines.append(f" 生成时间: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    lines.append(f" 数据目录: {os.path.join(root, year_month)}")
    lines.append(f" 检查范围: {year_month[:4]}-{year_month[4:]}-{day_start:02d} ~ {year_month[:4]}-{year_month[4:]}-{day_end:02d} ({total_days}天)")
    lines.append("=" * w)
    lines.append("")

    # 按小时汇总
    lines.append("-" * w)
    lines.append(f" {'小时(UTC)':>8} {'出现天数':>8} {'应有':>6} {'实有':>6} {'缺失':>5}  状态")
    lines.append("-" * w)
    for hour in sorted(hour_stats.keys(), key=int):
        s = hour_stats[hour]
        exp = s["total_days"] * 6
        miss = exp - s["present"]
        flag = "✓" if miss == 0 else f"✗缺{miss}"
        lines.append(f" {hour}:00 {'':>4} {s['total_days']:>6}天 {exp:>6} {s['present']:>6} {miss:>5}  {flag}")
    lines.append("")

    # 维护缺失
    lines.append("-" * w)
    lines.append(f" ◼ 卫星维护时段缺失: {len(maint_missing)} 个（正常）")
    if maint_missing:
        # 按slot汇总
        slot_counts = defaultdict(int)
        for day_str, hour, mm, bjt, reason in maint_missing:
            slot_counts[(hour, mm, reason)] += 1
        for (hour, mm, reason), cnt in sorted(slot_counts.items()):
            lines.append(f"    UTC {hour}:{mm} ({reason}) × {cnt}天")
    lines.append("")

    # 异常缺失
    lines.append("-" * w)
    lines.append(f" ◼ 异常缺失: {len(other_missing)} 个")
    if other_missing:
        for day_str, hour, mm, bjt in other_missing:
            lines.append(f"    {year_month[:4]}-{year_month[4:]}-{day_str} UTC {hour}:{mm} (北京{bjt})")
    else:
        lines.append("    无异常缺失 ✓")
    lines.append("")

    # 总结
    lines.append("=" * w)
    lines.append(f" 预期总文件: {total_expected}")
    lines.append(f" 实有总文件: {total_found}")
    lines.append(f" 缺失总文件: {total_missing}")
    lines.append(f"   其中维护缺失: {len(maint_missing)} 个")
    lines.append(f"   其中异常缺失: {len(other_missing)} 个")
    if total_expected > 0:
        lines.append(f" 完整率(总):   {total_found/total_expected*100:.2f}%")
        lines.append(f" 完整率(扣维护): {(total_expected-len(other_missing))/total_expected*100:.2f}%")
    lines.append("=" * w)

    return "\n".join(lines)


# ==================== 主流程 ====================
if __name__ == "__main__":
    # 支持命令行参数: python nc完整性检查.py YYYYMM [起始日] [结束日] [输出路径]
    if len(sys.argv) > 1:
        YEAR_MONTH = sys.argv[1]
    if len(sys.argv) > 2:
        DAY_START = int(sys.argv[2])
    if len(sys.argv) > 3:
        DAY_END = int(sys.argv[3])
    if len(sys.argv) > 4:
        OUTPUT_FILE = sys.argv[4]

    print(f"扫描 {DATA_ROOT}/{YEAR_MONTH}  (日期 {DAY_START:02d}~{DAY_END:02d}) ...")

    records = scan_directory(DATA_ROOT, YEAR_MONTH, DAY_START, DAY_END)
    maint_missing, other_missing, hour_stats = check_completeness(records, DAY_START, DAY_END)
    report = generate_report(DATA_ROOT, YEAR_MONTH, DAY_START, DAY_END,
                             maint_missing, other_missing, hour_stats)

    print(report)

