import pandas as pd
import matplotlib.pyplot as plt
import os
import matplotlib.dates as mdates

# ====================== 全局配置 ======================
# 数据源1：卫星辐射数据
CSV_PATH_1 = "/work/zfshu/24learn/hyx-test/武穴数据/0831/csv/2025.1.1-2025.12.31_no_rad2.csv"
X_COL_1 = "time"
Y_COL_1 = "pv_power"
LABEL_1 = "p1"

# 数据源2：场站辐射数据
CSV_PATH_2 = "/work/zfshu/24learn/hyx-test/Wind-Solar-Python-Scripts/光伏/光伏场站数据/Processed/WXJY/WXJY-2025-Processed.csv"
X_COL_2 = "time"
Y_COL_2 = "pv_power"
LABEL_2 = "p2"

# 时间筛选
START_TIME = "2025-4-1 00:00"
END_TIME = "2025-4-1 23:00"

# 图表样式配置
CHART_TITLE = f"Himawari vs Wuxue Radiation Data in BJT ({START_TIME} ~ {END_TIME})"  
X_LABEL = "Time (HH:MM)"
Y_LABEL = "Radiation Value"
plt.rcParams["axes.unicode_minus"] = False

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))

# ====================== 通用数据读取函数 ======================
def read_data(csv_path, x_col, y_col, label):
    """通用数据读取函数，适配两个数据源的读取逻辑"""
    try:
        # 1. 检查文件存在性
        if not os.path.exists(csv_path):
            raise FileNotFoundError(f"未找到{label}文件：{csv_path}")
        
        # 2. 多编码兼容读取
        try:
            df = pd.read_csv(csv_path, encoding='utf-8', low_memory=False)
        except UnicodeDecodeError:
            try:
                df = pd.read_csv(csv_path, encoding='gbk', low_memory=False)
            except UnicodeDecodeError:
                df = pd.read_csv(csv_path, encoding='iso-8859-1', low_memory=False)
        
        # 3. 检查列存在性
        if x_col not in df.columns or y_col not in df.columns:
            raise ValueError(f"{label}文件缺少指定列")
        
        # 4. 数据清洗（去空值）
        valid_data = pd.DataFrame({
            'time': df[x_col].dropna(),
            'value': df[y_col].dropna()
        }).dropna()
        
        # 5. 时间转换+过滤无效时间
        valid_data['time'] = pd.to_datetime(valid_data['time'], errors='coerce')
        valid_data = valid_data.dropna(subset=['time'])
        if len(valid_data) == 0:
            raise ValueError(f"{label}无有效时间数据")
        
        # 6. 时间范围筛选
        if START_TIME or END_TIME:
            start_dt = pd.to_datetime(START_TIME) if START_TIME else pd.Timestamp.min
            end_dt = pd.to_datetime(END_TIME) if END_TIME else pd.Timestamp.max
            if start_dt > end_dt:
                raise ValueError("起始时间不能晚于结束时间")
            time_mask = (valid_data['time'] >= start_dt) & (valid_data['time'] <= end_dt)
            valid_data = valid_data[time_mask]
            if len(valid_data) == 0:
                raise ValueError(f"{label}在指定时间段内无有效数据")
        
        # 7. Y值转数值型
        valid_data['value'] = pd.to_numeric(valid_data['value'], errors='coerce')
        valid_data = valid_data.dropna(subset=['value'])
        if len(valid_data) == 0:
            raise ValueError(f"{label}Y轴数据无法转换为数值")
        
        return valid_data['time'], valid_data['value']
    
    except Exception as e:
        print(f"读取{label}出错：{e}")
        return None, None

# ====================== 核心绘图逻辑（同图绘制两条曲线） ======================
def plot_combined_chart():
    # 1. 读取两个数据源
    time1, value1 = read_data(CSV_PATH_1, X_COL_1, Y_COL_1, LABEL_1)
    time2, value2 = read_data(CSV_PATH_2, X_COL_2, Y_COL_2, LABEL_2)
    
    if time1 is None or time2 is None:
        print("数据读取失败，终止绘图")
        return
    
    # 2. 创建唯一的绘图轴对象（核心：所有曲线画在同一个图上）
    fig, ax = plt.subplots(figsize=(12, 7))
    
    # 3. 绘制卫星数据：蓝色系（散点+折线）
    # ax.scatter(time1, value1, color='cornflowerblue', label=f'{LABEL_1} - Himawari-datapoint', alpha=0.7)
    sorted_1 = pd.DataFrame({'time': time1, 'value': value1}).sort_values('time')
    ax.plot(sorted_1['time'], sorted_1['value'], color='blue', label=f'{LABEL_1} - {Y_COL_1}', linewidth=1.5)
    
    # 4. 绘制场站数据：橙色系（散点+折线，和蓝色明显区分）
    # ax.scatter(time2, value2, color='orange', label=f'{LABEL_2} - WXJY-datapoint', alpha=0.7)
    sorted_2 = pd.DataFrame({'time': time2, 'value': value2}).sort_values('time')
    ax.plot(sorted_2['time'], sorted_2['value'], color='darkorange', label=f'{LABEL_2} - {Y_COL_2}', linewidth=1.5)
    
    # 5. 图表样式
    ax.set_title(CHART_TITLE, fontsize=14)
    ax.set_xlabel(X_LABEL, fontsize=12)
    ax.set_ylabel(Y_LABEL, fontsize=12)
    ax.legend(loc='upper right')
    ax.grid(True, alpha=0.3)
    
    # 智能设置X轴刻度（兼容不同时间跨度）
    all_times = pd.concat([time1, time2])
    time_range_hours = (all_times.max() - all_times.min()).total_seconds() / 3600
    if time_range_hours > 100:
        day_interval = max(1, int(time_range_hours / 24 / 10))
        ax.xaxis.set_major_locator(mdates.DayLocator(interval=day_interval))
        ax.xaxis.set_major_formatter(mdates.DateFormatter('%m-%d %H:%M'))
    else:
        hour_interval = max(1, int(time_range_hours / 10))
        ax.xaxis.set_major_locator(mdates.HourLocator(interval=hour_interval))
        ax.xaxis.set_major_formatter(mdates.DateFormatter('%H:%M'))
    
    # 纵轴从数据最小值开始+稀疏刻度（兼容不同范围）
    all_values = pd.concat([value1, value2])
    y_min = all_values.min()
    y_max = all_values.max()
    y_range = y_max - y_min
    step = max(1, round(y_range / 6))
    ax.set_ylim(bottom=y_min)
    ax.set_yticks(range(int(y_min), int(y_max + step), int(step)))
    
    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.savefig(os.path.join(SCRIPT_DIR, "卫星vs场站.png"))
    plt.show()
    
    # 打印数据量信息
    print(f"\n绘图完成！")
    print(f"{LABEL_1}有效行数：{len(time1)}")
    print(f"{LABEL_2}有效行数：{len(time2)}")

# ====================== 执行绘图 ======================
if __name__ == "__main__":
    plot_combined_chart()