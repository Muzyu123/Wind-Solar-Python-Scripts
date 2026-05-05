import pandas as pd
import sys

INPUT_CSV = "/work/zfshu/24learn/hyx-test/My_py/光伏/数据/EnRaw/HHFK-2024-En.csv"  # 输入文件路径
OUTPUT_CSV = "/work/zfshu/24learn/hyx-test/My_py/光伏/数据/Processed/HHFK/HHFK-2024-Processed.csv"  # 输出文件路径
COLUMNS_TO_REMOVE = ["time1", "Instantaneous value of horizontal radiation", "time2", "Instantaneous value of inclined radiation" , "time3" , "Instantaneous value of scattered radiation"]  # 要删除的列名列表

# ==================== 删除指定列 ====================
raw_df = pd.read_csv(INPUT_CSV, encoding="utf-8")

# 检查要删除的列是否存在
existing_columns = raw_df.columns.tolist()
columns_not_found = [col for col in COLUMNS_TO_REMOVE if col not in existing_columns]

if columns_not_found:
    print(f"警告: 以下列未找到: {columns_not_found}")
    print(f"现有列: {existing_columns}")

# 删除指定列
dropped_df = raw_df.drop(columns=[col for col in COLUMNS_TO_REMOVE if col in existing_columns])

dropped_df.columns = [
    "time_radiation", "radiation",
    "time_windspeed", "wind_speed",
    "time_winddirection" , "wind_direction",
    "time_envtemp", "env_temperature",
    "time_pressure", "pressure",
    "time_humidity", "humidity",
    "time_comptemp", "component_temperature",
    "time_power", "pv_power"
]

# 统一转为 datetime
time_cols = [
    "time_radiation", 
    "time_windspeed", 
    "time_winddirection", 
    "time_envtemp",
    "time_pressure",
    "time_humidity",  
    "time_comptemp", 
    "time_power" 
]
for c in time_cols:
    dropped_df[c] = pd.to_datetime(dropped_df[c], errors="coerce")

# 分别提取每个变量 
radiation_df = dropped_df[["time_radiation", "radiation"]].dropna().rename(columns={"time_radiation": "time"})
wind_df = dropped_df[["time_windspeed", "wind_speed"]].dropna().rename(columns={"time_windspeed": "time"})
wind_direction_df = dropped_df[["time_winddirection", "wind_direction"]].dropna().rename(columns={"time_wind_direction": "time"})
envtemp_df = dropped_df[["time_envtemp", "env_temperature"]].dropna().rename(columns={"time_envtemp": "time"})
press_df = dropped_df[["time_pressure", "pressure"]].dropna().rename(columns={"time_pressure": "time"})
humid_df = dropped_df[["time_humidity", "humidity"]].dropna().rename(columns={"time_humidity": "time"})
component_temp_df = dropped_df[["time_comptemp", "component_temperature"]].dropna().rename(columns={"time_comptemp": "time"})
power_df = dropped_df[["time_power", "pv_power"]].dropna().rename(columns={"time_power": "time"})

print(f"基准功率时间点数量：{len(power_df)}")

# 以 time_power 为基准
merged = power_df.copy()

# 逐个精确 merge
merged = merged.merge(radiation_df, on="time", how="left")
merged = merged.merge(wind_df, on="time", how="left")
merged = merged.merge(envtemp_df, on="time", how="left")
merged = merged.merge(press_df, on="time", how="left")
merged = merged.merge(humid_df, on="time", how="left")
merged = merged.merge(component_temp_df, on="time", how="left")

print(f"精确时间匹配后总行数：{len(merged)}")

# 去掉有缺失的行
aligned_final = merged.dropna().copy()
print(f"所有变量同时存在的有效数据：{len(aligned_final)} 行")

# 保存对齐后 CSV
aligned_final.to_csv(OUTPUT_CSV, index=False, encoding="utf-8")
print(f"对齐后数据已保存：{OUTPUT_CSV}")

