import pandas as pd
import matplotlib.pyplot as plt
import os
import matplotlib.dates as mdates

START_TIME = "2025-11-10 00:00" 
END_TIME = "2025-11-10 23:59"
X_COL = "time1"
Y_COL = "Instantaneous value of horizontal radiation"

df = pd.read_csv("/work/zfshu/24learn/hyx-test/My_py/光伏/数据/EnRaw/HHFK-2025-En.csv")

df[X_COL] = pd.to_datetime(df[X_COL], errors='coerce')

filtered_df = df[(df[X_COL] >= START_TIME) & (df[X_COL] <= END_TIME)]
# 核心修复：剔除值为0或缺失的行
filtered_df = filtered_df.dropna(subset=[Y_COL])  # 剔除缺失值
filtered_df = filtered_df.sort_values(by=X_COL)  # 按时间升序排列
filtered_df = filtered_df.drop_duplicates(subset=[X_COL], keep='first')  # 去重同一时间点的数据

plt.plot(filtered_df[X_COL], filtered_df[Y_COL])
plt.xlabel(X_COL)
plt.ylabel(Y_COL)
ax = plt.gca()

ax.xaxis.set_major_locator(mdates.HourLocator(interval=1))  # 每 1 小时一个主刻度
ax.xaxis.set_major_formatter(mdates.DateFormatter('%H:%M'))  # 设置格式为年-月-日，时：分

#ax.xaxis.set_major_locator(mdates.DayLocator(interval=1))  # 每 1 天一个主刻度
#ax.xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m-%d'))  # 显示年-月-日

plt.xticks(rotation=45)  # 旋转 x 轴标签以便更好地显示
plt.show()