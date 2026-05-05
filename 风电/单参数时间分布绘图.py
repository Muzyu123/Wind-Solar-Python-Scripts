import os
import zipfile
import pandas as pd
from io import StringIO
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import time

stat_time=time.time()

# 文件夹路径，替换为你压缩包所在的路径
zip_folder = "/work/zfshu/24learn/hyx-test/anadata"

# 存储所有 DataFrame
all_dfs = []

# 遍历文件夹中的所有 zip 文件
for zip_name in os.listdir(zip_folder):
    if zip_name.endswith('.zip'):
        zip_path = os.path.join(zip_folder, zip_name)
        
        with zipfile.ZipFile(zip_path, 'r') as zf:
            # 遍历压缩包内的所有文件
            for file_name in zf.namelist():
                if file_name.endswith('.log'):
                    # 读取 .log 文件内容
                    with zf.open(file_name) as f:
                        # 解码为字符串
                        text = f.read().decode('utf-8')
                        # 拆分成行（不保留换行符）
                        lines = text.splitlines()

                        # 插入新内容作为第一行
                        lines.insert(0, "日期时间,机器号,点位,数据\n")

                        # 重新合成为字符串（加回换行符）
                        new_text = "\n".join(lines) + "\n"  # 保证文件末尾也有换行
                        # 将文本转换为 DataFrame
                        # 用 StringIO 伪装成文件对象喂给 read_csv
                        df = pd.read_csv(StringIO(new_text))
                        # all_dfs存储了所有压缩包内的数据
                        all_dfs.append(df)

# 合并所有 DataFrame（如有必要）
# 使用 pandas 的 concat 函数将之前存储在 all_dfs 列表中的所有 DataFrame 合并为一个新的 DataFrame。
# ignore_index=True 参数表示重新设置合并后 DataFrame 的索引，忽略原有的索引，生成从 0 开始的连续整数索引。
final_df = pd.concat(all_dfs, ignore_index=True)
del final_df['机器号']

# 从 final_df 中筛选出 "点位" 列包含字符串 "XXXXX" 的行
# astype(str) 将 "点位" 列转换为字符串类型
# str.contains("XXXXX", case=True, na=False) 检查字符串是否包含 "XXXXX"，区分大小写，忽略缺失值
filtered_df = final_df[final_df["点位"].astype(str).str.contains("18649", case=True, na=False)]
# 将 filtered_df 中的 "数据" 列转换为数值类型
# errors='coerce' 表示当转换失败时，将对应的值设为 NaN
filtered_df.loc[:,'数据'] = pd.to_numeric(filtered_df['数据'], errors='coerce')

# 添加数值范围筛选，假设筛选数据范围在 0 到 100 之间
#filtered_df = filtered_df[filtered_df['数据'].between(-20, 20, inclusive='both')]

# 显示前几行
print(filtered_df.head())
print(filtered_df['数据'])

# 确保日期时间列为 datetime 类型
filtered_df['日期时间'] = pd.to_datetime(filtered_df['日期时间'], errors='coerce')

plt.plot(filtered_df['日期时间'], filtered_df['数据'])
plt.xlabel("time")
plt.ylabel("data")
ax = plt.gca()

ax.xaxis.set_major_locator(mdates.HourLocator(interval=1))  # 每 1 小时一个主刻度
ax.xaxis.set_major_formatter(mdates.DateFormatter('%H:%M'))  # 设置格式为年-月-日，时：分

#ax.xaxis.set_major_locator(mdates.DayLocator(interval=1))  # 每 1 天一个主刻度
#ax.xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m-%d'))  # 显示年-月-日

plt.xticks(rotation=45)  # 旋转 x 轴标签以便更好地显示
plt.show()

end_time = time.time()
print(f"运行时间: {end_time-stat_time}秒")
