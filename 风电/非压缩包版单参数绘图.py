import os
import pandas as pd
from io import StringIO
import matplotlib.pyplot as plt
import matplotlib.dates as mdates

# 文件夹路径，替换为你压缩包所在的路径
folder = "/work/zfshu/24learn/hyx-test/数据分离/filtered_logs"

# 存储所有 DataFrame
all_dfs = []

for file_name in os.listdir(folder):
    if file_name.endswith('.log'):
        # 读取 .log 文件内容
        with open(os.path.join(folder, file_name)) as f:
            # 解码为字符串
            text = f.read()
            # 拆分成行（不保留换行符）
            lines = text.splitlines()
            # 插入新内容作为第一行
            lines.insert(0, "日期时间,机器号,点位,数据\n")

            # 重新合成为字符串（加回换行符）
            new_text = "\n".join(lines) + "\n"  # 保证文件末尾也有换行
            # 将文本转换为 DataFrame
            # 用 StringIO 伪装成文件对象喂给 read_csv
            df = pd.read_csv(StringIO(new_text))
            all_dfs.append(df)

# 合并所有 DataFrame（如有必要）
final_df = pd.concat(all_dfs, ignore_index=True)
del final_df['机器号']

filtered_df = final_df[final_df["点位"].astype(str).str.contains("16416", case=True, na=False)]
filtered_df.loc[:,'数据'] = pd.to_numeric(filtered_df['数据'], errors='coerce')
# 示例：显示前几行

print(filtered_df.head())
print(filtered_df['数据'])

# 确保日期时间列为 datetime 类型
filtered_df['日期时间'] = pd.to_datetime(filtered_df['日期时间'], errors='coerce')

plt.plot(filtered_df['日期时间'], filtered_df['数据'])
plt.xlabel("time")
plt.ylabel("data")
ax = plt.gca()
ax.xaxis.set_major_locator(mdates.HourLocator(interval=1))  # 每 1 小时一个主刻度
ax.xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m-%d, %H:%M'))  # 设置格式为年-月-日，时：分
plt.xticks(rotation=45)  # 旋转 x 轴标签以便更好地显示
plt.show()