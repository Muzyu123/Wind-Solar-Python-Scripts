import os
import zipfile
import pandas as pd
from io import StringIO
import matplotlib.pyplot as plt
import matplotlib.dates as mdates

# 文件夹路径，替换为你压缩包所在的路径
zip_folder = "/work/zfshu/hyxtest/test1"

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
                        all_dfs.append(df)

# 合并所有 DataFrame（如有必要）
final_df = pd.concat(all_dfs, ignore_index=True)
del final_df['机器号']

filtered_df = final_df[final_df["点位"].astype(str).str.contains("17113", case=True, na=False)]
filtered_df.loc[:, '数据'] = pd.to_numeric(filtered_df['数据'], errors='coerce')

print(filtered_df['数据'])

filtered_df1 = final_df[final_df["点位"].astype(str).str.contains("17115", case=True, na=False)]
filtered_df1.loc[:, '数据'] = pd.to_numeric(filtered_df1['数据'], errors='coerce')

print(filtered_df1['数据'])

# 绘图
plt.scatter(filtered_df1['日期时间'],filtered_df1['数据'], filtered_df['数据'])
plt.xlabel("WindSpeed")
plt.ylabel("Power")
plt.show()