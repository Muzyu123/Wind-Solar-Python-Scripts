import os
import zipfile
import pandas as pd
from io import StringIO
import matplotlib.pyplot as plt
import matplotlib.dates as mdates

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
                        all_dfs.append(df)

# 合并所有 DataFrame（如有必要）
final_df = pd.concat(all_dfs, ignore_index=True)
del final_df['机器号']

#Y轴
X = final_df[final_df["点位"].astype(str).str.contains("16413", case=True, na=False)]
X.loc[:, '数据'] = pd.to_numeric(X['数据'], errors='coerce')

print(X.head())
print(X['数据'])

y = final_df[final_df["点位"].astype(str).str.contains("16415", case=True, na=False)]
y.loc[:, '数据'] = pd.to_numeric(y['数据'], errors='coerce')

print(y.head())
print(y['数据'])

# 绘图
plt.scatter(y['数据'], X['数据'])
plt.xlabel("WindSpeed(m/s)")
plt.ylabel("Power(kW)")
plt.show()