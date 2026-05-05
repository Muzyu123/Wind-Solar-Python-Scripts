import os
import time
import zipfile
import pandas as pd
from io import StringIO
import matplotlib.dates as mdates
import numpy as np
import matplotlib.pyplot as plt
from multiprocessing import Pool, cpu_count

# 记录程序开始时间
start_time = time.time()

zip_folder = "/work/zfshu/24learn/hyx-test/风电场数据分析/data/2023-09"

def process_zip(zip_path):#处理单个压缩包
    dfs = []
    with zipfile.ZipFile(zip_path, 'r') as zf:
        for file_name in zf.namelist():
            if file_name.endswith('.log'):
                with zf.open(file_name) as f:
                    text = f.read().decode('utf-8')
                    lines = text.splitlines()
                    lines.insert(0, "日期时间,机器号,点位,数据\n")
                    new_text = "\n".join(lines) + "\n"
                    df = pd.read_csv(StringIO(new_text))
                    dfs.append(df)
    return dfs


zip_files = [os.path.join(zip_folder, name) for name in os.listdir(zip_folder) if name.endswith('.zip')]#获取路径下所有压缩包名
with Pool(cpu_count()) as pool:#创建进程池
    results = pool.map(process_zip, zip_files)
# results 是嵌套列表，需要展开
all_dfs = [df for dfs in results for df in dfs]

final_df = pd.concat(all_dfs, ignore_index=True)
del final_df['机器号']

y = final_df[final_df["点位"].astype(str).str.contains("18639", case=True, na=False)]
y.loc[:, '数据'] = pd.to_numeric(y['数据'], errors='coerce')

# 查找 y 中 '数据' 列是否有缺失值
missing_values_y = y['数据'].isna().sum()
print(f"y 中 '数据' 列的缺失值数量: {missing_values_y}")
# 将 '数据' 列中的 NaN 值替换为 1.0
y.loc[:, '数据'] = y['数据'].fillna(1.0)

X = final_df[final_df["点位"].astype(str).str.contains("18647", case=True, na=False)]
X.loc[:, '数据'] = pd.to_numeric(X['数据'], errors='coerce')

# 查找 X 中 '数据' 列是否有缺失值
missing_values_X = X['数据'].isna().sum()
print(f"X 中 '数据' 列的缺失值数量: {missing_values_X}")
# 将 '数据' 列中的 NaN 值替换为 1.0
X.loc[:, '数据'] = X['数据'].fillna(1.0)

# y['日期时间'] = pd.to_datetime(y['日期时间'])  # 确保日期时间格式正确
y.set_index('日期时间', inplace=True)  # 设置Date列为行索引
# X['日期时间'] = pd.to_datetime(X['日期时间'])  # 确保日期时间格式正确
X.set_index('日期时间', inplace=True)  # 设置Date列为行索引

#调试
print(y.head())
print(y['数据'])
print(X.head())
print(X['数据'])

# 首先，使用 y.index 获取经过筛选后 y 数据的索引
# 接着，使用 X.index.isin(y.index) 检查 X 的每个索引是否存在于 y 的索引中，返回一个布尔数组
# 该布尔数组标记了 X 中哪些索引在 y 中存在（True），哪些不存在（False）
# 最后，使用 X.loc[...] 根据这个布尔数组筛选 X 数据，仅保留索引存在于 y 中的那些行
# 这样做的目的是确保 X 和 y 的数据在索引上保持一致，方便后续的模型训练和分析
# 但仅仅是单方面筛选 X
X = X.loc[X.index.isin(y.index)]

#print(X.head())
#print(X['数据'])

#双向交集过滤筛选 X 和 y 中共同存在的索引
common_index = X.index.intersection(y.index)
X = X.loc[common_index]
y = y.loc[common_index]

common_index = X.index.intersection(y.index)
if len(common_index) == 0:
    raise ValueError("X 和 y 没有共同的索引，请检查数据对齐")
# 继续后续处理（如 X = X.loc[common_index]）

print("X 样本数:", X.shape[0], "y 样本数:", y.shape[0])

# 绘图
plt.scatter(X['数据'], y['数据'], alpha=0.5 , s = 8)
plt.xlabel("WindSpeed(m/s)")
plt.ylabel("Power(kW)")
plt.show()

end_time = time.time()
run_time = end_time - start_time
print(run_time)