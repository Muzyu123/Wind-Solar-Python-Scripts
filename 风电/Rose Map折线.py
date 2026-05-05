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

zip_folder = "/work/zfshu/24learn/onemonth"

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

y = final_df[final_df["点位"].astype(str).str.contains("16416", case=True, na=False)]
y.loc[:, '数据'] = pd.to_numeric(y['数据'], errors='coerce')

# 查找 y 中 '数据' 列是否有缺失值
missing_values_y = y['数据'].isna().sum()
print(f"y 中 '数据' 列的缺失值数量: {missing_values_y}")
# 将 '数据' 列中的 NaN 值替换为 1.0
#y.loc[:, '数据'] = y['数据'].fillna(1.0)

X = final_df[final_df["点位"].astype(str).str.contains("16415", case=True, na=False)]
X.loc[:, '数据'] = pd.to_numeric(X['数据'], errors='coerce')

# 查找 X 中 '数据' 列是否有缺失值
missing_values_X = X['数据'].isna().sum()
print(f"X 中 '数据' 列的缺失值数量: {missing_values_X}")
# 将 '数据' 列中的 NaN 值替换为 1.0
#X.loc[:, '数据'] = X['数据'].fillna(1.0)

y['日期时间'] = pd.to_datetime(y['日期时间'])  # 确保日期时间格式正确
y.set_index('日期时间', inplace=True)  # 设置Date列为行索引
X['日期时间'] = pd.to_datetime(X['日期时间'])  # 确保日期时间格式正确
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

# 提取风向和风速数据
wind_direction = y['数据']
wind_speed = X['数据']

# 定义风向区间，将 360 度划分为 16 个区间
# 定义风向区间的数量，这里将 360 度的风向划分为 16 个区间
num_directions = 16
# 使用 np.linspace 函数生成风向区间的边界值
# 从 0 度开始，到 360 度结束，总共生成 num_directions + 1 个均匀分布的点
# 这些点将作为后续使用 pd.cut 函数时的区间边界
direction_bins = np.linspace(0, 360, num_directions + 1)
# 使用 np.arange 函数生成每个风向区间的标签
# 从 0 度开始，以 360/num_directions 为步长，生成到 360 度之前的所有区间起始角度
# 这些角度将作为每个风向区间的标识
direction_labels = np.arange(0, 360, 360/num_directions)

# 统计各风向区间出现的频率
# 使用 pd.cut 函数将风向数据 wind_direction 进行区间划分
# bins 参数指定区间的边界，这里使用之前定义的 direction_bins 数组
# labels 参数指定每个区间的标签，使用之前定义的 direction_labels 数组
# right=False 表示区间为左闭右开，即每个区间包含左边界值，不包含右边界值
# 划分结果存储在 wind_direction_categories 中
wind_direction_categories = pd.cut(wind_direction, bins=direction_bins, labels=direction_labels, right=False)
# 对划分好区间的风向数据 wind_direction_categories 进行计数统计
# value_counts() 函数会统计每个区间出现的次数
# sort_index() 函数会按照区间标签的索引顺序对统计结果进行排序
# 最终得到每个风向区间的出现次数，存储在 direction_counts 中
direction_counts = wind_direction_categories.value_counts().sort_index()
# 计算每个风向区间的出现频率
# 将每个风向区间的出现次数 direction_counts 除以所有区间出现次数的总和 direction_counts.sum()
# 得到每个风向区间的出现频率，存储在 direction_frequencies 中
direction_frequencies = direction_counts / direction_counts.sum()

# 打印各风向出现频率
print("各风向出现频率:")
for direction, freq in zip(direction_labels, direction_frequencies):
    print(f"{direction} 度: {freq * 100:.2f}%")

# 绘制风玫瑰图
fig = plt.figure(figsize=(8, 8))
# 使用 fig.add_subplot() 方法在图形中添加一个子图
# 参数 111 表示将图形划分为 1 行 1 列，并选择第 1 个子图位置
# polar=True 表示创建一个极坐标子图，用于后续绘制风玫瑰图
ax = fig.add_subplot(111, polar=True)
# 将角度转换为弧度
# 之前定义的 direction_labels 数组中存储了每个风向区间的起始角度
# np.deg2rad() 函数将这些角度转换为弧度值，用于后续绘制折线图
angles = np.deg2rad(direction_labels)

# 为了使折线图闭合，将第一个点添加到数据末尾
closed_angles = np.append(angles, angles[0])
closed_frequencies = np.append(direction_frequencies, direction_frequencies[0])

# 使用 ax.plot() 方法在极坐标子图上绘制折线图
ax.plot(closed_angles, closed_frequencies, marker='o', linestyle='-')

# 填充折线图内部区域
ax.fill(closed_angles, closed_frequencies, alpha=0.25)

# 设置图形属性
# 设置极坐标图中 0 度的起始位置为北方（North），让风玫瑰图的 0 度方向对应实际的正北方向
ax.set_theta_zero_location('N')  # 0 度指向北方
# 设置极坐标图的角度增加方向为顺时针方向，因为在风玫瑰图中，风向通常按顺时针方向表示
ax.set_theta_direction(-1)  # 顺时针方向
# 设置极坐标图的角度刻度线和对应的标签
# direction_labels 作为刻度线的位置，使用列表推导式将每个刻度值转换为带度数符号的字符串作为标签
ax.set_thetagrids(direction_labels, labels=[f'{int(d)}°' for d in direction_labels])
# 设置极坐标图的标题，表明该图为风玫瑰图，展示各风向出现的频率
ax.set_title('Wind Rose Diagram - Frequency of Each Wind Direction')

plt.show()

end_time = time.time()
run_time = end_time - start_time
print(f"程序运行时间: {run_time:.4f} 秒")
