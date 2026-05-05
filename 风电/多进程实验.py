import os
import zipfile
import pandas as pd
from io import StringIO
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
from multiprocessing import Pool, cpu_count

zip_folder = "/work/zfshu/hyxtest/err_test"

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

if __name__ == "__main__":
    zip_files = [os.path.join(zip_folder, name) for name in os.listdir(zip_folder) if name.endswith('.zip')]#获取路径下所有压缩包名
    with Pool(cpu_count()) as pool:#创建进程池
        results = pool.map(process_zip, zip_files)
    # results 是嵌套列表，需要展开
    all_dfs = [df for dfs in results for df in dfs]

    final_df = pd.concat(all_dfs, ignore_index=True)
    del final_df['机器号']

    filtered_df = final_df[final_df["点位"].astype(str).str.contains("16415", case=True, na=False)]
    filtered_df.loc[:,'数据'] = pd.to_numeric(filtered_df['数据'], errors='coerce')
    
    print(filtered_df.head())
    print(filtered_df['数据'])

    filtered_df['日期时间'] = pd.to_datetime(filtered_df['日期时间'], errors='coerce')

    plt.plot(filtered_df['日期时间'], filtered_df['数据'])
    plt.xlabel("time")
    plt.ylabel("data")
    ax = plt.gca()
    ax.xaxis.set_major_locator(mdates.DayLocator(interval=1))
    ax.xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m-%d'))
    plt.xticks(rotation=45)
    plt.show()
