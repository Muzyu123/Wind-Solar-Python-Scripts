import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))

df = pd.read_csv("/work/zfshu/24learn/hyx-test/My_py/光伏/数据/Processed/WXJY/WXJY-2025-Processed.csv")

X = df["radiation"]
y = df["pv_power"]

plt.scatter(X, y, color="green", alpha=0.5, s=5)
plt.xlabel("Radiation")
plt.ylabel("Power")
plt.show()
plt.savefig(os.path.join(SCRIPT_DIR, "scatter.png"), dpi=300)

print(f"图像已保存为: {os.path.join(SCRIPT_DIR, 'scatter.png')}")