import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression

df = pd.read_csv("/work/zfshu/24learn/hyx-test/My_py/光伏/数据/Processed/WXJY/WXJY-2025-Processed.csv")

X = df["radiation"]
y = df["pv_power"]

plt.scatter(X, y, color="blue")
plt.xlabel("Radiation")
plt.ylabel("Power")
plt.show()