import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import numpy as np
from sklearn.svm import OneClassSVM
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import PolynomialFeatures
from sklearn.metrics import mean_squared_error,mean_absolute_error, r2_score

FILE_NAME = "/work/zfshu/24learn/hyx-test/My_py/光伏/数据/Processed/SSNK/SSNK-2024-Processed.csv"

X = pd.read_csv(FILE_NAME, encoding='utf-8')[['radiation']]
y = pd.read_csv(FILE_NAME, encoding='utf-8')[['pv_power']]

# 绘图
plt.scatter(X, y, alpha=0.5)
plt.xlabel("Radiation")
plt.ylabel("Power(kW)")
plt.show()

# 构建DataFrame
OCdf = pd.DataFrame({'radiation': X["radiation"], 'power': y["pv_power"]})

print(OCdf.head())

# 检查处理后的数据是否为空
if len(OCdf) == 0:
    print("警告：处理缺失值后数据为空，请检查数据质量。")
else:
    # 训练模型
    Y = OCdf[['radiation', 'power']].values

# 处理缺失值
imputer = SimpleImputer(strategy='mean')  # ！！！使用均值填充缺失值！！！
Y_imputed = imputer.fit_transform(Y)

# 标准化可以提高效果（可选）
scaler = StandardScaler()
Y_scaled = scaler.fit_transform(Y_imputed)

# One-Class SVM 模型
model = OneClassSVM(kernel='rbf', gamma='auto', nu=0.08)
model.fit(Y_scaled)

# 预测：+1 正常，-1 异常
OCdf['label'] = model.predict(Y_scaled)
OCdf_normal = OCdf[OCdf['label'] == 1]
OCdf_outlier = OCdf[OCdf['label'] == -1]

plt.figure(figsize=(10, 6))
if len(OCdf_normal) > 0:
    plt.scatter(OCdf_normal['radiation'], OCdf_normal['power'], c='green', label='Normal Data', alpha=0.5)#alpha为透明度
if len(OCdf_outlier) > 0:
    plt.scatter(OCdf_outlier['radiation'], OCdf_outlier['power'], c='red', label='Abnormal Data', marker='x')
plt.xlabel('Radiation')
plt.ylabel('Power (kW)')
plt.title('Radiation-Power Relationship Graph:Isolation Forest Exception Detection')
plt.legend()
plt.grid(True)
plt.show()

#调试
print(OCdf_normal.head())

# 以 OCdf_normal 的 radiation 和 power 列数据分别作为 x 坐标和 y 坐标绘图
plt.figure(figsize=(10, 6))
plt.scatter(OCdf_normal['radiation'], OCdf_normal['power'],alpha=0.5)
plt.xlabel('Radiation')
plt.ylabel('Power (kW)')
plt.title('Radiation-Power Relationship of Normal Data')
plt.show()

X=OCdf_normal[['radiation']]
y=OCdf_normal[['power']]

print(X.head())
print(y.head())

if X.shape[0] == 0 or y.shape[0] == 0:
    print("警告：数据为空，跳过多项式特征生成")
else:
    poly = PolynomialFeatures(degree=5)#！！！拟合方程次数！！！
    X_poly = poly.fit_transform(X)  # 确保只使用 '数据' 列进行转换

# 使用线性回归模型训练多项式特征
model = LinearRegression()
model.fit(X_poly, y)

# 可视化原始数据和拟合结果
# 为了绘图，需要对 X 进行排序
sorted_indices = X.values.flatten().argsort()
X_sorted = X.values[sorted_indices]
X_poly_sorted = poly.transform(X_sorted)
y_pred_sorted = model.predict(X_poly_sorted)
y_sorted = y.loc[X.iloc[sorted_indices].index]['power'].values  # 获取排序后对应的 y 值

plt.scatter(X_sorted.flatten(), y_sorted, color='black', label='True Data')  # 原始数据点
plt.plot(X_sorted.flatten(), y_pred_sorted.flatten(), color='blue', linewidth=3, label='Fitting Curve')  # 拟合曲线
plt.xlabel('Radiation')
plt.ylabel('Fan Power')
plt.legend()
plt.show()

# 获取模型的系数和截距
coefficients = model.coef_[0]
intercept = model.intercept_[0]

# 打印拟合函数
print("拟合函数: ", end="")
terms = []
# 处理截距项
if intercept != 0:
    terms.append(f"{intercept:.4f}")
# 处理多项式项
for i, coef in enumerate(coefficients[1:], start=1):
    if coef != 0:
        if i == 1:
            terms.append(f"{coef:.4f}*x")
        else:
            terms.append(f"{coef:.4f}*x^{i}")

print(" + ".join(terms))

# 预测所有数据
y_pred = model.predict(X_poly)

# 计算 MSE（均方误差）
mse = mean_squared_error(y, y_pred)
# 计算 RMSE（均方根误差）
rmse = np.sqrt(mse)
# 计算 MAE（平均绝对误差）
mae = mean_absolute_error(y, y_pred)
# 计算 R² 分数
r2 = r2_score(y, y_pred)
# 计算 MAPE（平均绝对百分比误差）
def mean_absolute_percentage_error(y_true, y_pred): 
    return np.mean(np.abs((y_true - y_pred) / y_true)) * 100
mape = mean_absolute_percentage_error(y.values, y_pred)

print(f"均方误差(MSE): {mse:.4f}")
print(f"均方根误差(RMSE): {rmse:.4f}")
print(f"平均绝对误差(MAE): {mae:.4f}")
print(f"决定系数(R²): {r2:.4f}")
print(f"平均绝对百分比误差(MAPE): {mape:.4f}%")