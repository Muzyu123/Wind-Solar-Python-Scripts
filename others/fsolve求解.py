import numpy as np
from scipy.optimize import fsolve

def f(θ):
    θ = np.radians(θ)
    return np.sin(2*θ - np.radians(50)) / np.sin(θ) - 6373/7373

result = fsolve(f, 30)[0]
print("θ =", round(result, 2), "度")

# 使用 fsolve 函数求解方程 f(θ) = 0
# 参数说明：
#   f: 目标函数，即上面定义的方程
#   30: 初始猜测值（单位：度），fsolve 会从这个值开始迭代寻找根
# [0]: fsolve 返回的是一个数组，取第一个元素得到标量结果