# 安装依赖（如在 Colab 中未预装）
!pip install mpmath

import mpmath
import cmath
import math
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.optimize import dual_annealing
from scipy.stats import linregress

# 使用前 20 个素数
p = [2, 3, 5, 7, 11, 13, 17, 19, 23, 29,
     31, 37, 41, 43, 47, 53, 59, 61, 67, 71]
A = [-1 / math.sqrt(p_n) for p_n in p]  # 固定振幅模型 A_n = -1/sqrt(p_n)

# 选取前 5 个黎曼 ζ 零点的虚部 t 值（来源：Riemann Zeros 数据）
riemann_zeros = [14.134725, 21.022040, 25.010858, 30.424876, 32.935061]

# 定义结构函数 φ(s)
def phi_s(theta, x):
    return sum(A[n] * cmath.exp(1j * (math.log(p[n]) * cmath.log(x) + theta[n])) for n in range(len(p)))

# 耦合比率 K 计算函数
def compute_K_structure(phi_func, zeta_s, theta_opt, x, num_samples=40, epsilon=0.01):
    logs_phi, logs_delta = [], []
    for _ in range(num_samples):
        perturb = np.random.normal(0, epsilon, size=len(theta_opt))
        theta_perturbed = theta_opt + perturb
        phi_val = phi_func(theta_perturbed, x)
        delta_val = abs(phi_val - zeta_s)
        logs_phi.append(np.log(abs(phi_val)))
        logs_delta.append(np.log(delta_val))
    slope, _, r_value, _, _ = linregress(logs_phi, logs_delta)
    return slope, r_value**2

# 主循环：针对多个零点 s = 0.5 + i t
results = []
for t0 in riemann_zeros:
    s = 0.5 + t0 * 1j
    x = cmath.exp(s)
    zeta_s = complex(mpmath.zeta(s))  # 转为复数类型

    theta_initial = [-t0 * math.log(p_n) for p_n in p]
    phi_initial = phi_s(theta_initial, x)
    residual_initial = abs(phi_initial - zeta_s)

    bounds = [(-2 * math.pi, 2 * math.pi)] * len(p)
    result = dual_annealing(lambda theta: abs(phi_s(theta, x))**2, bounds)
    theta_opt = result.x
    phi_opt = phi_s(theta_opt, x)
    residual_opt = abs(phi_opt - zeta_s)

    # 耦合比率计算
    K_val, R2 = compute_K_structure(phi_s, zeta_s, theta_opt, x)

    results.append({
        "Zero t": t0,
        "Initial Residual": residual_initial,
        "Optimized Residual": residual_opt,
        "K Coupling Ratio": K_val,
        "R² (log-log)": R2
    })

# 显示结果表格
df = pd.DataFrame(results)
print("结构函数拟合结果与耦合比率（前 5 个零点）")
display(df)

# 可视化示例：第一个零点 log-log 拟合图
K_val, R2 = compute_K_structure(phi_s, complex(mpmath.zeta(0.5 + riemann_zeros[0]*1j)), theta_opt, cmath.exp(0.5 + riemann_zeros[0]*1j))
logs_phi = []
logs_delta = []
for _ in range(40):
    perturb = np.random.normal(0, 0.01, size=len(theta_opt))
    theta_perturbed = theta_opt + perturb
    phi_val = phi_s(theta_perturbed, cmath.exp(0.5 + riemann_zeros[0]*1j))
    delta_val = abs(phi_val - complex(mpmath.zeta(0.5 + riemann_zeros[0]*1j)))
    logs_phi.append(np.log(abs(phi_val)))
    logs_delta.append(np.log(delta_val))

plt.figure(figsize=(6,4))
plt.scatter(logs_phi, logs_delta, label="Samples", alpha=0.6)
xline = np.linspace(min(logs_phi), max(logs_phi), 100)
plt.plot(xline, K_val * xline + np.mean(logs_delta) - K_val * np.mean(logs_phi),
         color='red', label=f"K ≈ {K_val:.3f}")
plt.xlabel("log |φ(s)|")
plt.ylabel("log |φ(s) - ζ(s)|")
plt.title("Structure Coupling Ratio K")
plt.grid(True)
plt.legend()
plt.tight_layout()
plt.show()
