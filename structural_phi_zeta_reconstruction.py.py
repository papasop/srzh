import mpmath
import cmath
import math
import numpy as np
from scipy.optimize import dual_annealing

# 参数设置
p = [2, 3, 5, 7, 11, 13, 17, 19, 23, 29, 31, 37, 41, 43, 47, 53, 59, 61, 67, 71, 73, 79, 83, 89, 97]
t_k = [14.134725, 21.022039, 25.010857]
A = [-1 / p_n**0.5 for p_n in p]  # A_n = mu(n)/p_n^0.5

# 计算 phi(s)
def phi_s(theta, s):
    x = cmath.exp(s)
    result = 0.0
    for n in range(len(p)):
        result += A[n] * cmath.exp(1j * (math.log(p[n]) * cmath.log(x) + theta[n]))
    return result

# 目标函数
def objective(theta, s):
    return abs(phi_s(theta, s))**2

# 测试每个零点
for t in t_k:
    s = 0.5 + t * 1j
    theta_initial = [-t * math.log(p_n) for p_n in p]
    phi_initial = phi_s(theta_initial, s)
    zeta_s = mpmath.zeta(s)
    residual_initial = abs(phi_initial - zeta_s)
    bounds = [(-2*math.pi, 2*math.pi)] * len(p)
    result = dual_annealing(lambda theta: objective(theta, s), bounds)
    theta_opt = result.x
    phi_opt = phi_s(theta_opt, s)
    residual_opt = abs(phi_opt - zeta_s)
    print(f"\ns = 0.5 + {t}j, x = e^(s)")
    print(f"Initial phi(s) = {phi_initial}")
    print(f"Optimized phi(s) = {phi_opt}")
    print(f"ζ(s) = {zeta_s}")
    print(f"Initial absolute residual = {residual_initial}")
    print(f"Optimized absolute residual = {residual_opt}")
