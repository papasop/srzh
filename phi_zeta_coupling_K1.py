import mpmath
import cmath
import math
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.optimize import dual_annealing
from scipy.stats import linregress

# Set mpmath precision
mpmath.mp.dps = 50

# First 20 primes
p = [2, 3, 5, 7, 11, 13, 17, 19, 23, 29,
     31, 37, 41, 43, 47, 53, 59, 61, 67, 71]
A = [-1 / mpmath.sqrt(p_n) for p_n in p]  # Fixed amplitude

# First 5 Riemann zeros
riemann_zeros = [14.134725, 21.022040, 25.010858, 30.424876, 32.935061]

# Phase function: theta_n = α log(p) + β p + γ mod 2π
def generate_theta(params, p):
    alpha, beta, gamma = params
    return [(alpha * math.log(p_n) + beta * p_n + gamma) % (2 * math.pi) for p_n in p]

# Structural function φ(s)
def phi_s(params, x):
    try:
        theta = generate_theta(params, p)
        return sum(A[n] * mpmath.exp(1j * (mpmath.log(p[n]) * mpmath.log(x) + theta[n]))
                   for n in range(len(p)))
    except:
        return mpmath.mpc(0, 0)

# Objective function: |φ(s) - ζ(s)|²
def objective(params, x, zeta_s):
    phi_val = phi_s(params, x)
    return float(abs(phi_val - zeta_s)**2)

# Coupling slope K, R² between log|φ| and log|δ|
def compute_K_structure(phi_func, zeta_s, params_opt, x, num_samples=40, epsilon=0.01):
    logs_phi, logs_delta = [], []
    for _ in range(num_samples):
        perturb = np.random.normal(0, epsilon, size=len(params_opt))
        params_perturbed = params_opt + perturb
        try:
            phi_val = phi_func(params_perturbed, x)
            delta_val = abs(phi_val - zeta_s)
            if abs(phi_val) > 1e-10 and delta_val > 1e-10:
                logs_phi.append(float(mpmath.log(abs(phi_val))))
                logs_delta.append(float(mpmath.log(delta_val)))
        except:
            continue
    if len(logs_phi) < 2:
        return np.nan, np.nan
    slope, _, r_value, _, _ = linregress(logs_phi, logs_delta)
    return slope, r_value**2

# 主程序
results = []
for t0 in riemann_zeros:
    s = mpmath.mpc(0.5, t0)
    zeta_val = mpmath.zeta(s)

    bounds = [(-5, 5), (-0.1, 0.1), (0, 2 * math.pi)]
    result = dual_annealing(objective, bounds, args=(mpmath.exp(t0), zeta_val), maxiter=100)

    opt_params = result.x
    phi_val = phi_s(opt_params, mpmath.exp(t0))
    error = abs(phi_val - zeta_val)

    K_slope, R2 = compute_K_structure(phi_s, zeta_val, opt_params, mpmath.exp(t0))
    results.append({
        't': t0,
        '|ζ(1/2 + it)|': float(abs(zeta_val)),
        '|φ(t)|': float(abs(phi_val)),
        'error': float(error),
        'K slope': K_slope,
        'R²': R2,
        'α': opt_params[0],
        'β': opt_params[1],
        'γ': opt_params[2] % (2 * math.pi)
    })

# 输出结果表
df = pd.DataFrame(results)
print(df.to_string(index=False))
