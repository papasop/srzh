import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score

# -------- 设置参数 --------
N = 20  # 零点个数
primes = np.array([2, 3, 5, 7, 11, 13, 17, 19, 23, 29])
log_p = np.log(primes)
K = len(log_p)
n_vals = np.arange(1, N + 1)

# -------- 构造模拟 α, β, γ --------
np.random.seed(42)
noise = lambda s: np.random.normal(0, s, N)
alpha = ((-1)**n_vals) * 0.98 + noise(0.005)
beta  = ((-1)**n_vals) * 0.34 + noise(0.005)
gamma = ((-1)**n_vals) * 0.23 + noise(0.005)

# -------- 构建混合特征矩阵 --------
def build_combined_feature_matrix(series, n_vals, log_p):
    phi_n   = series[1:-1]
    phi_n1  = series[:-2]
    n_core  = n_vals[2:]
    cos_matrix = np.column_stack([np.cos(n_core * w) for w in log_p])
    sin_matrix = np.column_stack([np.sin(n_core * w) for w in log_p])
    X = np.column_stack((phi_n, phi_n1, cos_matrix, sin_matrix))
    y = series[2:]
    return X, y

# -------- 混合模型拟合 --------
def fit_hybrid_model(series, n_vals, log_p, name='φ'):
    X, y = build_combined_feature_matrix(series, n_vals, log_p)
    model = LinearRegression().fit(X, y)
    y_pred = model.predict(X)
    
    # 提取参数
    a, b = model.coef_[:2]
    A_k = model.coef_[2:2+len(log_p)]
    B_k = model.coef_[2+len(log_p):]
    c = model.intercept_
    
    rmse = np.sqrt(mean_squared_error(y, y_pred))
    r2 = r2_score(y, y_pred)
    
    print(f"\n🔬 混合模型拟合（{name}）:")
    print(f"{name}ₙ₊₁ = {a:.4f}·{name}ₙ + {b:.4f}·{name}ₙ₋₁ + 共振项 + {c:.4f}")
    print(f"RMSE: {rmse:.6f} | R²: {r2:.5f}")
    
    return a, b, A_k, B_k, c, rmse, r2, y_pred

# -------- 提取振幅 & 相位 --------
def amp_phase(A, B):
    amps = np.sqrt(A**2 + B**2)
    phases = np.arctan2(-B, A)
    return amps, phases

# -------- 运行 α, β, γ --------
a_a, b_a, Ak_a, Bk_a, c_a, rmse_a, r2_a, pred_a = fit_hybrid_model(alpha, n_vals, log_p, name="α")
a_b, b_b, Ak_b, Bk_b, c_b, rmse_b, r2_b, pred_b = fit_hybrid_model(beta, n_vals, log_p, name="β")
a_g, b_g, Ak_g, Bk_g, c_g, rmse_g, r2_g, pred_g = fit_hybrid_model(gamma, n_vals, log_p, name="γ")

amps_a, phases_a = amp_phase(Ak_a, Bk_a)
amps_b, phases_b = amp_phase(Ak_b, Bk_b)
amps_g, phases_g = amp_phase(Ak_g, Bk_g)

# -------- 可视化 γ 拟合 --------
plt.figure(figsize=(10,4))
plt.plot(n_vals[2:], gamma[2:], label="真实 γₙ₊₁", marker='o')
plt.plot(n_vals[2:], pred_g, label="混合预测 γₙ₊₁", linestyle='--', marker='x')
plt.title("γ 混合模型拟合结果")
plt.xlabel("n"); plt.ylabel("γ 值")
plt.legend(); plt.grid(True); plt.tight_layout(); plt.show()

# -------- 奇偶交替检测可视化 --------
def plot_sign_flip(series, name):
    signs = np.sign(series)
    plt.figure(figsize=(10,2))
    plt.stem(n_vals, signs, basefmt=" ", use_line_collection=True)
    plt.title(f"{name} 符号序列 (+1 / -1)")
    plt.yticks([-1, 0, 1])
    plt.grid(True); plt.tight_layout(); plt.show()

plot_sign_flip(alpha, "α")
plot_sign_flip(beta, "β")
plot_sign_flip(gamma, "γ")
