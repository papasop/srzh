import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score, mean_squared_error

# 设置随机种子
np.random.seed(42)

# 1. 设置模拟参数
N = 20  # 零点个数 n = 1,...,20
primes = np.array([2, 3, 5, 7, 11, 13, 17, 19, 23, 29])
log_p = np.log(primes)
K = len(primes)

n_vals = np.arange(1, N + 1)

# 2. 模拟 α_n, β_n, γ_n（以 (-1)^n · 常数 + 噪声）
noise = lambda s: np.random.normal(0, s, N)
alpha = ((-1) ** n_vals) * 0.98 + noise(0.005)
beta  = ((-1) ** n_vals) * 0.34 + noise(0.005)
gamma = ((-1) ** n_vals) * 0.23 + noise(0.005)

# 3. 构造余弦拟合矩阵
def build_cos_matrix(n_vals, log_p):
    return np.column_stack([np.cos(n_vals * w) for w in log_p] + [np.sin(n_vals * w) for w in log_p])

X = build_cos_matrix(n_vals, log_p)

# 4. 拟合函数
def fit_cos_model(y, name='param'):
    model = LinearRegression().fit(X, y)
    y_pred = model.predict(X)
    rmse = np.sqrt(mean_squared_error(y, y_pred))
    r2 = r2_score(y, y_pred)
    print(f'✅ {name} 拟合完成: RMSE={rmse:.6f}, R²={r2:.5f}')
    return model.coef_[:K], model.coef_[K:], rmse, r2, y_pred

A_alpha, B_alpha, rmse_a, r2_a, pred_a = fit_cos_model(alpha, 'α')
A_beta,  B_beta,  rmse_b, r2_b, pred_b = fit_cos_model(beta, 'β')
A_gamma, B_gamma, rmse_g, r2_g, pred_g = fit_cos_model(gamma, 'γ')

# 5. 可视化
plt.figure(figsize=(12, 6))
plt.plot(n_vals, alpha, 'o-', label='α_true')
plt.plot(n_vals, pred_a, '--', label='α_fit')
plt.plot(n_vals, beta, 'o-', label='β_true')
plt.plot(n_vals, pred_b, '--', label='β_fit')
plt.plot(n_vals, gamma, 'o-', label='γ_true')
plt.plot(n_vals, pred_g, '--', label='γ_fit')
plt.legend()
plt.title('Prime-log Cosine Model Fit')
plt.xlabel('n'); plt.ylabel('Value')
plt.grid(True)
plt.show()

# 6. 输出振幅和相位
def amp_phase(A, B):
    amps = np.sqrt(A**2 + B**2)
    phases = np.arctan2(-B, A)  # 注意方向
    return amps, phases

amps_alpha, theta_alpha = amp_phase(A_alpha, B_alpha)
amps_beta,  theta_beta  = amp_phase(A_beta,  B_beta)
amps_gamma, theta_gamma = amp_phase(A_gamma, B_gamma)

print("\n🔍 α 振幅 A_k:", np.round(amps_alpha, 4))
print("    相位 θ_k:", np.round(theta_alpha, 4))

print("\n🔍 β 振幅 A_k:", np.round(amps_beta, 4))
print("    相位 θ_k:", np.round(theta_beta, 4))

print("\n🔍 γ 振幅 A_k:", np.round(amps_gamma, 4))
print("    相位 θ_k:", np.round(theta_gamma, 4))
