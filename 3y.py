import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score

# 示例 γ 序列（你应替换为自己的数据）
gamma = np.array([
    0.0045, -0.0042, 0.0091, -0.0103, 0.0208, -0.0236, 0.0474, -0.0539, 0.1082,
    -0.1223, 0.2459, -0.2784, 0.5592, -0.6322, 1.2698, -1.4361, 2.8827, -3.2579,
    6.5549, -7.4037
])

# 构建三项递推数据集
X = np.column_stack((gamma[1:-1], gamma[:-2]))  # γₙ, γₙ₋₁
y = gamma[2:]                                   # γₙ₊₁

# 拟合线性模型
model = LinearRegression()
model.fit(X, y)
y_pred = model.predict(X)

# 拟合参数
a, b = model.coef_
c = model.intercept_

# 拟合质量指标
rmse = np.sqrt(mean_squared_error(y, y_pred))
r2 = r2_score(y, y_pred)

# 输出拟合公式
print(f"三项递推: γₙ₊₁ = {a:.4f}·γₙ + {b:.4f}·γₙ₋₁ + {c:.4f}")
print(f"RMSE: {rmse:.6f}")
print(f"R²: {r2:.5f}")

# 生成 DataFrame 详细输出
df = pd.DataFrame({
    "γₙ₋₁": gamma[:-2],
    "γₙ": gamma[1:-1],
    "真实 γₙ₊₁": gamma[2:],
    "预测 γₙ₊₁": y_pred,
    "残差": gamma[2:] - y_pred
})

# 展示前几行
from IPython.display import display
display(df)

# 如需导出 CSV：
# df.to_csv("gamma_prediction_results.csv", index=False)

# 可视化真实 vs 预测
plt.figure(figsize=(10,4))
plt.plot(gamma[2:], label="真实 γₙ₊₁", marker='o')
plt.plot(y_pred, label="预测 γₙ₊₁", linestyle='--', marker='x')
plt.title("γₙ₊₁ 三项递推拟合结果")
plt.xlabel("n")
plt.ylabel("γ 值")
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.show()

# 残差图
plt.figure(figsize=(8,3))
plt.stem(np.arange(len(y_pred)), gamma[2:] - y_pred, use_line_collection=True)
plt.title("残差分布 γₙ₊₁ - 预测值")
plt.xlabel("n")
plt.ylabel("残差")
plt.grid(True)
plt.tight_layout()
plt.show()
