import matplotlib.pyplot as plt

from fealpy.backend import bm

bm.set_backend("numpy")
plt.rcParams['font.sans-serif'] = ['SimHei']  # 黑体

sigma = 1.0 
num_samples = 100000
K_db = [0, -40, 15] 

# 1. 生成复高斯随机变量 X_complex = W1 + j*W2
W1 = bm.random.normal(0, 1, num_samples)  # 实部
W2 = bm.random.normal(0, 1, num_samples)  # 虚部
X_complex = W1 + 1j * W2

# 2. 计算幅度 X = sqrt(W1^2 + W2^2)
X_magnitude = sigma * (bm.abs(X_complex))


# 3. 绘制幅度X的直方图，并与理论的瑞利PDF比较
x_range = bm.linspace(0, 5, 1000)
# 理论瑞利分布PDF： f(x) = (x/sigma^2) * exp(-x^2/(2*sigma^2))
rayleigh_pdf = (x_range / sigma**2) * bm.exp(-x_range**2 / (2 * sigma**2))

print(type(X_magnitude), type(x_range), type(rayleigh_pdf))
plt.figure(figsize=(10, 6))
# 绘制样本直方图（归一化为密度）
plt.hist(X_magnitude, bins=100, density=True, alpha=0.6, 
         label='样本直方图 (幅度 X)', edgecolor='black')
# 绘制理论曲线
plt.plot(x_range, rayleigh_pdf, 'r-', linewidth=2, label='理论瑞利PDF')
plt.xlabel('幅度 x (实数, x ≥ 0)', fontsize=12)
plt.ylabel('概率密度 f(x)', fontsize=12)
plt.title('复高斯变量的幅度服从瑞利分布', fontsize=14)
plt.grid(True, alpha=0.3)
plt.legend()
plt.show()
print("Done!")