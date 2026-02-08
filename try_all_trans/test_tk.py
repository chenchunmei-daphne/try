import matplotlib.pyplot as plt

from fealpy.backend import bm

bm.set_backend("numpy")
plt.rcParams['font.sans-serif'] = ['SimHei']  # 黑体

sigma = 1.0 
num_samples = 100000
K_db = [-40, 15]

# 生成瑞利分布样本
W1 = bm.random.normal(0, 1, num_samples)
W2 = bm.random.normal(0, 1, num_samples)
X_complex = W1 + 1j * W2
rayleigh_samples = sigma * bm.abs(X_complex)

# 生成莱斯分布样本
rician_samples = []
for K_dB in K_db:
    K = 10**(K_dB/10)
    A = bm.sqrt(2*K/(K+1))
    real_part = A + bm.random.normal(0, bm.sqrt(1/(K+1)), num_samples)
    imag_part = bm.random.normal(0, bm.sqrt(1/(K+1)), num_samples)
    complex_signal = real_part + 1j*imag_part
    rician_samples.append(bm.abs(complex_signal))

# 绘制直方图
plt.figure(figsize=(10, 6))
plt.hist(rayleigh_samples, bins=100, density=True, alpha=0.6, 
         label='瑞利分布', edgecolor='black', color='blue')

colors = ['green', 'orange', 'red']
for i, (K_dB, samples, color) in enumerate(zip(K_db, rician_samples, colors)):
    plt.hist(samples, bins=100, density=True, alpha=0.5, 
             label=f'莱斯分布 K={K_dB}dB', edgecolor='black', color=color)

plt.xlabel('幅度')
plt.ylabel('概率密度')
plt.title('瑞利分布与莱斯分布直方图对比')
plt.legend()
plt.grid(True, alpha=0.3)
plt.show()