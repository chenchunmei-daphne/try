import numpy as np
import matplotlib.pyplot as plt
from scipy import special
plt.rcParams['font.sans-serif'] = ['SimHei']  # 黑体
# 参数
K_dB, sigma2, N = 3, 0.5, 50000

# 生成莱斯样本
K = 10**(K_dB/10)
A = np.sqrt(2*sigma2*K)
samples = np.abs(A + np.random.normal(0, np.sqrt(sigma2), N) + 
                 1j*np.random.normal(0, np.sqrt(sigma2), N))

# 理论PDF
r = np.linspace(0, np.max(samples)+0.1, 5000)
pdf = (r/sigma2)*np.exp(-(r**2 + A**2)/(2*sigma2))*special.i0(A*r/sigma2)

# 绘图
plt.figure(figsize=(9,5))
plt.hist(samples, bins=60, density=True, alpha=0.6, label='仿真')
plt.plot(r, pdf, 'r-', lw=2.5, label='理论')
plt.xlabel('幅度'); plt.ylabel('概率密度')
plt.title(f'莱斯分布验证 (K={K_dB}dB)')
plt.legend(); plt.grid(True, alpha=0.3)
plt.show()