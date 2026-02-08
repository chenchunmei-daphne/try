import math
from fealpy.backend import bm
bm.set_backend('pytorch')

def pl_free(dist, fc=6.7e9, gt=1, gr=1, light_speed=3e8):
    """  
    Calculate free-space path loss in dB.
    pl_free = 20*log10(4*pi*fc*dist/light_speed) - 10*log10(gt*gr).

    Parameters:
        dist(tensor): distance between transmitter and receiver in meters.

        fc(float): carrier frequency in Hz.

        gt(float): transmitter gain (linear, not dB).

        gr(float): receiver gain (linear, not dB).

        light_speed(float): speed of light in m/s.

    Returns:
        tensor: free-space path loss in dB
    """
    wavelength = light_speed / fc
    if  isinstance(dist, float) or isinstance(dist, int):
        dist = bm.tensor(dist, dtype=bm.float32)

    path_loss = 20 * bm.log10(4 * bm.pi * dist / wavelength)
    antenna_gain = 10 * math.log10(gt * gr)
    pl = path_loss - antenna_gain
    
    return pl


def pl_logdist_or_norm(dist, dist0, n=2, sigma=3, fc=6.7e9, gt=1, gr=1, light_speed=3e8):
    """
    Calculate path loss using log-distance or log-normal shadowing model.
    
    Log-distance model:
        PL_LOD(d)[dB] = PL_F(d0) + 10n*log10(d/d0)                        (Eq. 1.4)
    
    Log-normal shadowing model:
        PL(d)[dB] = PL_F(d0) + 10n*log10(d/d0) + X_sigma                 (Eq. 1.5)
        where X_sigma ~ N(0, sigma^2)
    
    Parameters:
        dist(tensor): distance between transmitter and receiver in meters.

        dist0(float): reference distance in meters.

        n(float): path loss exponent.

        sigma(float): standard deviation of shadowing in dB (default 3 dB).

        fc(float): carrier frequency in Hz.

        gt(float): transmitter gain (linear, not dB).

        gr(float): receiver gain (linear, not dB).

        light_speed(float): speed of light in m/s.
        
    Returns:
        tensor: path loss in dB (with shadowing if sigma >= 0)
    """

    pl_d0 = pl_free(dist0, fc, gt, gr, light_speed)
    if n > 0:
        deterministic_pl = pl_d0 + 10 * n * bm.log10(dist / dist0)
    else:
        deterministic_pl = pl_d0
    
    if sigma > 0:
        if hasattr(bm, 'randn'):
            shadowing = sigma * bm.randn(dist.shape)
        else:
            shadowing = sigma * bm.random.randn(*dist.shape)
        total_pl = deterministic_pl + shadowing
        return total_pl
    
    elif sigma == 0:
        return deterministic_pl
    else:
        raise ValueError(f'sigma must be non-negative, or 0 to disable shadowing, but got {sigma}.')
# ===================== 运行示例 =====================

import matplotlib.pyplot as plt
import numpy as np


print("=" * 60)
print("运行示例 1: pl_free 函数")
print("=" * 60)

# 示例 1: pl_free 函数
# 创建距离数组 (1m 到 1000m)
dist_tensor = bm.linspace(1, 1000, 100)

# 计算自由空间路径损耗 (默认参数: fc=6.7GHz)
pl_free_result = pl_free(dist_tensor)
print(f"距离范围: 1m 到 1000m")
print(f"在 100m 处的路径损耗: {pl_free_result[10]:.2f} dB")
print(f"在 500m 处的路径损耗: {pl_free_result[49]:.2f} dB")
print(f"在 1000m 处的路径损耗: {pl_free_result[99]:.2f} dB")

print("\n" + "=" * 60)
print("运行示例 2: pl_logdist_or_norm 函数 - 对数距离模型")
print("=" * 60)

# 示例 2: 对数距离模型 (sigma=0)
# 使用不同路径损耗指数
pl_n2 = pl_logdist_or_norm(dist_tensor, dist0=100, n=2, sigma=0)  # 自由空间
pl_n3 = pl_logdist_or_norm(dist_tensor, dist0=100, n=3, sigma=0)  # 城市环境
pl_n4 = pl_logdist_or_norm(dist_tensor, dist0=100, n=4, sigma=0)  # 障碍物较多

print(f"路径损耗指数 n=2 (自由空间):")
print(f"  100m处: {pl_n2[10]:.2f} dB, 1000m处: {pl_n2[99]:.2f} dB")

print(f"路径损耗指数 n=3 (城市环境):")
print(f"  100m处: {pl_n3[10]:.2f} dB, 1000m处: {pl_n3[99]:.2f} dB")

print(f"路径损耗指数 n=4 (障碍物较多):")
print(f"  100m处: {pl_n4[10]:.2f} dB, 1000m处: {pl_n4[99]:.2f} dB")

print("\n" + "=" * 60)
print("运行示例 3: pl_logdist_or_norm 函数 - 对数正态阴影模型")
print("=" * 60)

# 示例 3: 对数正态阴影模型 (sigma>0)
# 计算10次模拟，展示随机性
dist_point = bm.tensor([100.0])  # 100m处的点
for i in range(5):
    pl_shadow = pl_logdist_or_norm(dist_point, dist0=1, n=3, sigma=3)
    print(f"模拟 {i+1}: 100m处路径损耗 = {pl_shadow[0]:.2f} dB")

# ===================== 绘图展示 =====================

plt.figure(figsize=(15, 10))

# 子图1: 自由空间路径损耗
plt.subplot(2, 2, 1)
plt.plot(dist_tensor.numpy(), pl_free_result.numpy(), 'b-', linewidth=2)
plt.xlabel('距离 (m)', fontsize=12)
plt.ylabel('路径损耗 (dB)', fontsize=12)
plt.title('自由空间路径损耗 (fc=6.7GHz, Gt=Gr=1)', fontsize=14)
plt.grid(True, alpha=0.3)
plt.xscale('log')
plt.xlim(1, 1000)

# 子图2: 不同路径损耗指数的比较
plt.subplot(2, 2, 2)
plt.plot(dist_tensor.numpy(), pl_n2.numpy(), 'g-', label='n=2 (自由空间)', linewidth=2)
plt.plot(dist_tensor.numpy(), pl_n3.numpy(), 'b-', label='n=3 (城市)', linewidth=2)
plt.plot(dist_tensor.numpy(), pl_n4.numpy(), 'r-', label='n=4 (障碍物多)', linewidth=2)
plt.xlabel('距离 (m)', fontsize=12)
plt.ylabel('路径损耗 (dB)', fontsize=12)
plt.title('不同路径损耗指数的对数距离模型', fontsize=14)
plt.legend()
plt.grid(True, alpha=0.3)
plt.xscale('log')
plt.xlim(1, 1000)

# 子图3: 对数正态阴影模型模拟 (单次)
plt.subplot(2, 2, 3)
pl_shadow_single = pl_logdist_or_norm(dist_tensor, dist0=1, n=3, sigma=3)
plt.plot(dist_tensor.numpy(), pl_shadow_single.numpy(), 'purple', linewidth=2, alpha=0.7)
plt.xlabel('距离 (m)', fontsize=12)
plt.ylabel('路径损耗 (dB)', fontsize=12)
plt.title('对数正态阴影模型 (n=3, σ=3dB, 单次模拟)', fontsize=14)
plt.grid(True, alpha=0.3)
plt.xscale('log')
plt.xlim(1, 1000)

# 子图4: 对数正态阴影模型多次模拟对比
plt.subplot(2, 2, 4)
# 固定距离点：100m
dist_fixed = bm.tensor([100.0])

# 进行50次模拟
n_simulations = 50
pl_values = []
for i in range(n_simulations):
    pl_val = pl_logdist_or_norm(dist_fixed, dist0=1, n=3, sigma=3)
    pl_values.append(pl_val.item())

# 绘制柱状图
plt.hist(pl_values, bins=15, edgecolor='black', alpha=0.7)
plt.axvline(np.mean(pl_values), color='red', linestyle='--', linewidth=2, 
           label=f'均值: {np.mean(pl_values):.2f} dB')
plt.xlabel('路径损耗 (dB)', fontsize=12)
plt.ylabel('频次', fontsize=12)
plt.title(f'100m处路径损耗分布 (n=3, σ=3dB, {n_simulations}次模拟)', fontsize=14)
plt.legend()
plt.grid(True, alpha=0.3)

plt.tight_layout()
plt.suptitle('无线信道路径损耗模型对比', fontsize=16, y=1.02)
plt.show()

# ===================== 额外分析 =====================
print("\n" + "=" * 60)
print("统计分析")
print("=" * 60)

# 计算阴影模型的统计特性
dist_analysis = bm.tensor([100.0])
n_sim = 1000
pl_samples = []
for _ in range(n_sim):
    pl = pl_logdist_or_norm(dist_analysis, dist0=1, n=3, sigma=3)
    pl_samples.append(pl.item())

pl_samples = np.array(pl_samples)
print(f"对数正态阴影模型 (n=3, σ=3dB) 在 100m 处的统计:")
print(f"  样本数: {n_sim}")
print(f"  均值: {np.mean(pl_samples):.2f} dB")
print(f"  标准差: {np.std(pl_samples):.2f} dB")
print(f"  最小值: {np.min(pl_samples):.2f} dB")
print(f"  最大值: {np.max(pl_samples):.2f} dB")
print(f"  95%置信区间: [{np.percentile(pl_samples, 2.5):.2f}, {np.percentile(pl_samples, 97.5):.2f}] dB")

# 验证确定性模型 vs 随机模型
print(f"\n确定性模型在 100m 处: {pl_logdist_or_norm(dist_analysis, dist0=1, n=3, sigma=0).item():.2f} dB")
print(f"随机模型均值与确定性模型差值: {np.mean(pl_samples) - pl_logdist_or_norm(dist_analysis, dist0=1, n=3, sigma=0).item():.2f} dB")