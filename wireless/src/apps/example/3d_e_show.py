import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation

# 参数
c = 3e8
f = 1e9        # 1 GHz
lam = c / f
omega = 2 * np.pi * f
k = 2 * np.pi / lam
L = 3          # 多径数

# 空间网格
x = np.linspace(0, 2*lam, 50)
y = np.linspace(0, 2*lam, 50)
X, Y = np.meshgrid(x, y)

# 多径参数（振幅、方向、初相位）
E0 = [1.0, 0.8, 0.6]
kx = [1, -1, 1]
ky = [1, 1, -1]
phi = [0, np.pi/4, np.pi/2]

# 时间设置（慢速动画）
t_max = 5e-8    # 总时间覆盖 5 个纳秒，慢一些
frames = 10000    # 帧数更多，更平滑
dt = t_max / frames

# 创建图形
fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')
ax.set_zlim(-3, 3)
ax.set_xlabel('x (m)')
ax.set_ylabel('y (m)')
ax.set_zlabel('E_z')
ax.set_title("3D Multi-path E-field Animation")

# 更新函数
def update(frame):
    t = frame * dt
    E = np.zeros_like(X)
    for ell in range(L):
        E += E0[ell] * np.cos(omega*t - k*(kx[ell]*X + ky[ell]*Y) - phi[ell])
    ax.clear()
    ax.set_zlim(-3, 3)
    ax.set_xlabel('x (m)')
    ax.set_ylabel('y (m)')
    ax.set_zlabel('E_z')
    ax.set_title("3D Multi-path E-field Animation")
    ax.plot_surface(X, Y, E, cmap='viridis')
    return []

# 动画，interval 调大 → 播放慢
ani = FuncAnimation(fig, update, frames=frames, interval=200)  # 每帧 200ms
plt.show()
