import numpy as np
import matplotlib.pyplot as plt

def generate_simple_MIMO_channel(
    n_rx=4,         # 接收天线数 (默认4)
    n_tx=4,         # 发射天线数 (默认4)  
    n_path=2,       # 路径数 (默认2，2-径模型)
    n_snap=1000     # 快照数 (默认1000)
):
    """
    最简单的MIMO信道生成程序（不考虑空间相关性）
    参数命名遵循QuaDriGa/3GPP标准
    
    参数:
    n_rx: 接收天线数
    n_tx: 发射天线数  
    n_path: 路径数
    n_snap: 快照数（时间采样数）
    
    返回:
    H_time: 时域信道张量 [n_rx, n_tx, n_tap, n_snap]
    pdp: 每条路径的功率 (Power Delay Profile)
    tap_indices: 每条路径对应的抽头索引
    """
    
    # 系统参数
    Ts = 10e-9  # 采样间隔10ns
    # 时延设置：2-径模型，时延为0和60ns
    delays = np.array([0, 60e-9])  # 单位：秒
    # 功率设置：两条路径功率相等
    pdp = np.array([0.5, 0.5])  # 每条路径的平均功率
    
    # 计算抽头索引（离散化时延）
    tap_indices = np.round(delays / Ts).astype(int)
    n_tap = np.max(tap_indices) + 1  # 最大抽头数
    
    # 初始化时域信道张量
    # 维度：[接收天线，发射天线，时延抽头，快照]
    H_time = np.zeros((n_rx, n_tx, n_tap, n_snap), dtype=complex)
    
    print("生成独立MIMO信道...")
    print(f"系统配置: {n_rx}×{n_tx} MIMO，{n_path}条路径，{n_snap}个快照")
    print(f"时延抽头: {tap_indices}，对应功率: {pdp}")
    
    # 生成信道实现（最简单的独立生成）
    for snap in range(n_snap):
        for path in range(n_path):
            # 生成该路径的独立复高斯系数
            # 实部和虚部都是独立标准正态分布
            H_iid = (np.random.randn(n_rx, n_tx) + 
                     1j * np.random.randn(n_rx, n_tx)) / np.sqrt(2)
            
            # 应用路径功率
            H_path = np.sqrt(pdp[path]) * H_iid
            
            # 放置到对应时延抽头
            tap = tap_indices[path]
            H_time[:, :, tap, snap] += H_path
    
    return H_time, pdp, tap_indices


def verify_simple_statistics(H_time, pdp, tap_indices):
    """
    验证简化信道的统计特性
    
    参数:
    H_time: 时域信道张量 [n_rx, n_tx, n_tap, n_snap]
    pdp: 每条路径的功率
    tap_indices: 每条路径对应的抽头索引
    """
    # 1. 验证平均功率
    # 沿接收天线、发射天线、快照维度求平均
    avg_power = np.mean(np.abs(H_time)**2, axis=(0, 1, 3))
    
    print("\n" + "="*60)
    print("统计验证结果:")
    print("="*60)
    
    for p in range(len(pdp)):
        tap = tap_indices[p]
        theoretical = pdp[p]  # 理论功率
        simulated = avg_power[tap]  # 仿真功率
        error = abs(theoretical - simulated) / theoretical * 100
        
        print(f"路径{p+1} (抽头{tap}):")
        print(f"  理论功率: {theoretical:.4f}")
        print(f"  仿真功率: {simulated:.4f}")
        print(f"  相对误差: {error:.2f}%")
    
    # 2. 验证瑞利分布
    print("\n" + "-"*60)
    print("瑞利分布验证:")
    print("-"*60)
    
    # 展平所有系数
    h_all = H_time.flatten()
    
    # 计算幅度
    magnitudes = np.abs(h_all)
    
    # 理论瑞利分布参数
    # 对于复高斯 h = x + jy, x,y ~ N(0, σ²/2)
    # 则 |h| 服从瑞利分布，参数σ满足 E[|h|²] = 2σ²
    avg_power_all = np.mean(magnitudes**2)
    sigma_theory = np.sqrt(avg_power_all / 2)
    
    # 计算统计量
    mean_mag = np.mean(magnitudes)
    var_mag = np.var(magnitudes)
    
    # 理论值
    # 瑞利分布的均值 = σ√(π/2)，方差 = (2-π/2)σ²
    mean_theory = sigma_theory * np.sqrt(np.pi/2)
    var_theory = (2 - np.pi/2) * sigma_theory**2
    
    print(f"幅度均值: {mean_mag:.4f} (理论: {mean_theory:.4f})")
    print(f"幅度方差: {var_mag:.4f} (理论: {var_theory:.4f})")
    
    # 3. 验证相位均匀分布
    print("\n" + "-"*60)
    print("相位分布验证:")
    print("-"*60)
    
    phases = np.angle(h_all)
    
    # 检查相位是否均匀分布在[-π, π]
    hist, bin_edges = np.histogram(phases, bins=20, range=(-np.pi, np.pi))
    uniformity = np.std(hist) / np.mean(hist)  # 均匀性指标
    
    print(f"相位分布标准差/均值比: {uniformity:.4f}")
    print("(越接近0表示分布越均匀)")
    
    return {
        'avg_power': avg_power,
        'magnitudes': magnitudes,
        'phases': phases,
        'sigma_theory': sigma_theory
    }


def plot_channel_properties(H_time, pdp, tap_indices, stats):
    """
    绘制信道特性图
    
    参数:
    H_time: 时域信道张量 [n_rx, n_tx, n_tap, n_snap]
    pdp: 每条路径的功率
    tap_indices: 每条路径对应的抽头索引
    stats: 统计验证结果
    """
    # 获取维度信息
    n_rx, n_tx, n_tap, n_snap = H_time.shape
    
    fig, axes = plt.subplots(2, 3, figsize=(15, 10))
    fig.suptitle(f'独立MIMO信道特性分析 ({n_rx}×{n_tx} MIMO, {len(pdp)}条路径, {n_snap}个快照)', 
                 fontsize=16, fontweight='bold')
    
    # 子图1: PDP对比
    axes[0, 0].stem(tap_indices, pdp, markerfmt='bo', 
                   basefmt=' ', label='理论PDP')
    axes[0, 0].stem(tap_indices, stats['avg_power'][tap_indices], 
                   markerfmt='rx', basefmt=' ', 
                   label='仿真PDP')
    axes[0, 0].set_xlabel('时延抽头索引')
    axes[0, 0].set_ylabel('功率')
    axes[0, 0].set_title('功率时延分布(PDP)对比')
    axes[0, 0].legend()
    axes[0, 0].grid(True, alpha=0.3)
    
    # 子图2: 幅度分布直方图
    magnitudes = stats['magnitudes']
    axes[0, 1].hist(magnitudes, bins=50, density=True, alpha=0.7, 
                   color='blue', edgecolor='black')
    
    # 绘制理论瑞利分布曲线
    sigma = stats['sigma_theory']
    x = np.linspace(0, np.max(magnitudes), 1000)
    pdf_theory = (x / sigma**2) * np.exp(-x**2 / (2 * sigma**2))
    axes[0, 1].plot(x, pdf_theory, label='理论瑞利分布')
    
    axes[0, 1].set_xlabel('幅度')
    axes[0, 1].set_ylabel('概率密度')
    axes[0, 1].set_title('幅度分布（瑞利分布验证）')
    axes[0, 1].legend()
    axes[0, 1].grid(True, alpha=0.3)
    
    # 子图3: 相位分布直方图
    axes[0, 2].hist(stats['phases'], bins=50, density=True, alpha=0.7, 
                   color='green', edgecolor='black', range=(-np.pi, np.pi))
    
    # 理论均匀分布
    x_phase = np.linspace(-np.pi, np.pi, 1000)
    uniform_pdf = np.ones_like(x_phase) / (2 * np.pi)
    axes[0, 2].plot(x_phase, uniform_pdf, label='理论均匀分布')
    
    axes[0, 2].set_xlabel('相位（弧度）')
    axes[0, 2].set_ylabel('概率密度')
    axes[0, 2].set_title('相位分布（均匀分布验证）')
    axes[0, 2].legend()
    axes[0, 2].grid(True, alpha=0.3)
    
    # 子图4: 信道系数实部分布
    h_real = np.real(H_time.flatten())
    axes[1, 0].hist(h_real, bins=50, density=True, alpha=0.7, 
                   color='purple', edgecolor='black')
    
    # 理论正态分布
    mu, std = np.mean(h_real), np.std(h_real)
    x_real = np.linspace(np.min(h_real), np.max(h_real), 1000)
    pdf_normal = 1/(std*np.sqrt(2*np.pi)) * np.exp(-0.5*((x_real-mu)/std)**2)
    axes[1, 0].plot(x_real, pdf_normal, label='理论正态分布')
    
    axes[1, 0].set_xlabel('实部值')
    axes[1, 0].set_ylabel('概率密度')
    axes[1, 0].set_title('信道系数实部分布（应为正态分布）')
    axes[1, 0].legend()
    axes[1, 0].grid(True, alpha=0.3)
    
    # 子图5: 信道系数虚部分布
    h_imag = np.imag(H_time.flatten())
    axes[1, 1].hist(h_imag, bins=50, density=True, alpha=0.7, 
                   color='orange', edgecolor='black')
    
    # 理论正态分布
    mu, std = np.mean(h_imag), np.std(h_imag)
    x_imag = np.linspace(np.min(h_imag), np.max(h_imag), 1000)
    pdf_normal = 1/(std*np.sqrt(2*np.pi)) * np.exp(-0.5*((x_imag-mu)/std)**2)
    axes[1, 1].plot(x_imag, pdf_normal, label='理论正态分布')
    
    axes[1, 1].set_xlabel('虚部值')
    axes[1, 1].set_ylabel('概率密度')
    axes[1, 1].set_title('信道系数虚部分布（应为正态分布）')
    axes[1, 1].legend()
    axes[1, 1].grid(True, alpha=0.3)
    
    # 子图6: IQ平面散点图
    sample_size = min(5000, len(h_real))
    idx = np.random.choice(len(h_real), sample_size, replace=False)
    axes[1, 2].scatter(h_real[idx], h_imag[idx], alpha=0.5, s=1, c='blue')
    
    # 添加理论圆
    theta = np.linspace(0, 2*np.pi, 1000)
    r = sigma * np.sqrt(-2 * np.log(0.1))  # 包含90%数据的圆半径
    axes[1, 2].plot(r * np.cos(theta), r * np.sin(theta), label=f'理论边界(σ={sigma:.2f})')
    
    axes[1, 2].set_xlabel('实部')
    axes[1, 2].set_ylabel('虚部')
    axes[1, 2].set_title('IQ平面散点图')
    axes[1, 2].set_aspect('equal', 'box')
    axes[1, 2].legend()
    axes[1, 2].grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.show()


def analyze_correlation(H_time):
    """
    分析信道系数的相关性（验证独立性假设）
    
    参数:
    H_time: 时域信道张量 [n_rx, n_tx, n_tap, n_snap]
    """
    print("\n" + "="*60)
    print("相关性分析（验证独立性）:")
    print("="*60)
    
    # 获取维度信息
    n_rx, n_tx, n_tap, n_snap = H_time.shape
    
    # 1. 同一抽头不同天线对之间的相关性
    print("\n1. 同一抽头不同天线对之间的相关性:")
    
    # 选取第一个抽头
    tap = 0
    h1 = H_time[0, 0, tap, :]  # 天线对(1,1)
    h2 = H_time[0, 1, tap, :]  # 天线对(1,2)
    h3 = H_time[1, 0, tap, :]  # 天线对(2,1)
    
    corr_12 = np.abs(np.corrcoef(h1, h2)[0, 1])
    corr_13 = np.abs(np.corrcoef(h1, h3)[0, 1])
    
    print(f"  天线对(1,1)与(1,2)的相关系数: {corr_12:.6f}")
    print(f"  天线对(1,1)与(2,1)的相关系数: {corr_13:.6f}")
    print(f"  (理论上应为0，实际接近0)")
    
    # 2. 不同抽头之间的相关性
    print("\n2. 不同抽头之间的相关性:")
    
    if n_tap > 1:
        h_tap0 = H_time[0, 0, 0, :]  # 抽头0
        h_tap1 = H_time[0, 0, 1, :]  # 抽头1
        corr_tap = np.abs(np.corrcoef(h_tap0, h_tap1)[0, 1])
        print(f"  抽头0与抽头1的相关系数: {corr_tap:.6f}")
        print(f"  (理论上应为0，实际接近0)")
    
    # 3. 同一天线对不同快照之间的相关性（应接近0）
    print("\n3. 不同快照之间的相关性（时间/空间独立性）:")
    
    if n_snap > 2:
        h_snap1 = H_time[0, 0, 0, 0:n_snap-1]
        h_snap2 = H_time[0, 0, 0, 1:n_snap]
        corr_snap = np.abs(np.corrcoef(h_snap1, h_snap2)[0, 1])
        print(f"  相邻快照之间的相关系数: {corr_snap:.6f}")
        print(f"  (理论上应为0，实际接近0)")


def analyze_channel_structure(H_time, pdp, tap_indices):
    """
    分析信道矩阵结构
    
    参数:
    H_time: 时域信道张量 [n_rx, n_tx, n_tap, n_snap]
    pdp: 每条路径的功率
    tap_indices: 每条路径对应的抽头索引
    """
    print("\n" + "="*60)
    print("信道矩阵结构分析:")
    print("="*60)
    
    n_rx, n_tx, n_tap, n_snap = H_time.shape
    print(f"信道张量维度: {n_rx}×{n_tx}×{n_tap}×{n_snap}")
    print(f"总信道系数数量: {n_rx * n_tx * n_tap * n_snap:,}")
    
    # 显示第一个快照的信道矩阵
    print(f"\n第一个快照的信道矩阵（维度: {n_rx}×{n_tx}×{n_tap}）:")
    
    for tap in range(n_tap):
        if np.any(H_time[:, :, tap, 0] != 0):  # 只显示非零抽头
            print(f"\n抽头 {tap}:")
            # 显示前2×2的天线对，避免输出过多
            if n_rx >= 2 and n_tx >= 2:
                print("前2×2天线对:")
                print(H_time[0:2, 0:2, tap, 0])
            else:
                print(H_time[:, :, tap, 0])
    
    # 频域转换示例（OFDM系统）
    print("\n" + "-"*60)
    print("频域转换示例（OFDM）:")
    print("-"*60)
    
    n_subcarriers = 64  # 子载波数
    # 沿时延维度（第2轴）做FFT
    H_freq = np.fft.fft(H_time, n_subcarriers, axis=2)
    
    print(f"频域信道维度: {H_freq.shape}")
    print(f"频域信道总系数: {H_freq.size:,}")
    
    # 计算相干带宽
    max_delay = np.max(tap_indices) * 10e-9  # 最大时延（秒）
    coherence_bw = 1 / (5 * max_delay)  # 粗略估计相干带宽
    
    print(f"\n信道时延特性:")
    print(f"最大时延: {max_delay*1e9:.1f} ns")
    print(f"近似相干带宽: {coherence_bw/1e6:.2f} MHz")
    
    # 分析频率选择性
    bandwidth = 20e6  # 系统带宽20MHz
    is_freq_selective = bandwidth > coherence_bw
    print(f"系统带宽: {bandwidth/1e6:.1f} MHz")
    print(f"是否为频率选择性信道: {is_freq_selective}")
    
    # 如果是频率选择性信道，说明需要均衡
    if is_freq_selective:
        print("提示: 信道具有频率选择性，需要频域均衡")
    
    return H_freq


# 主程序
if __name__ == "__main__":
    print("="*60)
    print("独立MIMO信道生成与验证")
    print("="*60)
    
    # 生成独立MIMO信道
    H_time, pdp, tap_indices = generate_simple_MIMO_channel(
        n_rx=2,      # 2个接收天线
        n_tx=2,      # 2个发射天线
        n_path=2,    # 2条路径
        n_snap=5000  # 5000个快照
    )
    
    # 验证统计特性
    stats = verify_simple_statistics(H_time, pdp, tap_indices)
    
    # 绘制特性图
    plot_channel_properties(H_time, pdp, tap_indices, stats)
    
    # 分析相关性
    analyze_correlation(H_time)
    
    # 分析信道结构
    H_freq = analyze_channel_structure(H_time, pdp, tap_indices)
    
    # 额外分析：信道容量初步估计
    print("\n" + "="*60)
    print("信道容量初步估计:")
    print("="*60)
    
    n_rx, n_tx, _, n_snap = H_time.shape
    
    # 计算平均信道容量（不考虑预编码和干扰）
    # 使用公式: C = log2(det(I + (SNR/N_t) * H*H^H))
    SNR_dB = 20  # 信噪比20dB
    SNR_linear = 10**(SNR_dB/10)
    
    capacities = []
    for snap in range(min(100, n_snap)):  # 只计算前100个快照
        # 获取当前快照的信道矩阵（合并所有抽头）
        H_snap = np.sum(H_time[:, :, :, snap], axis=2)
        
        # 计算信道容量
        if n_rx >= n_tx:
            # 当接收天线数 >= 发射天线数时
            H_Hermitian = H_snap @ H_snap.conj().T
            identity = np.eye(n_rx)
            capacity = np.log2(np.linalg.det(
                identity + (SNR_linear/n_tx) * H_Hermitian
            ))
        else:
            # 当接收天线数 < 发射天线数时
            H_Hermitian = H_snap.conj().T @ H_snap
            identity = np.eye(n_tx)
            capacity = np.log2(np.linalg.det(
                identity + (SNR_linear/n_tx) * H_Hermitian
            ))
        
        capacities.append(np.real(capacity))
    
    avg_capacity = np.mean(capacities)
    print(f"在SNR={SNR_dB}dB时，平均信道容量: {avg_capacity:.2f} bps/Hz")
    print(f"容量范围: {np.min(capacities):.2f} ~ {np.max(capacities):.2f} bps/Hz")