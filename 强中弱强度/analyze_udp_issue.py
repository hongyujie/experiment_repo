import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy import stats

plt.rcParams['font.sans-serif'] = ['SimHei']
plt.rcParams['axes.unicode_minus'] = False

def load_and_analyze(filepath, name):
    """加载并分析数据"""
    df = pd.read_csv(filepath)
    if 'F' in df.columns:
        data = df['F'].values
    else:
        data = df['电流(mA)'].values

    print(f"\n{'='*60}")
    print(f"{name} 数据分析")
    print(f"{'='*60}")
    print(f"数据点数: {len(data)}")
    print(f"均值: {np.mean(data):.4f} mA")
    print(f"标准差: {np.std(data):.4f} mA")
    print(f"最小值: {np.min(data):.4f} mA")
    print(f"最大值: {np.max(data):.4f} mA")
    print(f"峰峰值: {np.max(data) - np.min(data):.4f} mA")

    # 差分特征
    diff = np.diff(data)
    print(f"\n差分统计:")
    print(f"  差分均值: {np.mean(diff):.4f}")
    print(f"  差分标准差: {np.std(diff):.4f}")
    print(f"  差分绝对值均值: {np.mean(np.abs(diff)):.4f}")

    # 频域特征
    fft = np.fft.fft(data)
    fft_magnitude = np.abs(fft)
    freqs = np.fft.fftfreq(len(data))
    pos_mask = freqs > 0
    pos_freqs = freqs[pos_mask]
    pos_magnitude = fft_magnitude[pos_mask]

    print(f"\n频域特征:")
    print(f"  主频位置: {pos_freqs[np.argmax(pos_magnitude)]:.6f}")
    print(f"  主频幅度: {np.max(pos_magnitude):.2f}")

    # 与正常数据的相似度（欧氏距离）
    return data

# 加载所有数据
normal = load_and_analyze('data_process/normal_40Hz.csv', '正常流量')
tcp = load_and_analyze('data_process/tcp_high_40Hz.csv', 'TCP攻击')
udp = load_and_analyze('data_process/udp_high_40Hz.csv', 'UDP攻击')
icmp = load_and_analyze('data_process/icmp_high_40Hz.csv', 'ICMP攻击')

print(f"\n{'='*60}")
print("与正常数据的差异对比（前1000个样本）")
print(f"{'='*60}")

n_samples = 1000
normal_sample = normal[:n_samples]

def calc_distance(data, name):
    sample = data[:n_samples]
    # 欧氏距离
    euclid = np.linalg.norm(normal_sample - sample)
    # 相关系数
    corr = np.corrcoef(normal_sample, sample)[0, 1]
    # 均值差异
    mean_diff = abs(np.mean(normal_sample) - np.mean(sample))
    # 标准差差异
    std_diff = abs(np.std(normal_sample) - np.std(sample))

    print(f"\n{name}:")
    print(f"  欧氏距离: {euclid:.2f} (越小越相似)")
    print(f"  相关系数: {corr:.4f} (越接近1越相似)")
    print(f"  均值差异: {mean_diff:.4f} mA")
    print(f"  标准差差异: {std_diff:.4f} mA")

calc_distance(tcp, 'TCP攻击')
calc_distance(udp, 'UDP攻击')
calc_distance(icmp, 'ICMP攻击')

print(f"\n{'='*60}")
print("关键发现分析")
print(f"{'='*60}")
print("""
从统计数据可以看出：

1. UDP攻击的电流特征与正常流量非常接近：
   - 均值差异很小
   - 标准差与正常流量相近
   - 峰峰值也接近正常流量

2. 相比之下：
   - TCP攻击有明显的电流波动特征
   - ICMP攻击也有较明显的特征
   - UDP攻击的电流变化最为"隐蔽"

3. 这就是为什么模型难以区分UDP攻击的原因：
   - UDP攻击在电流特征上与正常流量过于相似
   - 缺乏明显的区分性特征
   - 模型学到的特征难以有效识别UDP
""")
