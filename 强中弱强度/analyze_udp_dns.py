import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy import stats
import os

plt.rcParams['font.sans-serif'] = ['SimHei']
plt.rcParams['axes.unicode_minus'] = False

def extract_advanced_features(data, name):
    """提取高级特征"""
    data = data.flatten()
    n = len(data)

    features = {}

    features['basic'] = {
        '均值': np.mean(data),
        '标准差': np.std(data),
        '峰峰值': np.max(data) - np.min(data),
        '均方根': np.sqrt(np.mean(data**2)),
    }

    diff = np.diff(data)
    features['diff'] = {
        '差分均值': np.mean(diff),
        '差分标准差': np.std(diff),
        '差分绝对值均值': np.mean(np.abs(diff)),
        '差分最大值': np.max(diff),
        '差分最小值': np.min(diff),
    }

    rolling_diff = np.diff(data)
    features['rolling'] = {
        '变符号次数': np.sum(np.diff(np.sign(rolling_diff)) != 0),
        '过零点个数': np.sum(np.diff(np.sign(data)) != 0),
    }

    fft = np.fft.fft(data)
    fft_magnitude = np.abs(fft)
    freqs = np.fft.fftfreq(n)

    pos_mask = freqs > 0
    pos_freqs = freqs[pos_mask]
    pos_magnitude = fft_magnitude[pos_mask]

    total_power = np.sum(fft_magnitude**2)

    features['fft'] = {
        '直流分量': fft_magnitude[0]**2 / total_power,
        '主频位置': pos_freqs[np.argmax(pos_magnitude)],
        '前5频分量占比': np.sum(pos_magnitude[:5]**2) / np.sum(pos_magnitude**2),
        '前10频分量占比': np.sum(pos_magnitude[:10]**2) / np.sum(pos_magnitude**2),
        '频谱熵': -np.sum((pos_magnitude**2 / (np.sum(pos_magnitude**2) + 1e-10)) * np.log(pos_magnitude**2 / (np.sum(pos_magnitude**2) + 1e-10) + 1e-10)),
        '谱密度最大值': np.max(pos_magnitude**2) / total_power,
    }

    window_size = 10
    local_stds = []
    for i in range(0, n - window_size, window_size):
        local_stds.append(np.std(data[i:i+window_size]))
    features['local'] = {
        '局部标准差均值': np.mean(local_stds),
        '局部标准差标准差': np.std(local_stds),
        '局部标准差最大值': np.max(local_stds),
        '局部标准差最小值': np.min(local_stds),
    }

    zero_crossings = np.sum(np.diff(np.sign(data - np.mean(data))) != 0)
    features['waveform'] = {
        '零交叉率': zero_crossings / n,
        '偏度': stats.skew(data),
        '峰度': stats.kurtosis(data),
    }

    return features

def compare_two_classes(df1, df2, name1, name2):
    """对比两个类别的特征"""
    print(f'\n{"="*60}')
    print(f'{name1} vs {name2} 特征对比')
    print(f'{"="*60}')

    features1 = extract_advanced_features(df1.values, name1)
    features2 = extract_advanced_features(df2.values, name2)

    categories = ['basic', 'diff', 'rolling', 'fft', 'local', 'waveform']

    for cat in categories:
        print(f'\n--- {cat.upper()} 特征 ---')
        f1 = features1[cat]
        f2 = features2[cat]

        print(f'{name1:12} {name2:12} 差异     差异%')
        print('-' * 60)

        for key in f1.keys():
            val1 = f1[key]
            val2 = f2[key]
            diff = abs(val1 - val2)
            diff_pct = diff / (abs(val1) + 1e-8) * 100
            print(f'{val1:12.4f} {val2:12.4f} {diff:12.4f} {diff_pct:8.2f}%')

def plot_signal_comparison(df_dict, names):
    """绘制信号对比图"""
    os.makedirs('img', exist_ok=True)

    fig, axes = plt.subplots(3, 1, figsize=(14, 12))

    colors = ['green', 'blue', 'red', 'orange', 'purple']
    short_names = ['正常', 'TCP', 'UDP', 'ICMP', 'DNS']

    for idx, (name, color) in enumerate(zip(names, colors)):
        ax = axes[0]
        data = df_dict[name].values[:500]
        ax.plot(data, color=color, alpha=0.7, label=name)

    axes[0].set_title('原始信号对比（前500点）')
    axes[0].set_xlabel('样本点')
    axes[0].set_ylabel('电流(mA)')
    axes[0].legend()
    axes[0].grid(True, alpha=0.3)

    for idx, (name, color) in enumerate(zip(names, colors)):
        ax = axes[1]
        data = df_dict[name].values[:500]
        fft = np.fft.fft(data)
        magnitude = np.abs(fft[:len(fft)//2])
        freqs = np.fft.fftfreq(len(data))[:len(data)//2]
        ax.plot(freqs, magnitude, color=color, alpha=0.7, label=name)

    axes[1].set_title('频谱对比（前500点）')
    axes[1].set_xlabel('频率')
    axes[1].set_ylabel('幅度')
    axes[1].legend()
    axes[1].grid(True, alpha=0.3)

    for idx, (name, color) in enumerate(zip(names, colors)):
        ax = axes[2]
        data = df_dict[name].values[:500]
        diff = np.abs(np.diff(data))
        ax.hist(diff, bins=50, color=color, alpha=0.5, label=name)

    axes[2].set_title('差分幅度分布')
    axes[2].set_xlabel('差分绝对值')
    axes[2].set_ylabel('频数')
    axes[2].legend()
    axes[2].grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig('img/udp_dns_comparison.png', dpi=150)
    plt.close()
    print('\n已保存对比图到 img/udp_dns_comparison.png')

def analyze_udp_vs_dns():
    """专门分析UDP和DNS的区别"""
    data_dir = 'data_process'

    print('='*60)
    print('UDP vs DNS 专项分析')
    print('='*60)

    df_udp = pd.read_csv(os.path.join(data_dir, 'udp_high_40Hz.csv'))
    df_dns = pd.read_csv(os.path.join(data_dir, 'dns_high_40Hz.csv'))

    if 'F' in df_udp.columns:
        col = 'F'
    else:
        col = '电流(mA)'

    df_udp = df_udp[col]
    df_dns = df_dns[col]

    compare_two_classes(df_udp, df_dns, 'UDP', 'DNS')

    df_all = {}
    for fname in ['normal_40Hz.csv', 'tcp_high_40Hz.csv', 'udp_high_40Hz.csv',
                  'icmp_high_40Hz.csv', 'dns_high_40Hz.csv']:
        df = pd.read_csv(os.path.join(data_dir, fname))
        if 'F' in df.columns:
            col = 'F'
        else:
            col = '电流(mA)'
        df_all[fname] = df[col]

    plot_signal_comparison(df_all, ['normal_40Hz.csv', 'tcp_high_40Hz.csv', 'udp_high_40Hz.csv',
                                    'icmp_high_40Hz.csv', 'dns_high_40Hz.csv'])

    print('\n' + '='*60)
    print('UDP/DNS vs 其他类别 特征对比')
    print('='*60)

    df_tcp = df_all['tcp_high_40Hz.csv']
    df_icmp = df_all['icmp_high_40Hz.csv']
    df_normal = df_all['normal_40Hz.csv']

    print('\n--- UDP vs TCP ---')
    udp_diff = np.abs(np.diff(df_udp.values))
    tcp_diff = np.abs(np.diff(df_tcp.values))
    print(f'UDP差分均值: {np.mean(udp_diff):.6f}, TCP差分均值: {np.mean(tcp_diff):.6f}')
    print(f'UDP差分标准差: {np.std(udp_diff):.6f}, TCP差分标准差: {np.std(tcp_diff):.6f}')

    print('\n--- DNS vs TCP ---')
    dns_diff = np.abs(np.diff(df_dns.values))
    print(f'DNS差分均值: {np.mean(dns_diff):.6f}, TCP差分均值: {np.mean(tcp_diff):.6f}')
    print(f'DNS差分标准差: {np.std(dns_diff):.6f}, TCP差分标准差: {np.std(tcp_diff):.6f}')

    print('\n--- UDP vs DNS ---')
    print(f'UDP差分均值: {np.mean(udp_diff):.6f}, DNS差分均值: {np.mean(dns_diff):.6f}')
    print(f'UDP差分标准差: {np.std(udp_diff):.6f}, DNS差分标准差: {np.std(dns_diff):.6f}')
    print(f'UDP峰峰值: {np.max(df_udp)-np.min(df_udp):.4f}, DNS峰峰值: {np.max(df_dns)-np.min(df_dns):.4f}')

    print('\n' + '='*60)
    print('关键发现')
    print('='*60)

    print('\n1. UDP和DNS的整体统计特征非常接近:')
    print(f'   - 峰峰值差异: {abs((np.max(df_udp)-np.min(df_udp)) - (np.max(df_dns)-np.min(df_dns))):.4f}')
    print(f'   - 标准差差异: {abs(np.std(df_udp) - np.std(df_dns)):.4f}')

    print('\n2. 时域变化特征:')
    print(f'   - UDP和DNS的差分特征差异很小')
    print(f'   - 这解释了为什么模型难以区分它们')

    print('\n3. 为什么UDP被误判为TCP:')
    udp_tcp_diff = abs(np.mean(udp_diff) - np.mean(tcp_diff))
    udp_dns_diff = abs(np.mean(udp_diff) - np.mean(dns_diff))
    print(f'   - UDP与TCP的差分均值差异: {udp_tcp_diff:.6f}')
    print(f'   - UDP与DNS的差分均值差异: {udp_dns_diff:.6f}')
    if udp_tcp_diff < udp_dns_diff:
        print('   - UDP更接近TCP，所以更容易被误判为TCP')

    print('\n4. 为什么DNS被误判为正常和TCP:')
    dns_normal_diff = abs(np.mean(dns_diff) - np.mean(np.abs(np.diff(df_normal.values))))
    dns_tcp_diff = abs(np.mean(dns_diff) - np.mean(tcp_diff))
    print(f'   - DNS与正常的差分均值差异: {dns_normal_diff:.6f}')
    print(f'   - DNS与TCP的差分均值差异: {dns_tcp_diff:.6f}')
    if dns_normal_diff < dns_tcp_diff:
        print('   - DNS更接近正常信号')
    else:
        print('   - DNS更接近TCP信号')

if __name__ == '__main__':
    analyze_udp_vs_dns()
