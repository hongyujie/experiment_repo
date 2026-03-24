import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import os

plt.rcParams['font.sans-serif'] = ['SimHei']
plt.rcParams['axes.unicode_minus'] = False

ATTACK_MAPPING = {
    'normal_40Hz.csv': ('正常', 'green'),
    'tcp_high_40Hz.csv': ('TCP', 'blue'),
    'udp_high_40Hz.csv': ('UDP', 'red'),
    'icmp_high_40Hz.csv': ('ICMP', 'orange'),
    'dns_high_40Hz.csv': ('DNS', 'purple')
}

def analyze_statistics(data, name):
    """分析单个信号的统计特征"""
    data = data.flatten()
    stats = {
        '均值': np.mean(data),
        '标准差': np.std(data),
        '最小值': np.min(data),
        '最大值': np.max(data),
        '峰峰值': np.max(data) - np.min(data),
        '均方根': np.sqrt(np.mean(data**2)),
        '偏度': np.mean(((data - np.mean(data)) / (np.std(data) + 1e-8))**3),
        '峰度': np.mean(((data - np.mean(data)) / (np.std(data) + 1e-8))**4) - 3
    }

    fft = np.fft.fft(data)
    power = np.abs(fft)**2
    freqs = np.fft.fftfreq(len(data))

    pos_mask = freqs > 0
    pos_freqs = freqs[pos_mask]
    pos_power = power[pos_mask]

    total_power = np.sum(power)
    dc_power = power[0] / total_power * 100 if total_power > 0 else 0

    top_freqs_idx = np.argsort(pos_power)[-5:][::-1]
    top_freqs = pos_freqs[top_freqs_idx]
    top_powers = pos_power[top_freqs_idx]

    print(f'\n{"="*50}')
    print(f'{name} 统计分析')
    print(f'{"="*50}')
    print(f'样本数: {len(data)}')
    print(f'均值: {stats["均值"]:.4f}')
    print(f'标准差: {stats["标准差"]:.4f}')
    print(f'峰峰值: {stats["峰峰值"]:.4f}')
    print(f'均方根: {stats["均方根"]:.4f}')
    print(f'偏度: {stats["偏度"]:.4f}')
    print(f'峰度: {stats["峰度"]:.4f}')
    print(f'直流分量占比: {dc_power:.2f}%')
    print(f'前5频率成分: {", ".join([f"{f:.2f}Hz" for f in top_freqs[:5]])}')

    return stats, top_freqs, top_powers

def main():
    print('='*60)
    print('训练数据分析')
    print('='*60)

    data_dir = 'data_process'
    all_stats = {}
    all_top_freqs = {}
    all_top_powers = {}

    os.makedirs('img', exist_ok=True)

    for filename, (name, color) in ATTACK_MAPPING.items():
        filepath = os.path.join(data_dir, filename)
        if os.path.exists(filepath):
            df = pd.read_csv(filepath)
            if 'F' in df.columns:
                data = df['F'].values
            else:
                data = df['电流(mA)'].values

            stats, top_freqs, top_powers = analyze_statistics(data, name)
            all_stats[name] = stats
            all_top_freqs[name] = top_freqs
            all_top_powers[name] = top_powers

    print('\n' + '='*60)
    print('类别间统计特征对比')
    print('='*60)

    stats_df = pd.DataFrame(all_stats).T
    print(stats_df.to_string())

    print('\n' + '='*60)
    print('关键发现')
    print('='*60)

    normal_stats = all_stats['正常']
    print('\n正常信号参考值:')
    print(f'  均值: {normal_stats["均值"]:.2f}')
    print(f'  标准差: {normal_stats["标准差"]:.4f}')
    print(f'  峰峰值: {normal_stats["峰峰值"]:.4f}')

    print('\n各类别与正常的差异:')
    for name in ['TCP', 'UDP', 'ICMP', 'DNS']:
        if name in all_stats:
            diff = all_stats[name]
            mean_diff = abs(diff['均值'] - normal_stats['均值'])
            std_diff = abs(diff['标准差'] - normal_stats['标准差'])
            range_diff = abs(diff['峰峰值'] - normal_stats['峰峰值'])
            print(f'\n{name}:')
            print(f'  均值差异: {mean_diff:.4f} ({mean_diff/normal_stats["均值"]*100:.2f}%)')
            print(f'  标准差差异: {std_diff:.4f} ({std_diff/normal_stats["标准差"]*100:.2f}%)')
            print(f'  峰峰值差异: {range_diff:.4f} ({range_diff/normal_stats["峰峰值"]*100:.2f}%)')

    print('\n' + '='*60)
    print('结论')
    print('='*60)

    all_stds = [all_stats[name]['标准差'] for name in all_stats]
    all_ranges = [all_stats[name]['峰峰值'] for name in all_stats]

    std_cv = np.std(all_stds) / np.mean(all_stds) * 100
    range_cv = np.std(all_ranges) / np.mean(all_ranges) * 100

    print(f'\n标准差变异系数: {std_cv:.2f}%')
    print(f'峰峰值变异系数: {range_cv:.2f}%')

    if std_cv < 10:
        print('\n⚠️ 警告: 各类别的标准差非常接近（差异<10%）')
        print('   这表明信号的波动程度在不同攻击类型间基本相同')
        print('   模型难以仅通过波动程度区分攻击类型')

    if range_cv < 10:
        print('\n⚠️ 警告: 各类别的峰峰值非常接近（差异<10%）')
        print('   这表明信号的变化范围在不同攻击类型间基本相同')
        print('   模型难以通过信号的幅度变化区分攻击类型')

    print('\n📊 数据可行性评估:')
    if std_cv < 10 and range_cv < 10:
        print('   ❌ 数据本身缺乏区分性特征')
        print('   ❌ 不同攻击类型的电流信号统计上几乎无法区分')
        print('   ❌ 不建议仅用电流信号进行五分类')
    else:
        print('   ✅ 数据存在一定的区分性')

if __name__ == '__main__':
    main()
