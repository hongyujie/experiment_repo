import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import os
from pathlib import Path

# 设置中文字体
plt.rcParams['font.sans-serif'] = ['SimHei']  # 用来正常显示中文标签
plt.rcParams['axes.unicode_minus'] = False  # 用来正常显示负号

# 数据文件路径
data_dir = Path("data")

# 读取所有数据文件
def load_all_data():
    data_files = {
        "正常": "normal_40Hz.csv",
        "TCP+ICMP攻击": "tcp_icmp_40Hz.csv", 
        "TCP+UDP攻击": "tcp_udp_40Hz.csv",
        "TCP+DNS攻击": "tcp_dns_40Hz.csv",
        "ICMP+UDP攻击": "icmp_udp_40Hz.csv",
        "ICMP+DNS攻击": "icmp_dns_40Hz.csv",
        "UDP+DNS攻击": "udp_dns_40Hz.csv"
    }
    
    all_data = {}
    for attack_type, filename in data_files.items():
        file_path = data_dir / filename
        if file_path.exists():
            df = pd.read_csv(file_path)
            df.columns = ['时间(s)', '电流(mA)']  # 重命名列
            all_data[attack_type] = df
            print(f"已加载 {attack_type} 数据，共 {len(df)} 个数据点")
        else:
            print(f"警告：文件 {filename} 不存在")
    
    return all_data

# 绘制所有攻击类型的叠加图
def plot_overlay_comparison(all_data):
    plt.figure(figsize=(12, 8))
    
    # 使用更鲜明、对比度更高的颜色组合
    colors = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd', '#8c564b', '#e377c2', '#7f7f7f', '#bcbd22', '#17becf']
    
    for i, (attack_type, df) in enumerate(all_data.items()):
        # 只绘制前1000个点以避免图像过于密集
        time_data = df['时间(s)'].iloc[:1000]
        current_data = df['电流(mA)'].iloc[:1000]
        plt.plot(time_data, current_data, label=attack_type, linewidth=1, alpha=0.8, color=colors[i % len(colors)])
    
    plt.title('不同攻击类型下摄像头电流对比')
    plt.xlabel('时间 (s)')
    plt.ylabel('电流 (mA)')
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig('img/overlay_comparison.png', dpi=300, bbox_inches='tight')
    plt.show()

# 主函数
def main():
    # 确保img目录存在
    os.makedirs('img', exist_ok=True)
    
    print("开始分析摄像头电流数据...")
    
    # 加载数据
    all_data = load_all_data()
    
    if not all_data:
        print("未找到任何数据文件！")
        return
    
    # 绘制叠加对比图
    print("\n正在生成图表...")
    plot_overlay_comparison(all_data)
    
    print("\n分析完成！叠加对比图已保存到 img/overlay_comparison.png")

if __name__ == "__main__":
    main()