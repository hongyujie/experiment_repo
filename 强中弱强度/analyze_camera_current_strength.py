import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import os
from pathlib import Path

# 设置中文字体
plt.rcParams['font.sans-serif'] = ['SimHei']  # 用来正常显示中文标签
plt.rcParams['axes.unicode_minus'] = False  # 用来正常显示负号

# 获取当前脚本所在目录（项目根目录）
project_root = Path(__file__).parent

# 数据文件路径
data_dir = project_root / "data_process"
img_dir = project_root / "img"

# 确保图片目录存在
img_dir.mkdir(exist_ok=True)

# 读取十三种攻击类型数据文件
def load_all_attacks_data():
    data_files = {
        "正常": "normal_40Hz.csv",
        "TCP攻击-弱": "tcp_low_40Hz.csv",
        "TCP攻击-中": "tcp_middle_40Hz.csv",
        "TCP攻击-强": "tcp_high_40Hz.csv",
        "UDP攻击-弱": "udp_low_40Hz.csv",
        "UDP攻击-中": "udp_middle_40Hz.csv",
        "UDP攻击-强": "udp_high_40Hz.csv",
        "ICMP攻击-弱": "icmp_low_40Hz.csv",
        "ICMP攻击-中": "icmp_middle_40Hz.csv",
        "ICMP攻击-强": "icmp_high_40Hz.csv",
        "DNS攻击-弱": "dns_low_40Hz.csv",
        "DNS攻击-中": "dns_middle_40Hz.csv",
        "DNS攻击-强": "dns_high_40Hz.csv"
    }
    
    all_data = {}
    for attack_type, filename in data_files.items():
        file_path = data_dir / filename
        if file_path.exists():
            df = pd.read_csv(file_path)
            # 检查列名并统一格式
            if len(df.columns) >= 2:
                df.columns = ['时间(s)', '电流(mA)']  # 重命名列
                all_data[attack_type] = df
                print(f"已加载 {attack_type} 数据，共 {len(df)} 个数据点")
            else:
                print(f"警告：文件 {filename} 列数不足")
        else:
            print(f"警告：文件 {filename} 不存在")
    
    return all_data

# 绘制十三种攻击类型叠加图
def plot_all_attacks_overlay(all_data):
    """绘制十三种攻击类型叠加图"""
    
    # 创建更大的图表以适应13个数据系列
    plt.figure(figsize=(16, 10))
    
    # 使用13种不同的颜色方案
    colors = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd', 
              '#8c564b', '#e377c2', '#7f7f7f', '#bcbd22', '#17becf',
              '#aec7e8', '#ffbb78', '#98df8a']
    
    # 为每种攻击类型绘制曲线
    attack_types = list(all_data.keys())
    
    for i, attack_type in enumerate(attack_types):
        df = all_data[attack_type]
        # 绘制前500个点以避免图像过于密集
        time_data = df['时间(s)'].iloc[:500]
        current_data = df['电流(mA)'].iloc[:500]
        
        # 根据攻击类型设置不同的线宽和透明度
        if attack_type == "正常":
            linewidth = 2.0
            alpha = 1.0
        elif "强" in attack_type:
            linewidth = 2.0
            alpha = 0.9
        elif "中" in attack_type:
            linewidth = 1.8
            alpha = 0.8
        else:
            linewidth = 1.6
            alpha = 0.7
        
        plt.plot(time_data, current_data, 
                label=attack_type, 
                linewidth=linewidth,
                alpha=alpha, 
                color=colors[i % len(colors)])
    
    plt.title('十三种攻击类型下摄像头电流对比图', fontsize=16, fontweight='bold', pad=20)
    plt.xlabel('时间 (s)', fontsize=12)
    plt.ylabel('电流 (mA)', fontsize=12)
    
    # 优化图例显示，使用两列布局
    plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left', borderaxespad=0., 
              ncol=2, fontsize=9)
    
    plt.grid(True, alpha=0.3, linestyle='--')
    
    # 调整布局，为图例留出更多空间
    plt.tight_layout()
    plt.subplots_adjust(right=0.75)
    
    plt.savefig(img_dir / 'all_attacks_overlay.png', dpi=300, bbox_inches='tight')
    plt.show()
    
    print("十三种攻击类型叠加图已生成")

# 绘制TCP攻击强度叠加图
def plot_tcp_strength_overlay(all_data):
    """绘制TCP攻击强度叠加图"""
    
    # 筛选TCP相关的攻击类型
    tcp_related_attacks = ["正常", "TCP攻击-弱", "TCP攻击-中", "TCP攻击-强"]
    
    # 创建图表
    plt.figure(figsize=(14, 8))
    
    # 使用专门的颜色方案，突出强度差异
    colors = {
        "正常": '#2ca02c',      # 绿色 - 正常状态
        "TCP攻击-弱": '#1f77b4',  # 蓝色 - 弱攻击
        "TCP攻击-中": '#ff7f0e',  # 橙色 - 中攻击
        "TCP攻击-强": '#d62728'   # 红色 - 强攻击
    }
    
    # 线宽设置，强度越高线越粗
    linewidths = {
        "正常": 1.5,
        "TCP攻击-弱": 1.8,
        "TCP攻击-中": 2.1,
        "TCP攻击-强": 2.5
    }
    
    # 为每个强度级别绘制曲线
    for attack_type in tcp_related_attacks:
        if attack_type in all_data:
            df = all_data[attack_type]
            # 绘制前600个点
            time_data = df['时间(s)'].iloc[:600]
            current_data = df['电流(mA)'].iloc[:600]
            
            plt.plot(time_data, current_data, 
                    label=attack_type, 
                    linewidth=linewidths[attack_type],
                    alpha=0.9, 
                    color=colors[attack_type])
    
    plt.title('TCP攻击强度对摄像头电流影响分析', fontsize=16, fontweight='bold', pad=20)
    plt.xlabel('时间 (s)', fontsize=12)
    plt.ylabel('电流 (mA)', fontsize=12)
    
    # 优化图例显示
    plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left', borderaxespad=0., 
              fontsize=11)
    
    plt.grid(True, alpha=0.3, linestyle='--')
    
    # 调整布局
    plt.tight_layout()
    plt.subplots_adjust(right=0.85)
    
    plt.savefig(img_dir / 'tcp_strength_overlay.png', dpi=300, bbox_inches='tight')
    plt.show()
    
    print("TCP攻击强度叠加图已生成")

# 统计信息显示
def display_all_attacks_statistics(all_data):
    """显示十三种攻击类型的统计信息"""
    print("\n" + "="*60)
    print("十三种攻击类型电流数据统计信息")
    print("="*60)
    
    stats_data = []
    
    for attack_type, df in all_data.items():
        current_data = df['电流(mA)']
        stats = {
            '攻击类型': attack_type,
            '数据点数': len(current_data),
            '平均值(mA)': round(current_data.mean(), 2),
            '标准差(mA)': round(current_data.std(), 2),
            '最小值(mA)': round(current_data.min(), 2),
            '最大值(mA)': round(current_data.max(), 2),
            '变化范围(mA)': round(current_data.max() - current_data.min(), 2)
        }
        stats_data.append(stats)
        
        print(f"\n{attack_type}:")
        print(f"  数据点数: {stats['数据点数']}")
        print(f"  平均值: {stats['平均值(mA)']} mA")
        print(f"  标准差: {stats['标准差(mA)']} mA")
        print(f"  最小值: {stats['最小值(mA)']} mA")
        print(f"  最大值: {stats['最大值(mA)']} mA")
        print(f"  变化范围: {stats['变化范围(mA)']} mA")
    
    # 保存统计信息到CSV
    stats_df = pd.DataFrame(stats_data)
    stats_df.to_csv(img_dir / 'all_attacks_statistics.csv', index=False, encoding='utf-8-sig')
    print(f"\n统计信息已保存到: img/all_attacks_statistics.csv")

# 分析攻击类型差异
def analyze_attack_differences(all_data):
    """分析不同攻击类型与正常状态的差异"""
    print("\n" + "="*60)
    print("攻击类型与正常状态差异分析")
    print("="*60)
    
    if "正常" in all_data:
        normal_mean = all_data["正常"]['电流(mA)'].mean()
        print(f"正常状态平均电流: {normal_mean:.2f} mA")
        
        # 按攻击类型分组分析
        attack_categories = {
            "TCP攻击": ["TCP攻击-弱", "TCP攻击-中", "TCP攻击-强"],
            "UDP攻击": ["UDP攻击-弱", "UDP攻击-中", "UDP攻击-强"],
            "ICMP攻击": ["ICMP攻击-弱", "ICMP攻击-中", "ICMP攻击-强"],
            "DNS攻击": ["DNS攻击-弱", "DNS攻击-中", "DNS攻击-强"]
        }
        
        for category, attacks in attack_categories.items():
            print(f"\n{category}:")
            for attack_type in attacks:
                if attack_type in all_data:
                    attack_mean = all_data[attack_type]['电流(mA)'].mean()
                    difference = attack_mean - normal_mean
                    percentage = (difference / normal_mean) * 100
                    
                    print(f"  {attack_type}:")
                    print(f"    平均电流: {attack_mean:.2f} mA")
                    print(f"    与正常状态差异: {difference:+.2f} mA ({percentage:+.1f}%)")

# 主函数
def main():
    print("开始分析十三种攻击类型对摄像头电流的影响...")
    
    # 加载数据
    all_data = load_all_attacks_data()
    
    if not all_data:
        print("错误：没有加载到任何数据！")
        return
    
    # 显示统计信息
    display_all_attacks_statistics(all_data)
    
    # 分析攻击类型差异
    analyze_attack_differences(all_data)
    
    # 绘制十三种攻击类型叠加图
    print("\n正在生成十三种攻击类型叠加图...")
    plot_all_attacks_overlay(all_data)
    
    # 绘制TCP攻击强度叠加图
    print("正在生成TCP攻击强度叠加图...")
    plot_tcp_strength_overlay(all_data)
    
    print("\n" + "="*60)
    print("分析完成！")
    print("生成的文件：")
    print("- img/all_attacks_overlay.png (十三种攻击类型叠加图)")
    print("- img/tcp_strength_overlay.png (TCP攻击强度叠加图)")
    print("- img/all_attacks_statistics.csv (统计信息)")
    print("="*60)

if __name__ == "__main__":
    main()