import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import os
from pathlib import Path

# 设置中文字体
plt.rcParams['font.sans-serif'] = ['SimHei']  # 用来正常显示中文标签
plt.rcParams['axes.unicode_minus'] = False  # 用来正常显示负号

# 获取当前脚本所在目录（项目根目录）
project_root = Path(__file__).parent

# 数据文件路径
data_dir = project_root / "data"

# 改进的去除尖峰函数 - 结合IQR和局部窗口方法
def improved_remove_spikes(data, column='电流(mA)', iqr_threshold=1.0, std_threshold=2.5, window_size=5):
    """
    结合IQR方法和局部窗口标准差检测去除数据中的尖峰
    data: DataFrame数据
    column: 要处理的列名
    iqr_threshold: IQR阈值倍数（更敏感）
    std_threshold: 标准差阈值
    window_size: 局部窗口大小
    """
    data_cleaned = data.copy()
    
    # 方法1：IQR全局异常值检测
    Q1 = data[column].quantile(0.25)
    Q3 = data[column].quantile(0.75)
    IQR = Q3 - Q1
    
    # 定义更敏感的异常值边界
    lower_bound = Q1 - iqr_threshold * IQR
    upper_bound = Q3 + iqr_threshold * IQR
    
    # 方法2：局部窗口标准差检测
    std_dev = data[column].std()
    half_window = window_size // 2
    
    # 统计检测到的异常值
    global_outliers = 0
    local_outliers = 0
    
    for i in range(len(data)):
        current_value = data.loc[i, column]
        
        # 检查是否为全局异常值（IQR方法）
        is_global_outlier = (current_value < lower_bound) or (current_value > upper_bound)
        
        # 检查是否为局部异常值（局部窗口方法）
        window_start = max(0, i - half_window)
        window_end = min(len(data), i + half_window + 1)
        
        # 排除当前点计算窗口平均值
        window_data = data.loc[window_start:window_end, column]
        window_without_current = window_data[window_data.index != i]
        
        if len(window_without_current) > 0:
            window_mean = window_without_current.mean()
            is_local_outlier = abs(current_value - window_mean) > std_threshold * std_dev
        else:
            is_local_outlier = False
        
        # 如果被任一方法检测为异常值，则进行修复
        if is_global_outlier or is_local_outlier:
            if is_global_outlier:
                global_outliers += 1
            if is_local_outlier:
                local_outliers += 1
            
            # 使用局部窗口平均值替换异常值（排除当前点）
            window_data = data.loc[window_start:window_end, column]
            window_without_current = window_data[window_data.index != i]
            
            if len(window_without_current) > 0:
                replacement_value = window_without_current.mean()
            else:
                # 如果窗口内没有其他数据点，使用全局平均值
                replacement_value = data[column].mean()
            
            data_cleaned.loc[i, column] = replacement_value
    
    total_outliers = global_outliers + local_outliers
    print(f"检测到 {global_outliers} 个全局异常值，{local_outliers} 个局部异常值，共 {total_outliers} 个尖峰")
    
    return data_cleaned

# 生成处理前后对比图
def plot_comparison(original_data, cleaned_data, filename, column='电流(mA)'):
    """绘制原始数据和处理后数据的详细对比图"""
    
    # 创建更大的图表
    fig, axes = plt.subplots(3, 1, figsize=(15, 12))
    
    # 子图1：原始数据
    axes[0].plot(original_data.index, original_data[column], 'b-', alpha=0.8, linewidth=1.5, label='原始数据')
    axes[0].set_title(f'{filename} - 原始数据', fontsize=14, fontweight='bold')
    axes[0].set_ylabel('电流 (mA)', fontsize=12)
    axes[0].legend()
    axes[0].grid(True, alpha=0.3)
    
    # 子图2：处理后数据
    axes[1].plot(cleaned_data.index, cleaned_data[column], 'r-', alpha=0.8, linewidth=1.5, label='处理后数据')
    axes[1].set_title(f'{filename} - 处理后数据', fontsize=14, fontweight='bold')
    axes[1].set_ylabel('电流 (mA)', fontsize=12)
    axes[1].legend()
    axes[1].grid(True, alpha=0.3)
    
    # 子图3：叠加对比
    axes[2].plot(original_data.index, original_data[column], 'b-', alpha=0.6, linewidth=1, label='原始数据')
    axes[2].plot(cleaned_data.index, cleaned_data[column], 'r-', alpha=0.8, linewidth=1.5, label='处理后数据')
    axes[2].set_title(f'{filename} - 叠加对比', fontsize=14, fontweight='bold')
    axes[2].set_xlabel('数据点索引', fontsize=12)
    axes[2].set_ylabel('电流 (mA)', fontsize=12)
    axes[2].legend()
    axes[2].grid(True, alpha=0.3)
    
    plt.tight_layout()
    
    # 保存对比图
    comparison_dir = project_root / "preview_comparison"
    os.makedirs(comparison_dir, exist_ok=True)
    plt.savefig(comparison_dir / f"{filename}_comparison.png", dpi=300, bbox_inches='tight')
    plt.show()
    
    print(f"对比图已保存到: preview_comparison/{filename}_comparison.png")

# 分析单个文件
def analyze_single_file(filename, sample_size=1000):
    """分析单个数据文件"""
    print(f"\n正在分析文件: {filename}")
    print("="*60)
    
    # 读取数据
    file_path = data_dir / filename
    df = pd.read_csv(file_path)
    
    # 检查列名并重命名
    if len(df.columns) >= 2:
        df.columns = ['时间(s)', '电流(mA)']
    else:
        print(f"警告：文件 {filename} 列数不足")
        return
    
    # 只处理前sample_size个数据点
    df_sample = df.head(sample_size).copy()
    
    # 显示原始数据统计信息
    print("原始数据统计信息:")
    print(f"  数据点数: {len(df_sample)}")
    print(f"  平均值: {df_sample['电流(mA)'].mean():.3f} mA")
    print(f"  标准差: {df_sample['电流(mA)'].std():.3f} mA")
    print(f"  最小值: {df_sample['电流(mA)'].min():.3f} mA")
    print(f"  最大值: {df_sample['电流(mA)'].max():.3f} mA")
    
    # 使用改进的方法去除尖峰
    print("\n开始数据预处理...")
    df_cleaned = improved_remove_spikes(df_sample)
    
    # 显示处理后数据统计信息
    print("\n处理后数据统计信息:")
    print(f"  数据点数: {len(df_cleaned)}")
    print(f"  平均值: {df_cleaned['电流(mA)'].mean():.3f} mA")
    print(f"  标准差: {df_cleaned['电流(mA)'].std():.3f} mA")
    print(f"  最小值: {df_cleaned['电流(mA)'].min():.3f} mA")
    print(f"  最大值: {df_cleaned['电流(mA)'].max():.3f} mA")
    
    # 生成对比图
    print("\n生成对比图...")
    plot_comparison(df_sample, df_cleaned, filename.replace('.csv', ''))
    
    print(f"\n文件 {filename} 分析完成！")
    print("="*60)

# 主函数
def main():
    print("数据预处理预览工具")
    print("功能：读取数据、预处理、生成对比图（不修改原数据）")
    print("="*60)
    
    # 列出所有可用的数据文件
    csv_files = list(data_dir.glob("*.csv"))
    print(f"找到 {len(csv_files)} 个数据文件:")
    
    for i, csv_file in enumerate(csv_files, 1):
        print(f"{i}. {csv_file.name}")
    
    # 让用户选择要分析的文件
    while True:
        try:
            choice = input("\n请选择要分析的文件编号 (输入0退出): ")
            choice = int(choice)
            
            if choice == 0:
                print("程序退出")
                break
            elif 1 <= choice <= len(csv_files):
                selected_file = csv_files[choice-1].name
                analyze_single_file(selected_file)
            else:
                print("无效的选择，请重新输入")
        except ValueError:
            print("请输入有效的数字")
        except KeyboardInterrupt:
            print("\n程序被用户中断")
            break

if __name__ == "__main__":
    main()