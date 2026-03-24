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
data_process_dir = project_root / "data_process"
comparison_dir = project_root / "batch_comparison"

# 确保目录存在
os.makedirs(data_process_dir, exist_ok=True)
os.makedirs(comparison_dir, exist_ok=True)

# 改进的去除尖峰函数 - 结合IQR和局部窗口方法（与data_preview_comparison.py相同）
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
            if len(window_without_current) > 0:
                replacement_value = window_without_current.mean()
            else:
                # 如果窗口内没有其他数据点，使用全局平均值
                replacement_value = data[column].mean()
            
            data_cleaned.loc[i, column] = replacement_value
    
    total_outliers = global_outliers + local_outliers
    print(f"检测到 {global_outliers} 个全局异常值，{local_outliers} 个局部异常值，共 {total_outliers} 个尖峰")
    
    return data_cleaned

# 生成处理前后对比图（与data_preview_comparison.py相同）
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
    plt.savefig(comparison_dir / f"{filename}_comparison.png", dpi=300, bbox_inches='tight')
    plt.close()
    
    print(f"对比图已保存到: batch_comparison/{filename}_comparison.png")

# 智能获取电流列名
def get_current_column(df):
    """智能获取电流列名，支持多种可能的列名"""
    possible_columns = ['F', '电流(mA)', '电流', 'current', 'Current']
    for col in possible_columns:
        if col in df.columns:
            return col
    return None

# 处理单个数据文件
def process_single_file(csv_file):
    """处理单个数据文件"""
    print(f"\n处理文件: {csv_file.name}")
    print("="*60)
    
    # 读取数据
    df = pd.read_csv(csv_file)
    
    # 检查列名并重命名
    if len(df.columns) >= 2:
        # 智能获取电流列名
        current_column = get_current_column(df)
        
        if current_column:
            # 如果列名不是'电流(mA)'，则重命名
            if current_column != '电流(mA)':
                df.rename(columns={current_column: '电流(mA)'}, inplace=True)
                current_column = '电流(mA)'
            
            # 确保有时间列
            if 't' in df.columns:
                df.rename(columns={'t': '时间(s)'}, inplace=True)
            elif '时间' not in df.columns[0]:
                df.columns = ['时间(s)', '电流(mA)']
        else:
            # 默认处理：第一列时间，第二列电流
            df.columns = ['时间(s)', '电流(mA)']
            current_column = '电流(mA)'
    else:
        print(f"警告：文件 {csv_file.name} 列数不足")
        return None
    
    # 显示原始数据统计信息
    print("原始数据统计信息:")
    print(f"  数据点数: {len(df)}")
    print(f"  平均值: {df[current_column].mean():.3f} mA")
    print(f"  标准差: {df[current_column].std():.3f} mA")
    print(f"  最小值: {df[current_column].min():.3f} mA")
    print(f"  最大值: {df[current_column].max():.3f} mA")
    
    # 使用改进的方法去除尖峰
    print("\n开始数据预处理...")
    df_cleaned = improved_remove_spikes(df)
    
    # 显示处理后数据统计信息
    print("\n处理后数据统计信息:")
    print(f"  数据点数: {len(df_cleaned)}")
    print(f"  平均值: {df_cleaned[current_column].mean():.3f} mA")
    print(f"  标准差: {df_cleaned[current_column].std():.3f} mA")
    print(f"  最小值: {df_cleaned[current_column].min():.3f} mA")
    print(f"  最大值: {df_cleaned[current_column].max():.3f} mA")
    
    # 保存处理后的数据
    output_file = data_process_dir / csv_file.name
    df_cleaned.to_csv(output_file, index=False)
    
    # 生成对比图（只处理前1000个点以节省时间）
    print("\n生成对比图...")
    sample_size = min(1000, len(df))
    df_sample = df.head(sample_size).copy()
    df_cleaned_sample = df_cleaned.head(sample_size).copy()
    
    plot_comparison(df_sample, df_cleaned_sample, csv_file.stem)
    
    print(f"\n文件 {csv_file.name} 处理完成！")
    print(f"处理后的数据保存到: data_process/{csv_file.name}")
    print("="*60)
    
    return df_cleaned

# 批量处理所有数据文件
def batch_process_all_data():
    """批量处理所有数据文件"""
    
    # 获取data目录下的所有CSV文件
    csv_files = list(data_dir.glob("*.csv"))
    
    print("开始批量处理数据文件...")
    print(f"找到 {len(csv_files)} 个数据文件:")
    
    for i, csv_file in enumerate(csv_files, 1):
        print(f"{i}. {csv_file.name}")
    
    print("\n" + "="*60)
    
    # 处理统计
    processed_count = 0
    failed_count = 0
    
    # 批量处理所有文件
    for csv_file in csv_files:
        try:
            result = process_single_file(csv_file)
            if result is not None:
                processed_count += 1
            else:
                failed_count += 1
        except Exception as e:
            print(f"处理文件 {csv_file.name} 时出错: {e}")
            failed_count += 1
    
    # 输出处理结果统计
    print("\n" + "="*60)
    print("批量处理完成！")
    print(f"成功处理: {processed_count} 个文件")
    print(f"处理失败: {failed_count} 个文件")
    print(f"处理后的数据保存在: data_process/")
    print(f"对比图保存在: batch_comparison/")
    print("="*60)

# 主函数
def main():
    print("批量数据预处理工具")
    print("功能：使用改进的双重检测方法批量处理所有数据文件")
    print("="*60)
    
    # 批量处理所有数据
    batch_process_all_data()

if __name__ == "__main__":
    main()