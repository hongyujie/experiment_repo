import pandas as pd

# 读取数据文件的前20行
data = pd.read_csv('password data-2.csv', header=None, nrows=20)

# 打印数据
print("数据文件前20行内容：")
print(data)

# 查看数据基本信息
data_full = pd.read_csv('password data-2.csv', header=None)
data_full.columns = ['timestamp', 'label', 'feature']
print("\n数据基本信息：")
print(f"总样本数: {len(data_full)}")
print(f"标签分布: {data_full['label'].value_counts()}")
print(f"feature列数据类型: {data_full['feature'].dtype}")
print(f"feature列的唯一值数量: {data_full['feature'].nunique()}")
print(f"\nfeature列的一些示例值：")
print(data_full['feature'].sample(10))