"""
icecream库基本用法演示
icecream是一个强大的调试工具，比print更智能和方便
"""

# 首先需要安装icecream库
# pip install icecream

from icecream import ic
import numpy as np
import torch

# 1. 基本用法 - 替代print
print("===== 1. 基本用法 =====")
x = 10
y = 20
z = x + y

# 传统print方式
print("x =", x)
print("y =", y)
print("z =", z)

# icecream方式（更简洁）
ic(x)
ic(y)
ic(z)

# 2. 自动显示变量名和值
print("\n===== 2. 自动显示变量名和值 =====")
name = "张三"
age = 25
salary = 8000.50

ic(name, age, salary)

# 3. 函数调用调试
print("\n===== 3. 函数调用调试 =====")
def calculate_bmi(weight, height):
    """计算BMI指数"""
    ic(weight, height)  # 调试输入参数
    bmi = weight / (height ** 2)
    ic(bmi)  # 调试计算结果
    return bmi

# 调用函数
result = calculate_bmi(70, 1.75)
ic(result)

# 4. 条件调试
print("\n===== 4. 条件调试 =====")
def process_data(data_list):
    """处理数据列表"""
    ic.enable()  # 启用调试
    total = 0
    for i, item in enumerate(data_list):
        ic(i, item)
        total += item
        if item > 50:  # 当值大于50时特别关注
            ic("发现大数值:", item)
    
    ic.disable()  # 禁用调试
    return total

data = [10, 25, 60, 30, 80, 15]
result = process_data(data)
ic(result)

# 5. 复杂数据结构调试
print("\n===== 5. 复杂数据结构调试 =====")
# 字典调试
person = {
    "name": "李四",
    "age": 30,
    "skills": ["Python", "机器学习", "深度学习"],
    "salary": 15000
}
ic(person)

# 列表调试
numbers = [1, 2, 3, 4, 5]
ic(numbers)

# NumPy数组调试
array = np.array([[1, 2, 3], [4, 5, 6]])
ic(array)

# PyTorch张量调试（与你的项目相关）
tensor = torch.tensor([[1.0, 2.0], [3.0, 4.0]])
ic(tensor)

# 6. 自定义前缀和格式化
print("\n===== 6. 自定义前缀和格式化 =====")
# 设置自定义前缀
ic.configureOutput(prefix='DEBUG | ')

def complex_calculation(a, b):
    ic(a, b)
    result = a * b + a - b
    ic(result)
    return result

complex_calculation(10, 5)

# 恢复默认前缀
ic.configureOutput(prefix='')

# 7. 在循环中使用
print("\n===== 7. 在循环中使用 =====")
def analyze_scores(scores):
    """分析分数"""
    passed = 0
    failed = 0
    
    for i, score in enumerate(scores):
        ic(i, score)
        if score >= 60:
            passed += 1
            ic("及格")
        else:
            failed += 1
            ic("不及格")
    
    ic(passed, failed)
    return passed, failed

scores = [85, 45, 90, 55, 70, 30]
passed, failed = analyze_scores(scores)
ic(f"及格人数: {passed}, 不及格人数: {failed}")

# 8. 与你的CNN项目结合的例子
print("\n===== 8. 与CNN项目结合的例子 =====")
def debug_cnn_training():
    """模拟CNN训练调试"""
    # 模拟训练数据
    batch_size = 64
    learning_rate = 0.01
    epochs = 10
    
    ic(batch_size, learning_rate, epochs)
    
    # 模拟训练过程
    for epoch in range(epochs):
        ic(epoch)
        
        # 模拟每个epoch的训练
        for batch in range(5):  # 简化，只模拟5个batch
            # 模拟损失计算
            loss = 1.0 / (epoch + 1) + 0.1 * batch
            accuracy = 0.8 + 0.02 * epoch + 0.005 * batch
            
            # 只在特定条件下显示详细信息
            if batch % 2 == 0:  # 每2个batch显示一次
                ic(epoch, batch, loss, accuracy)
    
    ic("训练完成")

debug_cnn_training()

# 恢复默认设置
ic.configureOutput(prefix='', includeContext=False)

print("\n===== icecream演示完成 =====")
print("icecream让调试变得更简单直观！")


