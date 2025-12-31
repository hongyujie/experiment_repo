1. 在PyTorch中，有两种获取DataLoader数据的方法
```python
# 方法一：使用for循环遍历DataLoader
for data, labels in train_loader:
    # 训练模型
    optimizer.zero_grad()
    outputs = model(data)
    loss = criterion(outputs, labels)
    loss.backward()
    optimizer.step()

# 方法二：使用iter()和next()函数手动获取数据
data_iter = iter(train_loader)
inputs, labels = next(data_iter)
```
2. 实验中数据的输入规格为data:[128, 1, 64],labels:[128]
3. 检查模型每一层的输出
```python
# 检查模型每一层的输出
layer1 = torch.nn.Sequential(
    torch.nn.Conv1d(1, 16, kernel_size=3, padding=1),
    torch.nn.ReLU(),
    torch.nn.Dropout(dropout),
    torch.nn.MaxPool1d(2)
    )

layer2 = torch.nn.Sequential(
    torch.nn.Conv1d(16, 32, kernel_size=3, padding=1),
    torch.nn.ReLU(),
    torch.nn.Dropout(dropout),
    torch.nn.MaxPool1d(2)
    )

layer3 = torch.nn.Sequential(
    torch.nn.Conv1d(32, 64, kernel_size=3, padding=1),
    torch.nn.ReLU(),
    torch.nn.Dropout(dropout),
    torch.nn.MaxPool1d(2)
    )

NetWork = torch.nn.Sequential(
    layer1,
    layer2,
    layer3,
    torch.nn.Flatten(),
    torch.nn.Linear(512, 32),
    torch.nn.Linear(32, 2)
)

X = torch.rand(size=(128, 1, 64))
for layer in NetWork:
    X = layer(X)
    print(layer.__class__.__name__,'output shape:\t', X.shape)
```
4. torch.nn.Conv1d和torch.nn.Conv2d的区别：
   - torch.nn.Conv1d：输入数据的维度为(batch_size, in_channels, seq_len)，输出数据的维度为(batch_size, out_channels, seq_len)
   - torch.nn.Conv2d：输入数据的维度为(batch_size, in_channels, height, width)，输出数据的维度为(batch_size, out_channels, height, width)
5. 样本不平衡：指的是在机器学习任务中，不同类别的样本数量不均衡，导致模型偏向数量较多的类别，少数类没有得到足够的重视，导致少数类别的分类精度降低。处理的方法有：
   - 上采样（Oversampling）：通过复制或生成新的少数类样本，使其数量与多数类样本数量相等或接近。
   - 下采样（Undersampling）：指从多数类样本中随机抽取一部分样本，使其数量与少数类样本数量相等或接近。
   - 修改超参数class_weight：在训练模型时，通过设置class_weight参数，为不同类别分配不同的权重，使模型在训练时更加关注少数类样本。
    >例如原本没有类别权重的时候：
    总损失 = 损失(样本1) + 损失(样本2) + ... + 损失(样本N)
    有类别权重：
    总损失 = 权重0×损失(标签0样本) + 权重1×损失(标签1样本)
    在这次实验中：
    标签0（正常）：14,353个样本（权重1.0）
    标签1（攻击）：3,619个样本（权重4.0）
    相当于每个标签1样本的损失被放大4倍
6. 混合采样（Mixed Sampling）：指的是在训练模型时，同时使用上采样和下采样等技术，以平衡不同类别的样本数量，提高模型的分类精度。
   基于SMOTEENN混合采样：
7. OCSVM由于不适合使用混合采样，实验结果不要好，因为1类的样本数量要少。