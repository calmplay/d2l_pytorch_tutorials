# -*- coding: utf-8 -*-
# @Time    : 2024/10/14 15:06
# @Author  : cfushn
# @Comments: 
# @Software: PyCharm
import pandas as pd
import torch
from torch.utils.data import DataLoader, TensorDataset
import torch.nn as nn
import torch.optim as optim
import matplotlib.pyplot as plt

# 加载数据
data_path = '../temp/data_melon.xlsx'
data = pd.read_excel(data_path)
X = data[['密度', '含糖率']].values  # 输入特征
y = data['好瓜'].values  # 输出标签，是否为好瓜 (0 或 1)

# 转换为 PyTorch 张量
X_tensor = torch.tensor(X, dtype=torch.float32)
y_tensor = torch.tensor(y, dtype=torch.float32).view(-1, 1)

# 构建数据集并创建 DataLoader
dataset = TensorDataset(X_tensor, y_tensor)
dataloader = DataLoader(dataset, batch_size=len(dataset), shuffle=True)

# 定义对数几率回归模型
class LogisticRegressionModel(nn.Module):
    def __init__(self):
        super(LogisticRegressionModel, self).__init__()
        self.linear = nn.Linear(2, 1)  # 输入 2 个特征，输出 1 个二分类标签

    def forward(self, x):
        return torch.sigmoid(self.linear(x))

# 初始化模型
model = LogisticRegressionModel()

# 定义损失函数和优化器
criterion = nn.BCELoss()  # 二元交叉熵损失
optimizer = optim.SGD(model.parameters(), lr=0.01)  # 学习率为 0.01

# 训练模型
num_epochs = 1000  # 迭代次数
for epoch in range(num_epochs):
    for inputs, labels in dataloader:
        # 前向传播
        outputs = model(inputs)
        loss = criterion(outputs, labels)

        # 反向传播和优化
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

    # 每 100 次迭代打印一次损失
    if (epoch+1) % 100 == 0:
        print(f'Epoch [{epoch+1}/{num_epochs}], Loss: {loss.item():.4f}')
        # (for语句不会创建新的局部作用域, 这里的loss可有效访问到)

# 获取模型的输出
with torch.no_grad():
    y_pred = model(X_tensor).numpy()

# 根据预测结果进行分类
y_pred_class = (y_pred > 0.5).astype(int)

# 绘制数据点
good_melon = X[y_pred_class.flatten() == 1]
bad_melon = X[y_pred_class.flatten() == 0]

plt.scatter(good_melon[:, 0], good_melon[:, 1], color='g', label='good')
plt.scatter(bad_melon[:, 0], bad_melon[:, 1], color='r', label='bad')
plt.xlabel('density')
plt.ylabel('sugar_content')
plt.legend()
plt.title('nice melon')
plt.show()
