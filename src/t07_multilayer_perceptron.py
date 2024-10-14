# %% md
# ### 多层感知机的从零开始实现
# %%
import torch
from torch import nn
from d2l import torch as d2l

batch_size = 256
train_iter, test_iter = d2l.load_data_fashion_mnist(batch_size)  ## ../data
# %%
# 实现一个具有单隐藏层的多层感知机, 它包含256个隐藏单元
num_inputs = 784  # 输入维度784=28*28展平
num_hiddens = 256  # 超参数:隐藏层256个单元(10<256<784)
num_outputs = 10

# todo, 两个w初始不是随机,而是0会怎样?
W1 = nn.Parameter(
        torch.randn(num_inputs, num_hiddens, requires_grad=True))
b1 = nn.Parameter(torch.zeros(num_hiddens, requires_grad=True))
W2 = nn.Parameter(
        torch.randn(num_hiddens, num_outputs, requires_grad=True))
b2 = nn.Parameter(torch.zeros(num_outputs, requires_grad=True))

params = [W1, b1, W2, b2]
print(W1.shape, b1.shape, W2.shape, b2.shape)


# %%
# ReLU激活函数
def relu(X):
    a = torch.zeros_like(X)
    return torch.max(X, a)


# %%
# 模型
def net(X):
    X = X.reshape((-1, num_inputs))  # -1表示拉成batch_size, inputs=784
    H = relu(X @ W1 + b1)  # 这里“@”代表矩阵乘法
    return H @ W2 + b2


loss = nn.CrossEntropyLoss()
# %% md
# $\mathbf h=\sigma(\mathbf{W_1X}+\mathbf {b_1})$<br>
# $\mathbf o=\mathbf{W_2}\mathbf h+\mathbf{b_2}$
# %%
from src.t06 import train_ch3

num_epochs, lr = 10, 0.1
updater = torch.optim.SGD(params, lr=lr)

train_ch3(net, train_iter, test_iter, loss, num_epochs, updater)
# %% md
# ---
# ### 多层感知机的简洁实现
# %%
import torch
from torch import nn
from d2l import torch as d2l

# %%
net = nn.Sequential(
        nn.Flatten(),  # 二维张量展成一维
        nn.Linear(784, 256),
        nn.ReLU(),
        nn.Linear(256, 10),
)


def init_weights(m):
    if type(m) == nn.Linear:
        nn.init.normal_(m.weight, std=0.01)


net.apply(init_weights)  # 初始化权重 (Apply fn recursively to every submodule)
# %%
batch_size, lr, num_epochs = 256, 0.1, 10
loss = nn.CrossEntropyLoss()
trainer = torch.optim.SGD(net.parameters(), lr=lr)

train_iter, test_iter = d2l.load_data_fashion_mnist(batch_size)
train_ch3(net, train_iter, test_iter, loss, num_epochs, trainer)
# %%
