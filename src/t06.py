# -*- coding: utf-8 -*-
# @Time    : 2024/9/26 23:14
# @Author  : cfushn
# @Comments: 
# @Software: PyCharm

import torch
from d2l.torch import Animator, Accumulator, accuracy


def evaluate_accuracy(net, data_iter):
    """计算在指定数据集上模型的精度"""
    if isinstance(net, torch.nn.Module):
        net.eval()  # 将模型设置为评估模式(与梯度相关的功能就会关闭)
    metric = Accumulator(2)  # 正确预测数、预测总数 # Accumulator是一个自定义的累加器类
    with torch.no_grad():
        for X, y in data_iter:
            metric.add(accuracy(net(X), y), y.numel())
    return metric[0] / metric[1]


# 单个epoch的训练
def train_epoch_ch3(net, train_iter, loss, updater):
    """训练模型一个迭代周期"""
    # 如果是nn模型,开启训练模式
    if isinstance(net, torch.nn.Module):
        net.train()
    # 长度为3的累加器,3个要记录累加的变量:训练损失总和、训练准确度总和、样本数
    metric = Accumulator(3)
    for X, y in train_iter:
        # 计算梯度并更新参数
        y_hat = net(X)
        l = loss(y_hat, y)
        if isinstance(updater, torch.optim.Optimizer):
            # 如果是PyTorch内置的优化器
            updater.zero_grad()
            l.backward()
            updater.step()
            metric.add(float(l) * len(y), accuracy(y_hat, y), y.numel())
        else:
            # 如果是自定义的优化器
            l.sum().backward()
            updater(X.shape[0])  # X.shape[0]即批量大小
            metric.add(float(l.sum()), accuracy(y_hat, y), y.numel())
    # 返回训练损失,训练精度(正确率)
    return metric[0] / metric[2], metric[1] / metric[2]


# 训练函数
def train_ch3(net, train_iter, test_iter, loss, num_epochs, updater):
    """训练模型"""
    animator = Animator(xlabel='epoch', xlim=[1, num_epochs], ylim=[0.3, 0.9],
                        legend=['train loss', 'train acc', 'test acc'])
    for epoch in range(num_epochs):
        train_metrics = train_epoch_ch3(net, train_iter, loss, updater)
        # 在测试数据集上评估模型的精度
        test_acc = evaluate_accuracy(net, test_iter)
        # 动画
        animator.add(epoch + 1, train_metrics + (test_acc,))
        print(train_metrics)
    train_loss, train_acc = train_metrics
