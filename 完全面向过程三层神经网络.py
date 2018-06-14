# !/usr/bin/python3
# -*- coding:utf-8 -*-

'''

Title : 完全面向过程的三层神经网络模型

Author : Ining

Reference ： 
  https://www.cnblogs.com/charlotte77/p/5629865.html
  https://blog.csdn.net/tudaodiaozhale/article/details/78632931

Update date :
  2018.6.5
  - 无法设置测试集 ：代入真实数据的时候再设置
  - 训练样本小 ：以后代入数量多一些的真实数据
  - 未向量化 ：无法拓展，隐藏层和节点数增多后无法计算速度慢
  - 手动初始化 ：以后加入随机初始化
'''

import numpy as np

# 超参数
alpha = 0.5
epoches = 10000

# 输入值（随意设置）
i1 = 0.06
i2 = 0.11
# 输出值（随意设置）
target1 = 0.03
target2 = 0.98

# 初始化权重、偏置（随意设置）
weight = [0.15, 0.2, 0.25, 0.3, 0.4, 0.45, 0.5, 0.55]
bias = [0.35, 0.60]

# sigmoid函数
def sigmoid(z):
    return 1.0 / (1.0 + np.exp(-z))
# sigmoid函数偏导
def dersig(y):
    return y * (1 - y)

for epoch in range(epoches):

    '''前向传播'''
    
    # 输入层 -> 隐藏层 / 隐藏层激活
    net_h1 = i1 * weight[0] + i2 * weight[1] + bias[0]
    out_h1 = sigmoid(net_h1)
    net_h2 = i1 * weight[2] + i2 * weight[3] + bias[0]
    out_h2 = sigmoid(net_h2)

    # 隐藏层 -> 输出层 / 输出层激活
    net_o1 = out_h1 * weight[4] + out_h2 * weight[5] + bias[1]
    out_o1 = sigmoid(net_o1)
    net_o2 = out_h1 * weight[6] + out_h2 * weight[7] + bias[1]
    out_o2 = sigmoid(net_o2)

    # 每一次迭代的误差
    print('epoch:', str(epoch + 1), '\n', 'target1:', str(target1 - out_o1), ',target2:', str(target2 - out_o2))
    # 每一次迭代的结果
    print('outputs:', '\n', 'y1:', str(out_o1), '\n', 'y2:', str(out_o2))

    # 来自o1和o2的误差
    e_o1 = 0.5 * (target1 - out_o1) ** 2
    e_o2 = 0.5 * (target2 - out_o2) ** 2
    # 总误差
    e_total = e_o1 + e_o2
    print('loss:', e_total, '\n')

    '''反向传播'''

    # 总误差 对 o1输出 的偏导数
    par_e_total_par_out_o1 = - (target1 - out_o1)
    # 总误差 对 o2输出 的偏导数
    par_e_total_par_out_o2 = - (target2 - out_o2)

    # o1输出 对 o1节点 的偏导数
    par_out_o1_par_net_o1 = dersig(out_o1)
    # o2输出 对 o2节点 的偏导数
    par_out_o2_par_net_o2 = dersig(out_o2)

    # o1节点 对 w5 的偏导数
    par_net_o1_par_w5 = out_h1
    # o1节点 对 w6 的偏导数
    par_net_o1_par_w6 = out_h2
    # o2节点 对 w7 的偏导数
    par_net_o2_par_w7 = out_h1
    # o2节点 对 w8 的偏导数
    par_net_o2_par_w8 = out_h2

    # o1节点 对 h1输出 的偏导数
    par_net_o1_par_out_h1 = weight[4]
    # o1节点 对 h2输出 的偏导数
    par_net_o1_par_out_h2 = weight[5]
    # o1节点 对 b2 的偏导数
    par_net_o1_par_b2 = 1
    # o2节点 对 h1输出 的偏导数
    par_net_o2_par_out_h1 = weight[6]
    # o2节点 对 h2输出 的偏导数
    par_net_o2_par_out_h2 = weight[7]
    # o2节点 对 b2 的偏导数
    par_net_o2_par_b2 = 1

    # h1输出 对 h1节点 的偏导数
    par_out_h1_par_net_h1 = dersig(out_h1)
    # h2输出 对 h2节点 的偏导数
    par_out_h2_par_net_h2 = dersig(out_h2)

    # h1节点 对 w1 的偏导数
    par_net_h1_par_w1 = i1
    # h1节点 对 w2 的偏导数
    par_net_h1_par_w2 = i2
    # h1节点 对 b1 的偏导数
    par_net_h1_par_b1 = 1
    # h2节点 对 w3 的偏导数
    par_net_h2_par_w3 = i1
    # h2节点 对 w4 的偏导数
    par_net_h2_par_w4 = i2
    # h2节点 对 b1 的偏导数
    par_net_h2_par_b1 = 1

    # 总误差 对 w5 的偏导数
    par_e_total_par_w5 = par_e_total_par_out_o1 * par_out_o1_par_net_o1 * par_net_o1_par_w5
    # 总误差 对 w6 的偏导数
    par_e_total_par_w6 = par_e_total_par_out_o1 * par_out_o1_par_net_o1 * par_net_o1_par_w6
    # 总误差 对 w7 的偏导数
    par_e_total_par_w7 = par_e_total_par_out_o2 * par_out_o2_par_net_o2 * par_net_o2_par_w7
    # 总误差 对 w8 的偏导数
    par_e_total_par_w8 = par_e_total_par_out_o2 * par_out_o2_par_net_o2 * par_net_o2_par_w8

    # 总误差 对 h1节点 的偏导数
    par_e_total_par_net_h1 = \
        par_e_total_par_out_o1 * par_out_o1_par_net_o1 * par_net_o1_par_out_h1 * par_out_h1_par_net_h1 +\
        par_e_total_par_out_o2 * par_out_o2_par_net_o2 * par_net_o2_par_out_h1 * par_out_o1_par_net_o1
    # 总误差 对 h2节点 的偏导数
    par_e_total_par_net_h2 = \
        par_e_total_par_out_o1 * par_out_o1_par_net_o1 * par_net_o1_par_out_h2 * par_out_h2_par_net_h2 +\
        par_e_total_par_out_o2 * par_out_o2_par_net_o2 * par_net_o2_par_out_h2 * par_out_o2_par_net_o2

    # 总误差 对 b2 的偏导数
    par_e_total_par_b2 = \
        par_e_total_par_out_o1 * par_out_o1_par_net_o1 * par_net_o1_par_b2 +\
        par_e_total_par_out_o2 * par_out_o2_par_net_o2 * par_net_o2_par_b2
    # 总误差 对 b1 的偏导数
    par_e_total_par_b1 = (par_e_total_par_net_h1 + par_e_total_par_net_h2) * par_net_h2_par_b1

    # 总误差 对 w1 的偏导数
    par_e_total_par_w1 = par_e_total_par_net_h1 * par_net_h1_par_w1
    # 总误差 对 w2 的偏导数
    par_e_total_par_w2 = par_e_total_par_net_h1 * par_net_h1_par_w2
    # 总误差 对 w3 的偏导数
    par_e_total_par_w3 = par_e_total_par_net_h2 * par_net_h2_par_w3
    # 总误差 对 w4 的偏导数
    par_e_total_par_w4 = par_e_total_par_net_h2 * par_net_h2_par_w4

    # 权重更新
    weight[0] -= alpha * par_e_total_par_w1
    weight[1] -= alpha * par_e_total_par_w2
    weight[2] -= alpha * par_e_total_par_w3
    weight[3] -= alpha * par_e_total_par_w4
    weight[4] -= alpha * par_e_total_par_w5
    weight[5] -= alpha * par_e_total_par_w6
    weight[6] -= alpha * par_e_total_par_w7
    weight[7] -= alpha * par_e_total_par_w8
    bias[0] -= alpha * par_e_total_par_b1
    bias[1] -= alpha * par_e_total_par_b2

# 输出权重
print('final :', '\n', 'weight:', weight, '\n', 'bias:', bias)
print('y1:', out_o1, '\n', 'y2:', out_o2)
print('loss', e_total)
