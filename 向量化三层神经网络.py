# !/usr/bin/python3
# -*- coding:utf-8 -*-

'''

Title :向量化三层神经网络

Author : Ining

Reference： 
  https://blog.csdn.net/tudaodiaozhale/article/details/78632931

Update date :
  2018.6.6 ：
  + 完成向量化 : 相加的地方要考虑 dot乘法
  + 完成权重、偏置的初始化 ： 多了解一些其他的初始化方法！！
  + 找到了速度慢的原因 ： 发现输出很占用时间！ / 输出 a2 与 输出 a[0]，a[1] 时间上差很多，不知道为什么？！
  + 将正向传播过程封装 ： 本打算用来测试，才发现手动设置的训练集没法设置测试集
  + 将反向传播过程封装 ： 可以进一步拓展
  + 将 sigmoid函数的导数 与 sigmoid函数合二为一 ： 自我感觉设计的很巧妙
  2018.6.5 ：
  + 完全面向过程
  - 无法设置测试集 ：代入真实数据的时候再设置
  - 训练样本小 ：以后代入数量多一些的真实数据
  - 未向量化 ：无法拓展
  - 手动初始化 ：以后加入随机初始化

'''

import numpy as np
import time

start = time.time()

# 超参数
alpha = 0.5
epoches = 930000

# 输入输入
x = np.array([0.05, 0.10])
y = np.array([0.01, 0.99])
# 权重偏置初始化
w1 = 0.01 * np.random.randn(2, 2)
w2 = 0.01 * np.random.randn(2, 2)
b1, b2 = 0.0, 0.0

def forward(x, w1, w2, b1, b2):

    z1 = np.dot(w1, x) + b1
    a1 = sigmoid(z1)
    z2 = np.dot(w2, a1) + b2
    a2 = sigmoid(z2)

    return z1, a1, z2, a2

def backprop(x, y, a1, a2):

    e_a2 = - (y - a2)
    a2_z2 = np.multiply(a2, (1 - a2))
    z2_w2 = a1
    z2_a1 = w2
    a1_z1 = np.multiply(a1, (1 - a1))
    z1_w1 = x

    e_z2 = np.multiply(e_a2, a2_z2)
    e_w2 = np.multiply(e_z2, z2_w2)
    e_a1 = np.dot(z2_a1, e_z2)
    e_z1 = np.multiply(e_a1, a1_z1)
    e_w1 = np.multiply(e_z1, z1_w1)

    return e_w1, e_w2

# sigmoid函数
def sigmoid(z, der = False):

    if der == True:
        return np.multiply(z, (1 - z))

    return 1.0 / (1.0 + np.exp(-z))

for epoch in range(epoches):

    '''前向传播'''

    z1, a1, z2, a2 = forward(x, w1, w2, b1, b2)

    print(epoch + 1, a2[0], a2[1])

    '''反向传播'''

    e_w1, e_w2 = backprop(x, y, a1, a2)

    '''更新权重'''

    for i in range(len(w2)):
        w2[i] -= np.multiply(alpha, e_w2[i])
    for j in range(len(w1)):
        w1[j] -= np.multiply(alpha, e_w1[j])

end = time.time()
print(end - start)
