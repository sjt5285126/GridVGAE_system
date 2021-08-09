# -*- coding: utf-8 -*-
import numpy as np
from numpy.random import rand
from sklearn import linear_model
from Ising_Metropolis import nn
import wolff
import Ising_Metropolis
import matplotlib.pyplot as plt


# 文献中使用的例子L = 10,20,40 我们可以适当进行缩小

# For K = 0, this model reduces to the standard Ising model which can be simulated efficiently by the Wolff method.
# 对于近邻，先考虑次近邻，再逐渐多进行考虑

def heff(grid, J=1, K=0.2):
    n = len(grid)
    label = []
    x = []
    for i in range(n):
        for j in range(n):
            a = i
            b = j
            s = grid[i][j]
            E = np.array(nn(grid, (i, j), 1, n)).sum()
            P = grid[(a - 1) % n][b] * grid[(a - 1) % n][(b - 1) % n] * grid[a][(b - 1) % n] + \
                grid[(a + 1) % n][b] * grid[(a + 1) % n][(b + 1) % n] * grid[a][(b + 1) % n] + \
                grid[(a - 1) % n][b] * grid[(a - 1) % n][(b + 1) % n] * grid[a][(b + 1) % n] + \
                grid[(a + 1) % n][b] * grid[(a + 1) % n][(b - 1) % n] * grid[a][(b - 1) % n]
            Heff = -1 * J * s * E - K * s * P
            # 计算出整个构型的H
            label.append(Heff)
            temp = []
            for k in range(1, 4):  # 对range(a,b)中 b的修改会改变计算近邻的数量
                e = np.array(nn(grid, (i, j), k, n)).sum()
                temp.append(e * s)
            x.append(temp)
    x = np.array(x)
    # print(x)
    # 列相加
    x = np.sum(x, axis=0)
    # print(x)
    # 计算整个图形的 heff
    return x, sum(label)


#  机器学习-线性回归
def linearRegression(x, y):
    # 进行线性回归
    x = np.array(x)
    y = np.array(y)
    reg = linear_model.LinearRegression()
    reg.fit(x, y)

    # this.coef_ 系数
    # this.intercept_ 截距
    return reg


# 设计考虑外场因素的Metropolis算法
def metropolis_flip(grid, beta, J=1, K=0.2):
    n = len(grid)
    for i in range(n):
        for j in range(n):
            a = np.random.randint(0, n)
            b = np.random.randint(0, n)
            s = grid[a][b]
            E = np.array(nn(grid, (a, b), 1, n)).sum()
            P = grid[(a - 1) % n][b] * grid[(a - 1) % n][(b - 1) % n] * grid[a][(b - 1) % n] + \
                grid[(a + 1) % n][b] * grid[(a + 1) % n][(b + 1) % n] * grid[a][(b + 1) % n] + \
                grid[(a - 1) % n][b] * grid[(a - 1) % n][(b + 1) % n] * grid[a][(b + 1) % n] + \
                grid[(a + 1) % n][b] * grid[(a + 1) % n][(b - 1) % n] * grid[a][(b - 1) % n]
            cost = 2 * (J * s * E + K * s * P)
            if cost < 0:
                s *= -1
            elif rand() < np.exp(-cost * beta):
                s *= -1
            grid[a][b] = s
    return grid


def wolff_flip(grid, beta, reg, J=1, K=0.2):
    x_a, E_a = heff(grid)
    # print(x_a,E_a)
    a = []
    a.append(x_a)
    # print(reg.coef_)
    Eeff_a = reg.predict(a)
    # 创建一个副本
    temp = grid.copy()
    Ising_Metropolis.flipping_wolff(temp, beta)
    x_b, E_b = heff(temp)
    b = []
    b.append(x_b)
    Eeff_b = reg.predict(b)
    if rand() < np.exp(-1 * beta * ((E_b - Eeff_b) - (E_a - Eeff_a))):
        grid = temp
    return grid


def wolff_1(grid,location:tuple, beta, reg):
    p = []
    j_n = len(reg.coef_) # total numbers of j
    p = -1*reg.coef_
    p = 1 - np.exp(-2*beta*p)
    stack = []
    x = location[0]
    y = location[1]
    s = grid[x][y]
    cluster = []
    stack.append([s,(x,y)])
    cluster.append((x,y))
    while stack:
        # first 1nn
        temp = stack.pop()







'''
def wollf(eqsteps,mcsteps,N,reg):
    T = np.linspace(0.1, 5, 10)
    n1 = 1 / (N * N * mcsteps)
    M = []
    j = 0
    for t in T:
        m = 0
        config = Ising_Metropolis.init_state(N)
        for i in range(eqsteps):
            wolff_flip(config, 1 / t,reg)
        for i in range(mcsteps):
            wolff_flip(config, 1 / t,reg)
            m += abs(np.sum(config))
            j += 1
            if j%10 == 0:
                print("已完成第%d步模拟" % j)
        M.append(m * n1)


    plt.xlabel('T')
    plt.ylabel('m')
    plt.title('wolff')
    plt.plot(T, abs(np.array(M)), "ob")
    plt.show()

'''


# 仿制文献图3
def wollf(eqsteps, mcsteps, N, reg, T):
    n1 = 1 / (N * N * mcsteps)

    return


def metropolis(eqsteps, mcsteps, N):
    T = np.linspace(0.1, 5, 100)
    n1 = 1 / (N * N * mcsteps)
    M = []
    j = 0
    for t in T:
        m = 0
        config = Ising_Metropolis.init_state(N)
        # 先进行平衡
        for i in range(eqsteps):
            metropolis_flip(config, 1 / t)
        # 之后进行计算
        for i in range(mcsteps):
            metropolis_flip(config, 1 / t)
            m += abs(np.sum(config))
            j += 1
            if j % 10 == 0:
                print("已完成第%d步模拟" % j)
        M.append(m * n1)
    plt.xlabel('T')
    plt.ylabel('m')
    plt.title('metropolis')
    plt.plot(T, abs(np.array(M)), "ob")
    plt.show()


if __name__ == '__main__':
    L = 4
    T = 2.493
    test_x = []
    test_y = []
    esteps = 500  # 选择平衡步数
    traningdata = []
    label_x = []
    label_y = []
    testdata = []
    testdata_x = []
    testdata_y = []
    init_times = 5000  # 选择用来拟合的构型
    for i in range(init_times):  # 先生成100个图片进行MC模拟
        config = Ising_Metropolis.init_state(L)
        # 进行热浴平衡
        for estep in range(esteps):
            # (1) 使用local update进行MC模拟实验,生成大量构型,作为训练数据

            # SLMC的原始哈密顿量是根据论文中考虑斑点内自旋影响的新的公式进行计算

            metropolis_flip(config, 1 / T)
        # 计算哈密顿量
        # print(config)
        # 得到哈密顿量
        x, y = heff(config)
        test_x.append(x)
        test_y.append(y)
    # (2)使用得到的大量哈密顿量进行机器学习
    reg = linearRegression(test_x, test_y)
    for i in range(100):  # 对模型进行误差分析
        config = Ising_Metropolis.init_state(L)
        traningdata.append(config.copy())
        label_x.append(heff(config)[0])
        label_y.append(heff(config)[1])
        for estep in range(esteps):
            metropolis_flip(config, 1 / T)
        testdata.append(config)
        testdata_x.append(heff(config)[0])
        testdata_y.append(heff(config)[1])
    error = reg.predict(testdata_x)
    print('init_config:', init_times, ' esteps:', esteps)
    print('balanced error:', np.mean(error - testdata_y))
    error = reg.predict(label_x)
    print('original configuration error:', np.mean(error - label_y))
    # 最后的结果 j2 j3 足够小，又决定只使用j1进行线性回归
    # print(reg.coef_)
    # wollf(1000,1000,8,reg)

    # metropolis(1000,1000,8) #计算斑点自旋后的local update算法

    # Ising_Metropolis.metropolis(1000,1000,8)
