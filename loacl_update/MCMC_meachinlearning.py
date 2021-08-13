# -*- coding: utf-8 -*-
import numpy as np
from numpy.random import rand
from sklearn import linear_model
from Ising_Metropolis import nn
import wolff
import Ising_Metropolis
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd

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
                temp.append(e * s) #表示 Si*Sj
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

# 算法的应用
def wolff_flip(grid, beta, reg, J=1, K=0.2):
    # 在有效模型中,我们要对wolff的每一次翻转进行有效性判断
    steps = 1
    for i in range(steps):
        Eeff = [] #存放通过有效模型计算的哈密顿量
        temp = grid.copy() #保存未翻转的构型
        nn_a,E_A = heff(temp) #计算构型的近邻关系以及哈密顿量
        Eeff.append(nn_a) #存放近邻关系以便于 使用有效模型计算哈密顿量

        # wollf算法
        a = np.random.randint(0,len(grid))
        b = np.random.randint(0,len(grid))
        wolff_1(grid,(a,b),beta,reg)

        #计算反转后的信息
        nn_b,E_B = heff(grid)
        Eeff.append(nn_b)
        Eeff = reg.predict(Eeff)
        Eeff_A = Eeff[0]
        Eeff_B = Eeff[1]
        if rand() > np.exp(-1*beta*((E_B-Eeff_B)-(E_A-Eeff_A))):
            grid = temp
    return grid


# 适用于多近邻的wolff算法
def wolff_1(grid,location:tuple, beta, reg):
    j_n = len(reg.coef_) # total numbers of j
    j = -1*reg.coef_
    p = 1 - np.exp(-2*beta*j)
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
        #  依次连接最近邻格点,2近邻格点,3近邻格点
        for i in range(j_n):
            neigh = wolff.nn(grid,temp[1],i+1,len(grid))
            for em in neigh:
                if em[1] not in cluster and em[0] == s and rand() < p[i]:
                    stack.append(em)
                    cluster.append(em[1])
    for i in cluster:
        x = i[0]
        y = i[1]
        grid[x][y] = -1*grid[x][y]
    return grid








def wolff_test(esteps,mcsteps,reg,N):
    T = np.linspace(0.1,5,10)
    n1 = 1 / (N*N*mcsteps)
    M = []
    j = 0
    for t in T:
        m = 0
        config = Ising_Metropolis.init_state(N)
        for i in range(esteps):
            wolff_flip(config,1/t,reg)
        for i in range(mcsteps):
            wolff_flip(config,1/t,reg)
            m += abs(np.sum(config))
            j += 1
            if j%10 == 0:
                print('已完成第%d步模拟')
        M.append(m*n1)
    plt.xlabel('T')
    plt.ylabel('m')
    plt.title('wollf')
    plt.plot(T,abs(np.array(M)),'ob')
    plt.show()



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

def test(reg,file):
    '''
    对模型进行误差分析
    '''
    for i in range(100):
        config = Ising_Metropolis.init_state(L) #生成构型
        traningdata.append(config.copy())  #将构型的初始量存放在训练集中
        label_x.append(heff(config)[0]) #存放初始的近邻关系
        label_y.append(heff(config)[1]) #存放初始的哈密顿量
        for estep in range(esteps):
            metropolis_flip(config, 1 / T) #经过local update算法进行平衡
        testdata.append(config) #放入到测试集当中
        testdata_x.append(heff(config)[0]) #将紧邻关系存放到测试集中
        # testdata_y 存放的是用有四体相互作用计算的哈密顿量
        testdata_y.append(heff(config)[1]) #将平衡后的哈密顿量存放到测试集中
    error = reg.predict(testdata_x) #使用有效模型计算平衡后的哈密顿量
    file.write('init_config: {} esteps: {}'.format(init_times,esteps))
    print('init_config:', init_times, ' esteps:', esteps)
    file.write('balanced error:{}'.format(np.mean(error - testdata_y)))
    print('balanced error:', np.mean(error - testdata_y))
    error = reg.predict(label_x) #使用模型计算初始的哈密顿量
    # 计算误差应用平衡后的哈密顿量来计算误差
    #  公式是用来计算最终的哈密顿量 所以理论上应用平衡后的哈密顿量计算
    file.write('original configuration error(originala h):{}'.format(np.mean(error-label_y)))
    print('original configuration error(originala h):',np.mean(error-label_y))
    file.write('original configuration error(balance h):{}'.format(np.mean(error - testdata_y)))
    print('original configuration error(balance h):', np.mean(error - testdata_y))
    # 最后的结果 j2 j3 足够小，又决定只使用j1进行线性回归
    # print(reg.coef_)
    # wollf(1000,1000,8,reg)

    # metropolis(1000,1000,8) #计算斑点自旋后的local update算法

    # Ising_Metropolis.metropolis(1000,1000,8)


if __name__ == '__main__':
    L = 4
    T = 2.493
    beta = 1 / T
    test_x = []
    test_y = []
    esteps = 500  # 选择平衡步数
    traningdata = []
    label_x = []
    label_y = []
    testdata = []
    testdata_x = []
    testdata_y = []
    Jarray = []
    init_times = 1000  # 选择用来拟合的构型
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

    file = open('testresult.txt','a+')
    print('wollf更新参与迭代之前')
    test(reg,file)
    '''
    #下面进行第三步 使用wollf全局更新算法来 对构型进行翻转
    '''
    new_test = []
    # 使用wollf处理与初始状态相同的构型 来迭代模型
    '''
    对模型的迭代可以用j大小的变化来直观的感受模型的收敛
    新的图形中 横坐标代表迭代次数
    纵坐标代表 j的大小
    将j_1 j_2 j_3 同时放在一个图像中动态的进行表示
    '''
    count = 1
    times = []
    Jclass = []
    while count<10:
        times.append(count)
        Jarray.extend(reg.coef_)
        Jclass.extend([1,2,3])
        for i in range(init_times):
            config = Ising_Metropolis.init_state(L)
            wolff_flip(config,beta,reg)
            new_test.append(config.copy()) #得到wollf更新后的测试集
        # test_x,test_y = [],[]
        for tests in new_test:
            x,y = heff(tests)
            # 在测试集的选取上面可以选择叠加到原来的测试集上 和 重新用该测试集构建
            test_x.append(x)
            test_y.append(y)
        reg = linearRegression(test_x, test_y)
        print('第{}次迭代后的模型误差'.format(count))
        test(reg,file)
        count += 1
    df = pd.DataFrame({'times':times,'value':Jarray,'class':Jclass})
    sns.lineplot(x='time',y='value',hue='class',data=df,ci=None,markers=True,dashes=False)
    plt.show()
    file.close()





