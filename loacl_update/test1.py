import numpy as np
import pandas as pd
from numpy.random import rand
from sklearn import linear_model
import matplotlib.pyplot as plt

N = 3

def init_state(N):
    state = 2 * np.random.randint(2, size=(N, N)) - 1  # 让矩阵中的值 只有+1或-1
    return state

grid = init_state(N)

def flipping(grid, beta):
    J = 1
    K = 1
    n = len(grid)
    # 存在可能，并不是一个N*N的矩阵，可能是N*M的矩阵？
    for i in range(0, n):
        for j in range(0, n):
            # 随机进行反转 相当于同一个点可能翻转多次？
            a = np.random.randint(0, n)
            b = np.random.randint(0, n)
            s = grid[a][b]
            E = grid[(a + 1) % n][b] + grid[a][(b + 1) % n] + grid[next_a][b] + grid[a][next_b]
            P = grid[(a - 1) % n][b] * grid[(a - 1) % n][(b - 1) % n] * grid[a][(b - 1) % n] + \
                grid[(a + 1) % n][b] * grid[(a + 1) % n][(b + 1) % n] * grid[a][(b + 1) % n] + \
                grid[(a - 1) % n][b] * grid[(a - 1) % n][(b + 1) % n] * grid[a][(b + 1) % n] + \
                grid[(a + 1) % n][b] * grid[(a + 1) % n][(b - 1) % n] * grid[a][(b - 1) % n]
            cost = 2 * s * (J * E + K * P)  #后面会用到的H
            # 如果能量降低接受反转
            if cost < 0:
                s *= -1
            elif rand() < np.exp(-cost * beta):
                s *= -1
            grid[a][b] = s
    #这一步的输出是得到一个local update的构型，生成多个构型变成训练子集
    return grid


def linearRegression():
    '''
    首先要确定标签，即进行回归所用的 X,Y标签
    x为构型中的点与k近邻的相乘合
    y为论文中用eq(1)得到的哈密顿量
    最后拟合一个多项式线性回归方程
    '''
    J = 1
    K = 1
    n = N
    x = []
    y = []
    for i in range(0,N):
        for j in range(0,N):
            s = grid[i][j]
            E = np.array(nn((i,j),1)).sum()
            P = grid[(i - 1) % n][j] * grid[(i - 1) % n][(j - 1) % n] * grid[i][(j - 1) % n] + \
                grid[(i + 1) % n][j] * grid[(i + 1) % n][(j + 1) % n] * grid[i][(j + 1) % n] + \
                grid[(i - 1) % n][j] * grid[(i - 1) % n][(j + 1) % n] * grid[i][(j + 1) % n] + \
                grid[(i + 1) % n][j] * grid[(i + 1) % n][(j - 1) % n] * grid[i][(j - 1) % n]
            H = 2 * s * (J * E + K * P)
            y.append(H)
            temp = [] #存放一个x的特征
            for k in range(1,N-1):
                e = np.array(nn((i,j),k)).sum()
                temp.append(e*s)
            x.append(temp)
    '''
    class sklearn.linear_model.LinearRegression (fit_intercept=True,
    normalize=False, copy_X=True, n_jobs=None)
    fit_intercept 默认为True 代表是否计算截距，默认为计算，在该题中，截距为E0
    normalize 是否进行归一化
    copy_x 是否在原矩阵的copy上做，这样可以避免影响原矩阵
    吧、
    n_jobs 用在多标签的线性回归上
    '''
    x = np.array(x)
    y = np.array(y)
    print(x.shape)
    print(y.shape)
    reg = linear_model.LinearRegression()
    reg.fit(X=x,y=y)
    y_predict = reg.predict(x)
    E_0 = 0




#简易版 可以修改增加大量的邻近节点函数
def nn_1(location: tuple):
    nn = []
    x = location[0]
    y = location[1]
    nn.append(grid[(x - 1) % N][y])
    nn.append(grid[x][(y + 1) % N])
    nn.append(grid[(x + 1) % N][y])
    nn.append(grid[x][(y - 1) % N])
    return nn


def nn_2(location: tuple):
    nn = []
    x = location[0]
    y = location[1]
    nn.append(grid[(x - 1) % N][(y + 1) % N])
    nn.append(grid[(x + 1) % N][(y + 1) % N])
    nn.append(grid[(x - 1) % N][(y - 1) % N])
    nn.append(grid[(x + 1) % N][(y - 1) % N])
    return nn

def nn_3(location:tuple):
    nn = []
    x = location[0]
    y = location[1]
    nn.append(grid[(x - 2) % N][y])
    nn.append(grid[x][(y + 2) % N])
    nn.append(grid[(x + 2) % N][y])
    nn.append(grid[x][(y - 2) % N])

#任意计算k近邻
def nn(location:tuple,k:int): #k代表k-近邻
    nn = []
    x = location[0]
    y = location[1]
    t = k-1
    if k%2 == 0:
        nn.append(grid[(x - t) % N][(y + t) % N])
        nn.append(grid[(x + t) % N][(y + t) % N])
        nn.append(grid[(x - t) % N][(y - t) % N])
        nn.append(grid[(x + t) % N][(y - t) % N])
    else:
        nn.append(grid[(x - t) % N][y])
        nn.append(grid[x][(y + t) % N])
        nn.append(grid[(x + t) % N][y])
        nn.append(grid[x][(y - t) % N])
    return nn

if __name__ == '__main__':
    linearRegression()
