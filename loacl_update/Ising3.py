import time
import numpy as np
from numpy.random import rand
import matplotlib.pyplot as plt


def init_state(N):
    ''' generates a random spin configuration for initial condition'''
    # np.random.randint(2, size=(N,N)) 表示产生N*N随机数,返回为列表
    state = 2 * np.random.randint(2, size=(N, N)) - 1
    return state


# 线性拟合的时候 会计算 次近邻 3近邻  难点在次
def flipping(grid, beta):
    '''Monte Carlo move using Metropolis algorithm '''
    N = len(grid)
    for i in range(N):
        for j in range(N):
            # 产生0到N-1的随机数
            a = np.random.randint(0, N)
            b = np.random.randint(0, N)
            s = grid[a][b]
            E = grid[(a + 1) % N][b] + grid[a][(b + 1) % N] + grid[(a - 1) % N][b] + grid[a][(b - 1) % N]
            cost = 2 * s * E
            # 如果能量降低接受翻转
            if cost < 0:
                s *= -1
            # 在0到1产生随机数，如果概率小于exp(-E/(kT))翻转
            elif rand() < np.exp(-cost * beta):
                s *= -1
            grid[a][b] = s
    return grid


def calculate_energy(grid):
    '''Energy of a given configuration'''
    energy = 0
    N = len(grid)
    for i in range(N):
        for j in range(N):
            S = grid[i][j]
            E = grid[(i + 1) % N][j] + grid[i][(j + 1) % N] + grid[(i - 1) % N][j] + grid[i][(j - 1) % N]
            # 负号来自哈密顿量
            energy += -E * S
    # 最近邻4个格点
    return energy / 4


def calculate_magnetic(grid):
    '''Magnetization of a given configuration'''
    mag = np.sum(grid)
    return mag


nt = 2 ** 8  # 温度取点数量
N = 8  # 点阵尺寸, N x N
eqSteps = 50  # MC方法平衡步数
mcSteps = 50  # MC方法计算步数

Energy = []  # 内能
Magnetization = []  # 磁矩
SpecificHeat = []  # 比热容/温度的涨落
Susceptibility = []  # 磁化率/磁矩的涨落

T = np.linspace(1.2, 3.8, nt)
T = list(T)

n1 = 1 / (mcSteps * N * N)
n2 = 1 / mcSteps ** 2 * N * N

time_start = time.time()
j = 0
for t in T:
    # 初始构型
    E = 0  # 每一温度下的能量
    M = 0  # 每一温度下的磁矩
    cv = 0  # 每一温度下的热容
    k = 0  # 每一温度下的磁化率
    config = init_state(N)
    # 热浴，达到平衡态
    for i in range(eqSteps):
        flipping(config, 1 / t)
    # 抽样计算
    for i in range(mcSteps):
        flipping(config, 1 / t)
        e = calculate_energy(config)
        m = calculate_magnetic(config)
        E += e
        M += m
        cv += e * e
        k += m * m
    Cv = (n1 * cv - n2 * E * E) / (t * t)
    K = (n1 * k - n2 * M * M) / t
    Energy.append(E * n1)
    Magnetization.append(M * n1)
    SpecificHeat.append(Cv)
    Susceptibility.append(K / (mcSteps ** 2 * N * N))
    j += 1
    if j % 10 == 0:
        print("已完成第%d步模拟" % j)
time_end = time.time()
print('totally cost', time_end - time_start)

Magnetization = np.array(Magnetization)

f = plt.figure(figsize=(18, 10));  # plot the calculated values

sp = f.add_subplot(2, 2, 1);
plt.plot(T, Energy, 'o', color="red");
plt.xlabel("Temperature (T)", fontsize=20);
plt.ylabel("Energy ", fontsize=20);

sp = f.add_subplot(2, 2, 2);
plt.plot(T, abs(Magnetization), 'o', color="blue");
plt.xlabel("Temperature (T)", fontsize=20);
plt.ylabel("Magnetization ", fontsize=20);


sp = f.add_subplot(2, 2, 3);
plt.plot(T, SpecificHeat, 'o', color="red");
plt.xlabel("Temperature (T)", fontsize=20);
plt.ylabel("Specific Heat ", fontsize=20);

sp = f.add_subplot(2, 2, 4);
plt.plot(T, Susceptibility, 'o', color="blue");
plt.xlabel("Temperature (T)", fontsize=20);
plt.ylabel("Susceptibility", fontsize=20);
plt.show()