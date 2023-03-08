import matplotlib.pyplot as plt
import numpy as np
import copy
import math

from numpy import pi


def getEnergy_XY(i, j, S, size, angle=None):
    width = size
    height = size
    # print('矩阵的宽和高是',width,height)
    top_i = i - 1 if i > 0 else width - 1
    bottom_i = i + 1 if i < (width - 1) else 0
    left_j = j - 1 if j > 0 else height - 1
    right_j = j + 1 if j < (height - 1) else 0
    enviroment = [[top_i, j], [bottom_i, j], [i, left_j], [i, right_j]]
    #  print(i,j,enviroment)
    # print(enviroment)
    energy = 0
    if angle == None:
        for num_i in range(0, 4, 1):
            energy += -(np.cos(S[i][j] - S[enviroment[num_i][0]][enviroment[num_i][1]]))
    else:
        for num_i in range(0, 4, 1):
            energy += -(np.cos(angle - S[enviroment[num_i][0]][enviroment[num_i][1]]))
    return energy


# 画成箭头图表示出现
def plot(S, ax):
    X, Y = np.meshgrid(np.arange(0, S.shape[0]), np.arange(0, S.shape[0]))
    U = np.cos(S)
    V = np.sin(S)

    # ax.set_xticklabels([''])
    # ax.set_yticklabels([''])
    # ax.tick_params(axis=u'both', which=u'both', length=0)
    # labels = range(len(S))
    # labels_grid = np.array(list(labels)) + 0.5

    # ax.set_yticks(labels_grid, minor=True)
    # ax.set_xticks(labels_grid, minor=True)
    # ax.yaxis.grid(True, which='minor')
    # ax.xaxis.grid(True, which='minor')
    ax.axis("off")
    Q = ax.quiver(X, Y, U, V, units='width', color='#4672C4', pivot='middle', scale=15)
    # new_Q = ax.quiver(X,Y,new_u,new_v,units='width',color='g',pivot='middle',scale=15)    #plt.pause(0.1)

    # qk = plt.quiverkey(Q, 0.3, 0.3, 1, r'$2 \frac{m}{s}$', labelpos='E',
    #         coordinates='figure')
    # plt.cla()
    '''
    plt.scatter(idx[0], idx[1], marker='+', color='k')
    plt.scatter(idx_[0], idx_[1], marker='_', color='k')
    plt.title('Arrows scale with plot width, not view')
    '''
    plt.savefig("xuan.svg")

def MetropolisHastings(S, size, T, numsOfItera):
    deltamax = 0.5
    delta = 2 * np.random.random() * np.pi  # 伸成一个2*pi的角度
    k = 1  # 玻尔兹曼常数
    for sdw in range(numsOfItera):
        # k2 = np.random.randint(0,S.shape[0]**2)
        i = np.random.randint(0, S.shape[0])
        j = np.random.randint(0, S.shape[0])
        # print('产生的随机位置是：',i,j)
        # time.sleep(0.1)
        for m in range(1):
            '''
            newAngle = 2 * delta - S[i][j]
            if newAngle > 2 * pi:
                newAngle -= 2 * pi
            elif newAngle < 0:
                newAngle += 2 * pi
            # print(delta)
            '''
            # delta = (2 * np.random.random() - 1) * deltamax * np.pi
            newAngle = (2 * delta - S[i][j]) % (2 * np.pi)
            energyBefore = getEnergy_XY(i=i, j=j, S=S, size=size)
            energyLater = getEnergy_XY(i=i, j=j, S=S, size=size, angle=newAngle)
            de = energyLater - energyBefore
            if de < 0.0:
                S[i][j] = newAngle
            elif np.random.uniform(0.0, 1.0) <= np.exp(-de / (k * T)):
                S[i][j] = newAngle
            # alpha = math.exp(-(energyLater - energyBefore) / (k * T))
            # print(alpha)
            # if alpha>=1:
            #  print('大于1的哦')
            # if alpha >= 1:
            #     S[i][j] = newAngle
            # elif np.random.uniform(0.0, 1.0) <= 1.0 * alpha:
            #     S[i][j] = newAngle

    return S


def getWeightValue(numsOfSample, sizeOfSample, temperature):
    # matplotlib 添加一个画板
    for i in range(numsOfSample):  # 产生个数
        fig, ax = plt.subplots(figsize=(6, 6))
        print('+++++++正在计算第%s个样本++++++++++' % i)
        # 使初始角度在0到2pi之间
        S = 2 * np.pi * np.random.rand(sizeOfSample, sizeOfSample)
        # 画出初始箭头
        # X, Y = np.meshgrid(np.arange(0, S.shape[0]), np.arange(0, S.shape[0]))

        # U = np.cos(S)
        # V = np.sin(S)
        # Q = ax.quiver(X, Y, U, V, units='width', color='g',pivot='middle',scale=12)
        # Q = ax.quiver(X-U/3, Y-V/3, U/3*2, V/3*2,units='width',color='g')
        # 初始全为 +自旋
        newS = np.array(copy.deepcopy(S))
        for nseeps in range(3000):
            newS = MetropolisHastings(newS, sizeOfSample, temperature, sizeOfSample ** 2)
            # if nseeps % 10 == 0:
            # plot(newS, newS_Ising)
        plot(newS, ax)
        # 关键在于bbox_inches = 'tight',pad_inches = 0，去掉空白区域
        # plt.savefig('jiantou{}.eps'.format(i),bbox_inches = 'tight',pad_inches = 0)
        plt.show()
        reshaped = np.reshape(newS, (1, sizeOfSample ** 2))
        if i == 0:
            s = copy.deepcopy(reshaped)
            continue
        else:
            s = np.row_stack((s, reshaped))
    return s


'''
出现涡旋的条件是低温。
'''
if __name__ == '__main__':
    res = getWeightValue(1, 16, 0.00002)
