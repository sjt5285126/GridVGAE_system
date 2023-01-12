# -*- codeing =utf-8 -*-
# @Time : 2021/9/19 11:00
# @Author : 丁利利
# @File : jiantou.py
# @SOftware: PyCharm
import random
import matplotlib.pyplot as plt
import numpy as np
import copy
import math

from numpy import pi

import time


'''这个就是MCMC模拟，用来模拟某个温度下XY Ising模型分布，
几点注意：
注意1，二维伊辛模型，我们用矩阵来模拟。
注意2，旋转的方向，我们用0到2pi表示吧
算法过程：
一，用一个对称的分布，高斯分布初始化矩阵
二，下面是循环
    （1）产生一个候选的自旋方向，为了连续，假设新产生的自旋方向变化最多比原来变化pi/2
    也就是旋转90度
    （2）计算两个概率，这里热统中的配分函数正比于概率，因此我们用配分函数Z的比值
    Z(变化后)/Z(变化前)=exp(-(E后-E前)/kT) ,这里k是玻尔兹曼常数，T是开尔文温度。令
    这个值是alpha
    (3)判断是都接受*********这个规则有进一步讨论*************
         （3_1)产生一个随机数u在【0,1】之间
         （3_2）如果u<=alhpa,接受候选的，改变此时的自旋状态
         (3_3)如果u>alpha，不接受候选的，不改变此时的自旋状态
inputdata: S :matrix 随机分布,假设已经产生
param: T 温度
       delta 最大的变化度数，默认是90度，也可以调整为其他
outputdata:S
'''

'''
增加Ising_model

'''

'''
修改xy模型的metropolis 为 Ising与XY混合的模型

区别在于:
1. 两种模型的metropolis模型交错进行
'''


def MetropolisHastings(S, S_Ising, T, numsOfItera):
    # deltamax = 0.5
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
            delta = (2 * np.random.random() - 1) * deltamax
            newAngle = S[i][j] + delta
            '''
            newAngle = 2 * delta - S[i][j]
            if newAngle > 2 * pi:
                newAngle -= 2 * pi
            elif newAngle < 0:
                newAngle += 2 * pi
            # print(delta)
            energyBefore = getEnergy_XY(i=i, j=j, S=S, S_Ising=S_Ising, angle=None)
            energyLater = getEnergy_XY(i, j, S=S, S_Ising=S_Ising, angle=newAngle)
            alpha = math.exp(-(energyLater - energyBefore) / (k * T))
            # print(alpha)
            # if alpha>=1:
            #  print('大于1的哦')
            if alpha >= 1:
                S[i][j] = newAngle
            elif np.random.uniform(0.0, 1.0) <= 1.0 * alpha:
                S[i][j] = newAngle
            # 开始 Ising model反转
            i = np.random.randint(0, S.shape[0])
            j = np.random.randint(0, S.shape[0])
            energyNew = getEnergy_Ising(i, j, S, S_Ising)
            alpha_Ising = math.exp(-energyNew / (k * T))
            if alpha_Ising >= 1:
                S_Ising[i][j] *= -1
            elif np.random.uniform(0.0, 1.0) <= 1.0 * alpha_Ising:
                S_Ising[i][j] *= -1
    return S, S_Ising


# 计算i,j位置 Ising模型的能量
def getEnergy_Ising(i, j, S, S_Ising):
    width = S.shape[0]
    height = S.shape[1]
    # print('矩阵的宽和高是',width,height)
    top_i = i - 1 if i > 0 else width - 1
    bottom_i = i + 1 if i < (width - 1) else 0
    left_j = j - 1 if j > 0 else height - 1
    right_j = j + 1 if j < (height - 1) else 0
    enviroment = [[top_i, j], [bottom_i, j], [i, left_j], [i, right_j]]
    energy = 0
    for num_i in range(0, 4, 1):
        energy += 2 * S_Ising[i][j] * S_Ising[enviroment[num_i][0]][enviroment[num_i][1]] * np.cos(
            S[i][j] - S[enviroment[num_i][0]][enviroment[num_i][1]])
    return energy


# 计算i,j位置的能量 = 与周围四个的相互能之和
def getEnergy_XY(i, j, S, S_Ising, angle=None):
    width = S.shape[0]
    height = S.shape[1]
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
        # print('angle==None')
        for num_i in range(0, 4, 1):
            energy += -(1 + S_Ising[i][j] * S_Ising[enviroment[num_i][0]][enviroment[num_i][1]]) * np.cos(
                S[i][j] - S[enviroment[num_i][0]][enviroment[num_i][1]])
    else:
        # print('Angle')
        for num_i in range(0, 4, 1):
            energy += -(1 + S_Ising[i][j] * S_Ising[enviroment[num_i][0]][enviroment[num_i][1]]) * np.cos(
                angle - S[enviroment[num_i][0]][enviroment[num_i][1]])
    return energy


# S=2*np.pi*np.random.rand(30,30)
# 计算整个格子的能量，为了求平均内能
def calculateAllEnergy(S, S_Ising):
    energy = 0  # 设定初值
    for i in range(len(S)):
        for j in range(len(S[0])):
            # 对每一个 格点的能量进行运算
            energy += getEnergy_XY(i, j, S, S_Ising)
    averageEnergy = energy / (len(S[0]) * len(S))
    return averageEnergy / 2


# print(S)
# for j in range(1000):
#   print(j)
# MetropolisHastings(S,10)
# 这个是输入样本的多少，格子的尺寸，温度。中间那个循环，是随机取迭代的过程
def getWeightValue(numsOfSample, sizeOfSample, temperature):
    # matplotlib 添加一个画板
    for i in range(numsOfSample):  # 产生个数
        fig, ax = plt.subplots(figsize=(6,6))
        print('+++++++正在计算第%s个样本++++++++++' % i)
        # 使初始角度在0到2pi之间
        S = 2 * np.pi * np.random.rand(sizeOfSample, sizeOfSample)
        # 画出初始箭头
        #X, Y = np.meshgrid(np.arange(0, S.shape[0]), np.arange(0, S.shape[0]))

        #U = np.cos(S)
        #V = np.sin(S)
        #Q = ax.quiver(X, Y, U, V, units='width', color='g',pivot='middle',scale=12)
        #Q = ax.quiver(X-U/3, Y-V/3, U/3*2, V/3*2,units='width',color='g')
        # 初始全为 +自旋
        S_Ising = np.ones((sizeOfSample, sizeOfSample))
        print(len(S_Ising))
        initialEnergy = calculateAllEnergy(S, S_Ising)
        print('系统的初始能量是:', initialEnergy)
        newS = np.array(copy.deepcopy(S))
        newS_Ising = np.array(copy.deepcopy(S_Ising))
        for nseeps in range(500):
            newS, newS_Ising = MetropolisHastings(newS, newS_Ising, temperature, sizeOfSample ** 2)
            #if nseeps % 10 == 0:
                #plot(newS, newS_Ising)
        aveEnergy = calculateAllEnergy(newS, newS_Ising)
        plot(newS,newS_Ising,ax)
        # 关键在于bbox_inches = 'tight',pad_inches = 0，去掉空白区域
        # plt.savefig('jiantou{}.eps'.format(i),bbox_inches = 'tight',pad_inches = 0)
        plt.show()
        print('系统的平均能量是:', aveEnergy)
        reshaped = np.reshape(newS, (1, sizeOfSample ** 2))
        reshaped_Ising = np.reshape(newS_Ising, (1, sizeOfSample ** 2))
        if i == 0:
            s = copy.deepcopy(reshaped)
            s_Ising = copy.deepcopy(reshaped_Ising)
            continue
        else:
            s = np.row_stack((s, reshaped))
            s_Ising = np.row_stack((s_Ising, reshaped_Ising))
    return s, s_Ising


# print(len(res))
# 画成箭头图表示出现
def plot(S, S_Ising,ax):


    idx = np.where(S_Ising == 1)
    idx_ = np.where(S_Ising != 1)
    X, Y = np.meshgrid(np.arange(0, S.shape[0]), np.arange(0, S.shape[0]))
    U = np.cos(S)
    V = np.sin(S)

    labels = range(len(S_Ising))

    ax.set_xticklabels([''])
    ax.set_yticklabels([''])
    ax.tick_params(axis=u'both', which=u'both', length=0)

    labels_grid = np.array(list(labels))+0.5

    ax.set_yticks(labels_grid,minor=True)
    ax.set_xticks(labels_grid,minor=True)
    ax.yaxis.grid(True,which='minor')
    ax.xaxis.grid(True,which='minor')

    ax.matshow(S_Ising,cmap='gray',vmin=-5, vmax=0)

    Q = ax.quiver(X, Y, U, V,units='width',color='r',pivot='middle',scale=15)
    new_s = S + 0.5*pi*S_Ising
    new_u = np.cos(new_s)
    new_v = np.sin(new_s)
    #new_Q = ax.quiver(X,Y,new_u,new_v,units='width',color='g',pivot='middle',scale=15)    #plt.pause(0.1)

    # qk = plt.quiverkey(Q, 0.3, 0.3, 1, r'$2 \frac{m}{s}$', labelpos='E',
    #         coordinates='figure')
    #plt.cla()
    '''
    plt.scatter(idx[0], idx[1], marker='+', color='k')
    plt.scatter(idx_[0], idx_[1], marker='_', color='k')
    plt.title('Arrows scale with plot width, not view')
    '''

# 运行getweightValue函数，中间已经把结果会成图了

def plot_Ising(S):
    '''
    使用散点图 遴选出所有值为正和值为负的点，使用散点图来进行画图
    :param S:
    :return:
    '''


'''
出现涡旋的条件是低温。
'''
if __name__ == '__main__':
    res = getWeightValue(1, 16,  0.01)
