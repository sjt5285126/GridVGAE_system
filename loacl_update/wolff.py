# -*- coding: utf-8 -*-
import numpy as np
import math
from numpy.random import rand
import random


def flip(grid,location,beta):
    #print(grid)
    p = 1-np.exp(-2*beta)
    stack = []
    x = location[0]
    y = location[1]
    s = grid[x][y]
    cluster = []
    stack.append([s,(x,y)])
    cluster.append((x,y))
    #print("begin")
    while stack:
        temp = stack.pop()
        neigh = neighbor(grid,temp[1])
        #print("temp:",temp)
        #print("neigh:",neigh)
        for em in neigh:
            if em[1] not in cluster and em[0] == s and rand()<p:
                #该处是先将所有的近邻,并且状态尚未翻转的元素
                #print("em:",em)
                #print(em[1])
                stack.append(em)
                cluster.append(em[1])
        '''
        x = temp[1][0]
        y = temp[1][1]
        grid[x][y] = -1*grid[x][y]
        '''
    #print(cluster)
    for i in cluster:
        x = i[0]
        y = i[1]
        grid[x][y] *= -1

    #print(grid)
    #print("complete")
        



def neighbor(grid,location):
    N = len(grid)
    nn = []
    x = location[0]
    y = location[1]
    a1 = (x-1)%N
    a2 = (x+1)%N
    b1 = (y-1)%N
    b2 = (y+1)%N
    nn.append([grid[a1][y],(a1,y)])
    nn.append([grid[x][b1],(x,b1)])
    nn.append([grid[a2][y],(a2,y)])
    nn.append([grid[x][b2],(x,b2)])
    return nn

def nn(grid, location, k, N):
    nn = []
    x = location[0]
    y = location[1]
    t = math.ceil(k/2)
    a1 = (x-t)%N
    a2 = (x+t)%N
    b1 = (y-t)%N
    b2 = (y+t)%N
    if k % 2 == 0:
        nn.append([grid[a1][b2],(a1,b2)])
        nn.append([grid[a2][b2],(a2,b2)])
        nn.append([grid[a1][b1],(a1,b1)])
        nn.append([grid[a2][b1],(a2,b1)])
    else:
        nn.append([grid[a1][y],(a1,y)])
        nn.append([grid[x][b2],(x,b2)])
        nn.append([grid[a2][y],(a2,y)])
        nn.append([grid[x][b1],(x,b1)])
    return nn


