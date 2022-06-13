import numpy as np
import pandas as pd
import h5py
import sys
from sys import argv
import os
import re

'''
需要画图的数据有：
1. 16相变温度下的数据 graphConv GConv GATConv molGAN
2. 32相变温度下的数据 graphConv GConv GATConv molGAN
3. 16 3温度下的数据 graphConv GConv GATConv molGAN
4. 32 3温度下的数据 graphConv GConv GATConv molGAN
5. 16mix32 相变温度下的数据 graphConv GConv GATConv molGAN
6. 16mix32 3温度下的数据 graphConv GConv GATConv molGAN
'''


def explore_old(data):
    '''
    老版数据 data 并不包含温度信息，所以直接读取想要的属性就行
    :param data:
    :return:
    '''


def explore(dataPath):
    '''

    :param data: hdf5文件 将hdf5文件中的信息读取
    :return: csv/dat文件
    '''
    data = h5py.File(dataPath, 'r')
    df = pd.DataFrame()
    for key in data.keys():
        df[str(key)] = data[str(key)]
    df.to_csv('{}.dat'.format(dataPath), sep=' ')


def explore_dir(datadir):
    for item in os.scandir(datadir):
        if item.is_file() and re.search('.*\.hdf5', item.path):
            explore(item.path)


explore_dir('.')
