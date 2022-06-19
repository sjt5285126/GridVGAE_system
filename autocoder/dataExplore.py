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


def explore(dataPath):
    data = h5py.File(dataPath, 'r')
    f = h5py.File('{}Features.hdf5'.format(dataPath[:-5]), 'w')
    for key in data.keys():
        if re.search('T=(.*)_(.*)', str(key)):
            f.create_dataset(str(key), data=data[key])
    f.close()


explore('data/Ising/32_T_PTP.hdf5')
