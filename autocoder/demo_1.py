import pickle

import h5py
import torch
import numpy as np
from dataset import reshapeIsingHdf5, calculate, cacculateIsingXY
from InitIsingXY import genIsingXY

#
# batch_size = 5000
# # 填装测试数据,用来和生成模型的特征进行比对
# testData = h5py.File('data/Ising/32size.hdf5', 'r')
# print(testData.keys())
# for key in testData.keys():
#     testConfigs = testData[key][:5000]
# testData.close()
# print(testConfigs.shape)
# testConfigs = reshapeIsingHdf5(testConfigs, batch_size)
# print(testConfigs.shape)
# TotalM, TotalE, AvrM, AvrE = calculate(testConfigs)
#
# # 可以采用hdf5方式来存储数据的物理量，我们可以在生成数据的时候就将物理量全部计算出来，
# # 这样可以节省一部分空间
# f = h5py.File('32Features.hdf5','w')
# f['TotalM'] = TotalM
# f['TotalE'] = TotalE
# f['AvrM'] = AvrM
# f['AvrE'] = AvrE
# f.close()

# datafile = open('data/IsingXYGraph/dataIsing_IsingXYA0.32L16_0711.pkl', 'rb')
# data = pickle.load(datafile)
# datafile.close()
# print(data[0].x)

path = 'data/gouxingTestA0.32_1.dat'
size = 8

configs = genIsingXY(path, size)
print(configs.shape)

M_Ising, M_XY = cacculateIsingXY(configs)

file = h5py.File('lili/data/8sizeIsingXY_M.hdf5', 'w')

file.create_dataset('T={}_AvrM'.format('Ising'), data=M_Ising)
file.create_dataset('T={}_AvrM'.format('XY'), data=M_XY)

file.close()
