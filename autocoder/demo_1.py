import h5py
import torch
import numpy as np
from dataset import reshapeIsingHdf5,calculate


batch_size = 5000
# 填装测试数据,用来和生成模型的特征进行比对
testData = h5py.File('data/Ising/16eval.hdf5', 'r')
for key in testData.keys():
    testConfigs = testData[key][:batch_size]
testData.close()
print(testConfigs.shape)
testConfigs = reshapeIsingHdf5(testConfigs, batch_size)
print(testConfigs.shape)
TotalM, TotalE, AvrM, AvrE = calculate(testConfigs)

# 可以采用hdf5方式来存储数据的物理量，我们可以在生成数据的时候就将物理量全部计算出来，
# 这样可以节省一部分空间
f = h5py.File('16evalFeatures.hdf5','w')
f['TotalM'] = TotalM
f['TotalE'] = TotalE
f['AvrM'] = AvrM
f['AvrE'] = AvrE
f.close()