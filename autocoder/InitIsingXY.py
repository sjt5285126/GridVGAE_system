import numpy as np
import math
import torch


#
# x = np.loadtxt('data\gouxingTest(1).dat')
# print(x.shape)
# x = x.reshape(500, 2, 8, 8)
# x = x.reshape(500, 2, 64)
# x = np.transpose(x, axes=(0, 2, 1))
# # for config in x:
# #     if np.sum(config[:,0])<0:
# #         config[:,0] *= -1
# #
# x[:, :, 0] = np.where(np.sum(x[:, :, 0]) > 0, x[:, :, 0], -x[:, :, 0])
# x[:, :, 0] = np.where(x[:, :, 0] > 0, x[:, :, 0], 0)

def cacculateIsingXY(configs):
    size = configs.shape[1]
    M_Ising = np.zeros([configs.shape[0], ])
    M_XY = np.zeros([configs.shape[0], ])
    for i, config in enumerate(configs):
        M_Ising[i] = np.abs(config[:, :, 0].sum())
        M_XY[i] = np.abs(config[:, :, 1].sum())

    M_Ising = M_Ising / (size ** 2)
    M_XY = M_XY / (size ** 2)

    return M_Ising, M_XY, M_XY * (size ** 0.125)


def genIsingXY(path, size):
    configs = np.loadtxt(path)
    num = configs.shape[0] // (2 * size)
    configs = configs.reshape((num, 2, size, size))
    configs = configs.reshape((num, 2, size * size))
    configs = np.transpose(configs, axes=(0, 2, 1))
    # compute = configs.reshape(num,size,size,2)
    # print(cacculateIsingXY(compute)[2].mean())
    configs[:, :, 0] = np.where(np.sum(configs[:, :, 0]) > 0, configs[:, :, 0], -configs[:, :, 0])
    configs[:, :, 0] = np.where(configs[:, :, 0] > 0, configs[:, :, 0], 0)
    configs[:, :, 1] = configs[:, :, 1] % (2 * np.pi)
    configs[:, :, 1] = configs[:, :, 1] / (2 * np.pi)
    return configs

# configs, min, MaxSubMin = genIsingXY('data/gouxingTestA0.32L16_1.dat', 16)
# print(configs,min, MaxSubMin)
# configs = genIsingXY('data/gouxingTestA0.32_1.dat', 8)
# print(configs)