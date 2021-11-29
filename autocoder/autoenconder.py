import math

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch_geometric
import torch_geometric.nn as gnn
import dataset
import numpy as np
import pickle
import h5py
from IsingGrid import Grid
import pprint

#f = open('data/IsingGraph/data3.pkl','rb')
hdf = h5py.File('data/Ising/3size.hdf5','r')

print(list(hdf.keys()))
for key in list(hdf.keys()):
    #print('{}\n{}'.format(key,hdf[key][:]))
    print(hdf[key][0])
    size,config=dataset.reshape_Ising(hdf[key][0])
    gird = Grid(size,1,config)
    print(gird.canvas)

