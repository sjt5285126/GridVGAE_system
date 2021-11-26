import torch
import numpy as np
import torch.nn as nn
import torch.nn.functional as F
import torch_geometric.nn as gnn
from torch_geometric.nn.models.autoencoder import VGAE,GAE
from VGAE import VGAE_encoder as vgae
from torch_geometric.datasets import Planetoid
import torch_geometric.transforms as T
from torch_geometric.utils import train_test_split_edges
import os.path as osp
import dataset

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model = VGAE(vgae(1,2))
data = dataset.init_data(4,dataset.init_p(4),4)

# 得到采样的结果
# 采样得到的结果其实属于二值分布，所以想要选取更好的损失函数
z = model.encode(data[0].x,data[0].edge_index)
z = F.softmax(z)
#print(torch.argmax(F.softmax(z,dim=1),dim=1))
print("z:{}".format(z))
print("x:{}".format(data[0].x))
criterion1 = nn.MSELoss()
criterion2 = nn.NLLLoss()
criterion3 = nn.CrossEntropyLoss()
flatten = nn.Flatten(start_dim=0)
unflatten = nn.Unflatten(0,(4,4))
#loss1 = criterion1(z,data[0].x)+model.kl_loss()/16
#print(loss1)
loss2 = criterion2(z,flatten(data[0].x.to(torch.long)))
print(loss2)
loss3 = criterion3(z,flatten(data[0].x.to(torch.long)))
print(loss3)
x_ = model.decode(data[0].x,data[0].edge_index)

print(x_.shape)
print(data[0].x.shape)

z = torch.argmax(z,dim=1)
z = unflatten(z)
print(z)