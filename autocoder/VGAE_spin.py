import pickle
import torch_geometric.data as gdata
import torch_geometric.loader as gloader
import torch
import numpy as np
import torch.nn as nn
import torch.nn.functional as F
import torch_geometric.nn as gnn
from torch_geometric.nn.models.autoencoder import VGAE,GAE

from VGAE import VGAE_encoder as vgae
import torch_geometric.transforms as T
import os.path as osp
import dataset

class EncoderSpin(nn.Module):
    """
    推断模型,用来生成概率分布
    因为是对每一个点进行采样，所以要避免过拟合
    """
    def __init__(self):
        '''
        模型介绍:
        推断模型采用GraphConv卷积(具体什么卷积的效果好仍需要不断测试),同时采用了
        图归一化与dropout,后续仍可能使用adj_dropout
        '''
        super(EncoderSpin, self).__init__()
        self.conv1 = gnn.GraphConv(1, 8)
        self.conv_mu = gnn.GraphConv(8, 16)
        self.conv_logvar = gnn.GraphConv(8, 16)
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(p=0.5)
        self.graph_norm1= gnn.GraphNorm(8)

    def forward(self, x, edge_index,edge_weight,batch):
        x = self.relu(self.conv1(x, edge_index,edge_weight))
        # 在中间可以加入dropout操作
        x = self.dropout(x)
        x = self.graph_norm1(x,batch)
        mu = self.conv_mu(x, edge_index,edge_weight)
        logvar = self.conv_logvar(x, edge_index,edge_weight)

        return mu, logvar

class DecoderSpin(nn.Module):
    def __init__(self):
        super(DecoderSpin, self).__init__()
        self.decoder = nn.Sequential(
            nn.Linear(16,8),
            nn.ReLU(),
            nn.Linear(8,4),
            nn.ReLU(),
            nn.Linear(4,2),
            nn.Softmax()
        )

    def forward(self,z):
        return(self.decoder(z))



class SVGAE(VGAE):
    def __init__(self,encoder,decoder):
        super().__init__(encoder,decoder)
        self.loss = nn.CrossEntropyLoss()
        self.flatten = nn.Flatten(start_dim=0)

    def recon_loss(self,x,x_):
        size = x.shape[0]
        loss = self.loss(x_,self.flatten(x).to(torch.long)) + self.kl_loss()
        return loss


device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model = SVGAE(EncoderSpin(),DecoderSpin()).to(device)
print(model)
datafile = open('data/IsingGraph/data3.pkl','rb')
data = pickle.load(datafile)
datafile.close()
print(data)
data_train_batchs = gloader.DataLoader(data,batch_size=2,shuffle=True)

optim = torch.optim.Adam(model.parameters(),lr=0.01)

for epoch in range(100):
    model.train()
    for d in data_train_batchs:
        d = d.to(device)
        z = model.encode(d.x,d.edge_index,d.edge_attr,d.batch)
        x_ = model.decode(z)
        optim.zero_grad()
        loss = model.recon_loss(d.x,x_)
        print('loss:{}'.format(loss))
        loss.backward()
        optim.step()





