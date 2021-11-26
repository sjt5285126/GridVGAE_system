import torch
import torch.nn as nn
import torch.nn.functional as F
import torch_geometric
import torch_geometric.nn as gnn
import dataset
from torch_geometric.nn.models.autoencoder import VGAE,InnerProductDecoder

class VGAE_encoder(nn.Module):
    def __init__(self,in_channels,out_channels):
        super(VGAE_encoder, self).__init__()
        self.conv1 = gnn.GCNConv(in_channels,32,cached=True)
        self.conv_mu = gnn.GCNConv(32,out_channels,cached=True)
        self.conv_logvar = gnn.GCNConv(32,out_channels,cached=True)
        self.relu = nn.ReLU() #nn.Relu() 相当于一个模型,声明后才可使用


    def forward(self,x,edge_index):

        x = self.relu(self.conv1(x,edge_index))
        # 在中间可以加入dropout操作
        # x = nn.Dropout(x)
        mu = self.conv_mu(x,edge_index)
        logvar = self.conv_logvar(x,edge_index)

        return mu,logvar

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model = VGAE_encoder(1,2)
data = dataset.init_data(4,dataset.init_p(4),4)

mu,logvar = model(data[0].x,data[0].edge_index)
print(mu.shape)
print(logvar.shape)

