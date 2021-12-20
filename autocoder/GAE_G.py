import torch
import torch.nn as nn
import torch_geometric.nn as gnn
import torch_geometric.data as gdata
import torch_geometric.loader as gloader
import dataset

class gae(nn.Module):
    '''
    可以将模型修改为可以提取分类信息的autocoder
    或者可以采用现成有的无监督图分类方法
    '''
    def __init__(self,in_channels,out_channels):
        super(gae, self).__init__()
        self.conv1 = gnn.GraphConv(in_channels,2*out_channels)
        self.conv2 = gnn.GraphConv(2*out_channels,out_channels)
        self.encoder_S = nn.Sequential(
            nn.Linear(out_channels,8),
            nn.ReLU(),
            nn.Linear(8,4),
            nn.ReLU(),
            nn.Linear(4,1)
        )
        self.decoder = nn.Sequential(
            nn.Linear(2,4),
            nn.ReLU(),
            nn.Linear(4,8),
            nn.ReLU(),
            nn.Linear(8,out_channels)
        )
        self.relu = nn.ReLU()
    def encoder(self,x,edge_index,batch):
        h1 = self.relu(self.conv1(x,edge_index))
        h2 = self.relu(self.conv2(h1,edge_index))
        h3 = gnn.global_mean_pool(h2,batch)

        return h3,self.encoder_S(h3)

    def forward(self,x,edge_index,batch):

        x,z = self.encoder(x,edge_index,batch)
        x_ = self.decoder(z)

        return x,x_






