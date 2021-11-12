import torch
import torch.nn as nn
import torch_geometric.nn as gnn
import torch_geometric

class GAE_G(nn.Module):
    def __init__(self,in_channels,out_channels):
        super(GAE_G, self).__init__()
        # 对自编码网络模型进行设计
        self.conv1 = gnn.GCNConv(in_channels,out_channels)
        self.pool1 = gnn.TopKPooling(out_channels,ration=0.8)
        self.conv2 = gnn.GCNConv(out_channels,out_channels)
        self.pool2 = gnn.TopKPooling(out_channels,ratio=0.8)
        self.lin1 = nn.Linear(out_channels,8)
        self.lin2 = nn.Linear(8,2)
        self.relu = nn.ReLU()
    def encoder(self,x,edge_index):

        return

    def decoder(self,x,edge_index):
        # 反池化
        return
    def forward(self,x,edge_index):

        return  x


