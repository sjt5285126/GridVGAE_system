from math import ceil

import torch
import torch.nn.functional as F
import torch.nn as nn
import torch_geometric.nn as gnn
from torch_geometric.nn import DenseSAGEConv, dense_diff_pool

'''
data_type:
    x:[2]
    
2022/11/30 9：50
模型修改意见：
1. 增加l2正则化，防止过拟合
2. 增加训练集数量
3. 重新调整深度网路结构，可能是数据过于简单，模型过于复杂
'''


# diffPool --- GNN

class GNN(torch.nn.Module):
    def __init__(self, in_channels, hidden_channels, out_channels, normalize=False, lin=True):
        super().__init__()

        self.conv1 = DenseSAGEConv(in_channels, hidden_channels, normalize)
        self.bn1 = gnn.BatchNorm(hidden_channels)
        self.conv2 = DenseSAGEConv(hidden_channels, hidden_channels, normalize)
        self.bn2 = gnn.BatchNorm(hidden_channels)
        self.conv3 = DenseSAGEConv(hidden_channels, out_channels, normalize)
        self.bn3 = gnn.BatchNorm(out_channels)
        self.dropout = nn.Dropout(0.5)
        self.relu = nn.ReLU()
        self.graphBN1 = gnn.GraphNorm(hidden_channels)
        self.graphBN2 = gnn.GraphNorm(hidden_channels)
        self.graphBN3 = gnn.GraphNorm(out_channels)

        if lin is True:
            self.lin = torch.nn.Linear(2 * hidden_channels + out_channels, out_channels)
        else:
            self.lin = None

    def bn(self, i, x):
        batch_size, num_nodes, num_channels = x.size()
        x = x.view(-1, num_channels)
        x = getattr(self, f'bn{i}')(x)
        x = x.view(batch_size, num_nodes, num_channels)
        return x

    def forward(self, x, adj, mask=None):
        batch_size, num_nodes, in_channels = x.size()

        x0 = x
        x1 = self.bn(1, self.relu(self.conv1(x0, adj, mask)))
        x1 = self.graphBN1(x1)
        x1 = self.dropout(x1)
        x2 = self.bn(2, self.relu(self.conv2(x1, adj, mask)))
        x2 = self.graphBN2(x2)
        x2 = self.dropout(x2)
        x3 = self.bn(3, self.relu(self.conv3(x2, adj, mask)))
        x3 = self.graphBN3(x3)
        x3 = self.dropout(x3)
        # 将结果并在一起
        x = torch.cat([x1, x2, x3], dim=-1)

        if self.lin is not None:
            x = self.lin(x).relu()

        return x


# GNN --- Net
class Net(torch.nn.Module):
    def __init__(self, max_nodes, features, classes):
        super().__init__()

        num_nodes = ceil(0.5 * max_nodes)

        self.gnn1_pool = GNN(features, 64, num_nodes, lin=False)
        self.gnn1_embed = GNN(features, 64, 64, lin=False)

        num_nodes = ceil(0.5 * num_nodes)

        self.gnn2_pool = GNN(3 * 64, 64, num_nodes)
        self.gnn2_embed = GNN(3 * 64, 64, 64, lin=False)

        self.gnn3_embed = GNN(3 * 64, 64, 64, lin=False)

        # 下游全连接神经网络算法，或许可以继续进行 autocoder 或 加入其他算法
        self.lin1 = torch.nn.Linear(3 * 64, 64)
        self.lin2 = torch.nn.Linear(64, classes)

    def forward(self, x, adj, mask=None):
        s = self.gnn1_pool(x, adj, mask)
        x = self.gnn1_embed(x, adj, mask)

        """
        第一次先不加正则化，后期有可能用到损失正则化调整拟合 
        """
        x, adj, l1, e1 = dense_diff_pool(x, adj, s)

        s = self.gnn2_pool(x, adj, mask)
        x = self.gnn2_embed(x, adj, mask)

        x, adj, l2, e2 = dense_diff_pool(x, adj, s)

        x = self.gnn3_embed(x, adj, mask)

        x = x.mean(dim=1)

        x = self.lin1(x).relu()
        x = self.lin2(x)

        return F.log_softmax(x, dim=-1), l1 + l2, e1 + e2
