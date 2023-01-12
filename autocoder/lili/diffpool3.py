from math import ceil

import torch
import torch.nn.functional as F
import torch.nn as nn
import torch_geometric.nn as gnn
from torch_geometric.nn import DenseSAGEConv, dense_diff_pool


# diffpool --- GNN3

class GNN(torch.nn.Module):
    def __init__(self, in_channels, hidden_channels1, hidden_channels2, out_channels):
        super().__init__()

        self.conv1 = gnn.DenseGraphConv(in_channels, hidden_channels1)
        self.bn1 = gnn.BatchNorm(hidden_channels1)
        self.graphBN1 = gnn.GraphNorm(hidden_channels1)
        self.conv2 = gnn.DenseGraphConv(hidden_channels1, hidden_channels2)
        self.bn2 = gnn.BatchNorm(hidden_channels2)
        self.graphBN2 = gnn.GraphNorm(hidden_channels2)
        self.conv3 = gnn.DenseGraphConv(hidden_channels2, out_channels)
        self.bn3 = gnn.BatchNorm(out_channels)
        self.graphBN3 = gnn.GraphNorm(out_channels)
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout()

    def bn(self, i, x):
        batch_size, num_nodes, num_channels = x.size()
        x = x.view(-1, num_channels)
        x = getattr(self, f'bn{i}')(x)
        x = x.view(batch_size, num_nodes, num_channels)
        return x

    def forward(self, x, adj, mask=None):
        # x1 = self.bn1(self.relu(self.conv1(x0, adj, mask)))
        x1 = self.bn(1, self.relu(self.conv1(x, adj, mask)))
        x1 = self.graphBN1(x1)
        x1 = self.dropout(x1)
        x2 = self.bn(2, self.relu(self.conv2(x1, adj, mask)))
        x2 = self.graphBN2(x2)
        x2 = self.dropout(x2)
        x3 = self.bn(3, self.relu(self.conv3(x2, adj, mask)))
        x3 = self.graphBN3(x3)
        x3 = self.dropout(x3)

        return x3


class Net(torch.nn.Module):
    def __init__(self, max_nodes, features, classes):
        super().__init__()

        num_nodes = ceil(0.5 * max_nodes)

        self.gnn1_pool = GNN(features, 16, 32, num_nodes)
        self.gnn1_embed = GNN(features, 16, 32, 64)

        num_nodes = ceil(0.5 * num_nodes)

        self.gnn2_pool = GNN(64, 64, 64, num_nodes)
        self.gnn2_embed = GNN(64, 64, 64, 64)

        self.gnn3_embed = GNN(64, 64, 64, 64)

        self.lin1 = torch.nn.Linear(64, 32)
        self.lin2 = torch.nn.Linear(32, 16)
        self.lin3 = torch.nn.Linear(16, classes)
        # 增加dropout在全连接层
        self.dropout = nn.Dropout()
        self.relu = nn.ReLU()

    def forward(self, x, adj, mask=None):
        s = self.gnn1_pool(x, adj, mask)
        x = self.gnn1_embed(x, adj, mask)
        x, adj, l1, e1 = dense_diff_pool(x, adj, s)
        s = self.gnn2_pool(x, adj, mask)
        x = self.gnn2_embed(x, adj, mask)
        x, adj, l2, e2 = dense_diff_pool(x, adj, s)
        x = self.gnn3_embed(x, adj, mask)
        x = x.mean(dim=1)

        x = self.relu(self.lin1(x))
        x = self.dropout(x)

        x = self.relu(self.lin2(x))
        x = self.dropout(x)

        x = self.lin3(x)

        return F.log_softmax(x, dim=-1), l1 + l2, e1 + e2
