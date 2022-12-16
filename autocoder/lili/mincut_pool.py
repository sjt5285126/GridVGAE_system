from math import ceil

import torch
import torch.nn.functional as F
import torch.nn as nn

import torch_geometric.nn as gnn
from torch_geometric.nn import GCNConv, DenseGraphConv, dense_mincut_pool
from torch_geometric.utils import to_dense_batch, to_dense_adj


class Net(torch.nn.Module):
    def __init__(self, num_nodes):
        super().__init__()

        self.conv1 = GCNConv(1, 16)
        num_nodes = ceil(0.5 * num_nodes)
        self.pool1 = nn.Linear(16, num_nodes)

        self.conv2 = DenseGraphConv(16, 32)
        num_nodes = ceil(0.5 * num_nodes)
        self.pool2 = nn.Linear(32, num_nodes)

        self.conv3 = DenseGraphConv(32, 64)
        self.lin1 = nn.Linear(64, 32)
        self.lin2 = nn.Linear(32, 16)
        self.lin3 = nn.Linear(16, 3)
        self.relu = nn.ReLU()

    def forward(self, x, edge_index, batch):
        x = self.relu(self.conv1(x, edge_index))

        x, mask = to_dense_batch(x, batch)
        adj = to_dense_adj(edge_index, batch)

        s = self.pool1(x)
        x, adj, mc1, o1 = dense_mincut_pool(x, adj, s, mask)

        x = self.relu(self.conv2(x, adj))
        s = self.pool2(x)

        x, adj, mc2, o2 = dense_mincut_pool(x, adj, s)

        x = self.conv3(x, adj)
        x = x.mean(dim=1)
        x = self.relu(self.lin1(x))
        x = self.relu(self.lin2(x))
        x = self.lin3(x)

        return F.log_softmax(x, dim=-1), mc1 + mc2, o1 + o2

