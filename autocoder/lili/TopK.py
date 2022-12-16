import torch
import torch.nn as nn
import torch_geometric.nn as gnn
import torch.nn.functional as F
from torch_geometric.nn import GraphConv, TopKPooling
from torch_geometric.nn import global_max_pool as gmp
from torch_geometric.nn import global_mean_pool as gap


class Net(nn.Module):
    def __init__(self, num_features, num_classes):
        super().__init__()

        self.conv1 = GraphConv(num_features, 64)
        self.pool1 = TopKPooling(64, ratio=0.8)
        self.conv2 = GraphConv(64, 64)
        self.pool2 = TopKPooling(64, ratio=0.8)
        self.conv3 = GraphConv(64, 64)
        self.pool3 = TopKPooling(64, ratio=0.8)

        self.lin1 = nn.Linear(128, 64)
        self.lin2 = nn.Linear(64, 32)
        self.lin3 = nn.Linear(32, 16)
        self.lin4 = nn.Linear(16, num_classes)
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout()

    def forward(self, data):
        x, edge_index, edge_attr, batch = data.x, data.edge_index, data.edge_attr, data.batch
        # print(x.shape)
        x = self.relu(self.conv1(x, edge_index, edge_attr))
        # print("conv1:{}".format(x.shape))
        x, edge_index, edge_attr, batch, _, _ = self.pool1(x, edge_index, edge_attr, batch)
        # x1 = torch.cat([gmp(x,batch),gap(x,batch)],dim=1)
        x1 = torch.cat([gmp(x, batch), gap(x, batch)], dim=1)
        # print("pool1:{}".format(x.shape))
        x = self.relu(self.conv2(x, edge_index))
        # print("conv2:{}".format(x.shape))
        x, edge_index, edge_attr, batch, _, _ = self.pool2(x, edge_index, edge_attr, batch)
        # x2 = torch.cat([gmp(x,batch),gap(x,batch)],dim=1)
        # print("pool2:{}".format(x.shape))
        x2 = torch.cat([gmp(x, batch), gap(x, batch)], dim=1)
        x = self.relu(self.conv3(x, edge_index))
        x, edge_index, edge_attr, batch, _, _ = self.pool3(x, edge_index, edge_attr, batch)
        # x3 = torch.cat([gmp(x,batch),gap(x,batch)],dim=1)
        # print("conv3:{}".format(x.shape))
        x3 = torch.cat([gmp(x, batch), gap(x, batch)], dim=1)
        # x = x1 + x2 + x3
        x = x1 + x2 + x3
        x = self.relu(self.lin1(x))
        # print("lin1:{}".format(x.shape))
        x = self.dropout(x)
        x = self.relu(self.lin2(x))
        # print("lin2:{}".format(x.shape))
        x = self.dropout(x)
        x = self.relu(self.lin3(x))
        x = self.dropout(x)
        x = F.log_softmax(self.lin4(x), dim=-1)
        # print("lin3:{}".format(x.shape))
        return x
