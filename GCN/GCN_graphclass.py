import torch
import torch.nn as nn
import torch.nn.functional as F
import torch_geometric.nn as gnn


class Net(torch.nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.conv1 = gnn.GraphConv(1, 16)
        self.pool1 = gnn.TopKPooling(16, ratio=0.8)
        self.conv2 = gnn.GraphConv(16, 16)
        self.pool2 = gnn.TopKPooling(16, ratio=0.8)

        self.lin1 = nn.Linear(32,8)
        self.lin2 = nn.Linear(8,4)
        # 输出为2分类，所以输出为2维的
        self.lin3 = nn.Linear(4,2)




    def forward(self,x,edge_index,batch):
        x = F.relu(self.conv1(x, edge_index))
        x, edge_index, _, batch, _, _ = self.pool1(x, edge_index, None, batch)
        x1 = torch.cat([gnn.global_max_pool(x, batch), gnn.global_mean_pool(x, batch)], dim=1)

        x = F.relu(self.conv2(x, edge_index))
        x, edge_index, _, batch, _, _ = self.pool2(x, edge_index, None, batch)
        x2 = torch.cat([gnn.global_max_pool(x, batch), gnn.global_mean_pool(x, batch)], dim=1)

        x = x1 + x2

        x = F.relu(self.lin1(x))
        x = F.dropout(x, p=0.5, training=self.training)
        x = F.relu(self.lin2(x))
        x = F.log_softmax(self.lin3(x), dim=-1)

        return x