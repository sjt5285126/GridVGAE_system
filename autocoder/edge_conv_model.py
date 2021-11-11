
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch_geometric.nn as gnn

"""
    图级监督学习模型
    输入 一张图；包括节点特征和边特征；也就是说是一个连接
    输出：2维；两分类嘛
    使用CGConv(),TOPKPooling()
    两层卷积，三层池化
    现存问题：
        当batch改变时，网络模型就得变，得找一个公共变量
"""

class Net(torch.nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.conv1 = gnn.CGConv(channels=(1, 1), dim=1, batch_norm=True)
        self.pool1 = gnn.TopKPooling(800, ratio=0.1)
        self.conv2 = gnn.CGConv(channels=(1, 1), dim=1, batch_norm=True)
        self.pool2 = gnn.TopKPooling(80, ratio=0.1)
        self.pool3 = gnn.TopKPooling(8, ratio=0.25)


        self.lin1 = nn.Linear(1, 8)
        # 输出为2分类，所以输出为2维的
        self.lin2 = nn.Linear(8, 2)




    def forward(self, data, batch):
        x = data.x
        edge_index = data.edge_index
        edge_attr = data.edge_attr
        x = self.conv1(x=x, edge_index=edge_index, edge_attr=edge_attr)
        x, edge_index, edge_attr, batch, _, _ = self.pool1(x=x, edge_index=edge_index, edge_attr=edge_attr, batch=batch)

        x = self.conv2(x=x, edge_index=edge_index, edge_attr=edge_attr)
        x, edge_index, edge_attr, batch, _, _ = self.pool2(x=x, edge_index=edge_index, edge_attr=edge_attr, batch=batch)
        x, edge_index, edge_attr, batch, _, _ = self.pool3(x=x, edge_index=edge_index, edge_attr=edge_attr, batch=batch)
        # x, edge_index, edge_attr, batch, _, _ = self.pool4(x=x, edge_index=edge_index, edge_attr=edge_attr, batch=batch)
        x = F.relu(self.lin1(x))
        x = F.dropout(x, p=0.5, training=self.training)
        # x = F.relu(self.lin2(x))
        x = F.log_softmax(self.lin2(x), dim=1)

        return x


