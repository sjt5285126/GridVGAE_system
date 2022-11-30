import torch.nn as nn
import torch_geometric.nn as gnn
from torch_geometric.nn import GraphConv
from torch_geometric.nn import dense_diff_pool

'''
对于数据的构建，如果以自旋方式来建立图形，可能会导致学习效果飘忽
1. 仍然使用 0 1 构建数据集
2. 使用哈密顿量来构建数据集  



'''


class Encoder(nn.Module):
    def __init__(self):
        super(Encoder, self).__init__()
        self.conv1 = gnn.GraphConv(1, 16)
        self.conv2 = gnn.GraphConv(16, 32)
        self.conv3 = gnn.GraphConv(32, 64)
        self.conv_mu = gnn.GraphConv(64, 64)
        self.conv_logVar = gnn.GraphConv(64, 64)
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(p=0.5)
        self.batch_norm1 = gnn.BatchNorm(16)
        self.batch_norm2 = gnn.BatchNorm(32)
        self.batch_norm3 = gnn.BatchNorm(64)
        self.graph_norm1 = gnn.GraphNorm(16)
        self.graph_norm2 = gnn.GraphNorm(32)
        self.graph_norm3 = gnn.GraphNorm(64)
