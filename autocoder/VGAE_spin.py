import pickle
import random
from sys import argv
import torch_geometric.loader as gloader
import torch
import torch.nn as nn
import torch_geometric.nn as gnn
from torch_geometric.nn.models.autoencoder import VGAE, GAE
import os


class EncoderSpin(nn.Module):
    """
    推断模型,用来生成概率分布
    因为是对每一个点进行采样，所以要避免过拟合
    """

    def __init__(self):
        '''
        模型介绍:
        推断模型采用GraphConv卷积(具体什么卷积的效果好仍需要不断测试),同时采用了
        图归一化与dropout,后续仍可能使用adj_dropout
        '''
        super(EncoderSpin, self).__init__()
        self.conv1 = gnn.GraphConv(1, 16)
        self.conv2 = gnn.GraphConv(16, 32)
        self.conv3 = gnn.GraphConv(32, 64)
        self.conv_mu = gnn.GraphConv(64, 64)
        self.conv_logvar = gnn.GraphConv(64, 64)
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(p=0.5)
        self.graph_norm1 = gnn.GraphNorm(64)

    def forward(self, x, edge_index, edge_weight, batch):
        x = self.relu(self.conv1(x, edge_index, edge_weight))
        x = self.relu(self.conv2(x, edge_index, edge_weight))
        x = self.relu(self.conv3(x, edge_index, edge_weight))
        # 在中间可以加入dropout操作
        x = self.dropout(x)
        # 加入图归一化操作
        x = self.graph_norm1(x, batch)
        mu = self.conv_mu(x, edge_index, edge_weight)
        logvar = self.conv_logvar(x, edge_index, edge_weight)

        return mu, logvar


class DecoderSpin(nn.Module):
    def __init__(self):
        super(DecoderSpin, self).__init__()
        self.decoder = gnn.Sequential('x,edge_index', [
            (gnn.GraphConv(64, 32), 'x,edge_index -> x'),
            nn.ReLU(),
            (gnn.GraphConv(32, 16), 'x,edge_index -> x'),
            nn.ReLU(),
            nn.Linear(16, 8),
            nn.ReLU(),
            nn.Linear(8, 4),
            nn.ReLU(),
            nn.Linear(4, 2),
            nn.Softmax(dim=-1)
        ])

    def forward(self, z, edge_index):
        return (self.decoder(z, edge_index))


class SVGAE(VGAE):
    def __init__(self, encoder, decoder):
        super().__init__(encoder, decoder)
        # 定义交叉熵损失函数
        self.loss = nn.CrossEntropyLoss()
        self.flatten = nn.Flatten(start_dim=0)

    def recon_loss(self, x, x_):
        size = x.shape[0]
        loss = self.loss(x_, self.flatten(x).to(torch.long)) + self.kl_loss()
        return loss
