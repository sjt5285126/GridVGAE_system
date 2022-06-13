import pickle
import random
from sys import argv
import torch_geometric.loader as gloader
import torch
import torch.nn as nn
import torch_geometric.nn as gnn
from torch_geometric.nn.models.autoencoder import VGAE, GAE
import os
from dataset import reshapeIsing


# 保留修改意见 使用MSELoss 与 L1Loss

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
        self.conv1 = gnn.GATConv(1, 16, dropout=0.2, edge_dim=1)
        self.conv2 = gnn.GATConv(16, 32, dropout=0.2, edge_dim=1)
        self.conv3 = gnn.GATConv(32, 64, dropout=0.2, edge_dim=1)
        self.conv_mu = gnn.GATConv(64, 64, dropout=0.2, edge_dim=1)
        self.conv_logvar = gnn.GATConv(64, 64, dropout=0.2, edge_dim=1)
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(p=0.5)
        self.batch_norm1 = gnn.BatchNorm(16)
        self.batch_norm2 = gnn.BatchNorm(32)
        self.batch_norm3 = gnn.BatchNorm(64)
        self.graph_norm1 = gnn.GraphNorm(64)

    def forward(self, x, edge_index, edge_weight, batch):
        x = self.relu(self.conv1(x, edge_index, edge_weight))
        x = self.batch_norm1(x)
        x = self.relu(self.conv2(x, edge_index, edge_weight))
        x = self.batch_norm2(x)
        x = self.relu(self.conv3(x, edge_index, edge_weight))
        # 在中间可以加入dropout操作
        x = self.batch_norm3(x)
        # 加入图归一化操作
        x = self.graph_norm1(x, batch)
        mu = self.conv_mu(x, edge_index, edge_weight)
        logvar = self.conv_logvar(x, edge_index, edge_weight)

        return mu, logvar


class DecoderSpin(nn.Module):
    def __init__(self):
        super(DecoderSpin, self).__init__()
        self.decoder = nn.Sequential(
            nn.Linear(64, 32),
            nn.ReLU(),
            nn.Linear(32, 16),
            nn.ReLU(),
            nn.Linear(16, 8),
            nn.ReLU(),
            nn.Linear(8, 4),
            nn.ReLU(),
            nn.Linear(4, 1),
            nn.Sigmoid()
        )

    def forward(self, z):
        return (self.decoder(z))


class SVGAE(VGAE):
    def __init__(self, encoder, decoder):
        super().__init__(encoder, decoder)
        # 定义交叉熵损失函数
        self.loss = nn.MSELoss()
        self.flatten = nn.Flatten(start_dim=0)
        # 添加L1正则化 将x_模型还原为Ising模型的样式来进行L1正则化计算
        self.L1loss = nn.L1Loss()

    def recon_loss(self, x, x_):
        # 增加能量约束 loss_energy = (x_.energy - x.energy).mean()
        # 在model_16_220224 中 loss = -self.loss(x_,x)
        # 目前想法是在之后的model中给分别测试
        # loss = self.loss(x_,x).mean() **
        # loss = -self.loss(x_,x).mean()
        # loss = self.loss(x_,x)
        loss = self.loss(x_, x) + self.L1loss(x_, x)
        return loss

    def get_mu_logstd(self):
        return self.__mu__, self.__logstd__
