import torch
import numpy as np
import torch.nn as nn
import torch.nn.functional as F
import torch_geometric.nn as gnn
from torch_geometric.nn.models.autoencoder import VGAE,GAE
from VGAE import VGAE_encoder as vgae
from torch_geometric.datasets import Planetoid
import torch_geometric.transforms as T
from torch_geometric.utils import train_test_split_edges
import os.path as osp
import dataset

class EncoderSpin(nn.Module):
    def __init__(self):
        super(EncoderSpin, self).__init__()
        self.conv1 = gnn.GCNConv(1, 32, cached=True)
        self.conv_mu = gnn.GCNConv(32, 16, cached=True)
        self.conv_logvar = gnn.GCNConv(32, 16, cached=True)
        self.relu = nn.ReLU()

    def forward(self, x, edge_index):
        x = self.relu(self.conv1(x, edge_index))
        # 在中间可以加入dropout操作
        # x = nn.Dropout(x)
        mu = self.conv_mu(x, edge_index)
        logvar = self.conv_logvar(x, edge_index)

        return mu, logvar

class DecoderSpin(nn.Module):
    def __init__(self):
        self.decoder = nn.Sequential(
            nn.Linear(16,8),
            nn.ReLU(),
            nn.Linear(8,4),
            nn.ReLU(),
            nn.Linear(4,2),
            nn.Softmax()
        )

    def forward(self,z):
        return(self.decoder(z))



class SVGAE(VGAE):
    def __init__(self,encoder,decoder):
        super().__init__(encoder,decoder)
        self.loss = nn.CrossEntropyLoss()
        self.flatten = nn.Flatten(start_dim=0)

    def recon_loss(self,x,x_):
        size = x.shape[0]
        loss = self.loss(x_,self.flatten(x).to(torch.long))+self.kl_loss()/size
        return loss






