import torch
import torch.nn as nn
import torch_geometric.nn as gnn
from torch_geometric.nn.models.autoencoder import VGAE,GAE
import VGAE as vgae
from torch_geometric.datasets import Planetoid
import torch_geometric.transforms as T
from torch_geometric.utils import train_test_split_edges
import os.path as osp

# 准备数据集
dataset = 'Cora'
path = osp.join(osp.dirname(osp.realpath(__file__)),'data',dataset)
dataset = Planetoid(path,dataset,transform=T.NormalizeFeatures())
data = dataset[0] # dataset[0] 为适合pyg初始的Data数据

# 使用GCNconv 无法进行小批量次初始，可以使用别的模型来进行小批量次处理

channels = 8
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model = VGAE(vgae.VGAE_encoder(dataset.num_features,channels)).to(device)
# 自编码器的重构对象就是其本身，所以训练集，测试机，验证集，标签都是本身
data.train_mask = data.val_mask = data.test_mask = data.y = None

# 将data中的边 随机划分成 训练集边
data = train_test_split_edges(data)
x,train_pos_edge_index = data.x.to(device),data.train_pos_edge_index.to(device)
optimizer = torch.optim.Adam(model.parameters(),lr=0.01)

def train():
    model.train()
    optimizer.zero_grad()
    z = model.encode(x,train_pos_edge_index)
    # recon_loss 继承自GAE类
    loss = model.recon_loss(z,train_pos_edge_index)+model.kl_loss()/data.num_nodes
    loss.backward()
    optimizer.step()

def test(pos_edge_index,neg_edge_index):
    model.eval()
    # 使用torch.no_grad的语句不会被追踪梯度,在测试时可以节约内存
    with torch.no_grad():
        z = model.encode(x,train_pos_edge_index)

    # test方法继承自GAE类
    return model.test(z,pos_edge_index,neg_edge_index)

for epoch in range(1,401):
    train()
    auc,ap = test(data.test_pos_edge_index,data.test_neg_edge_index)
    print('Epoch: {:03d}, AUC: {:.4f}, AP: {:.4f}'.format(epoch, auc, ap))
