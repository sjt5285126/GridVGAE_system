import torch
import torch.nn as nn
import torch_geometric.nn as gnn
from dataset import *
import torch_geometric.data as gdata

# 改造成图级别的自编码器 可以识别不同的晶体结构
class GAE_encode(nn.Module):
    def __init__(self,in_channels,out_channels):
        super(GAE_encode, self).__init__()
        self.conv1 = gnn.GCNConv(in_channels,2*out_channels)
        self.conv2 = gnn.GCNConv(2*out_channels,out_channels)
        self.relu = nn.ReLU()
        self.softmax = nn.Softmax()

    def forward(self,x,edge_index):

        # 该处的卷积只对应节点卷积，所以边信息不变
        x = self.relu(self.conv1(x,edge_index))
        x = self.conv2(x,edge_index)

        return x


channels = 2
dev = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
# Ising模型的节点特征 只有1或者-1
model = gnn.GAE(GAE_encode(1,channels)).to(dev)

# 加载数据，划分数据
data_train = data[0:80]
data_test = data[80:1000]
data_trainloader = gdata.DataLoader(data_train, batch_size=5, shuffle=True)
#print(data_trainloader)
data_testloader = gdata.DataLoader(data_test, batch_size=50, shuffle=True)
#print(data_testloader)

optimizer = torch.optim.Adam(model.parameters(),lr=0.01)
print(model)

def train():
    for one_train_batch in data_trainloader:
        model.train()
        optimizer.zero_grad()
        one_train_batch = one_train_batch.to(dev)
        z = model.encode(one_train_batch.x,one_train_batch.edge_index)
        loss = model.recon_loss(z,one_train_batch.edge_index)
        loss.backward()
        optimizer.step()

def test():
    for one_test_batch in data_testloader:
        model.eval()
        one_test_batch = one_test_batch.to(dev)
        with torch.no_grad():
            z = model.encode(one_test_batch.x,one_test_batch.edge_index)
        ac,ap = model.test(z, one_test_batch.edge_index, one_test_batch.edge_index)

    return z,ac,ap

for epoch in range(1):
    train()
    z,ac,ap = test()
    if epoch%100 == 0:
        print('Epoch: {:03d}, z: {}, ac: {:.4f},ac: {:.4f}'.format(epoch, z, ac,ap))








