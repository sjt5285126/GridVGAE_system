import torch
import torch.nn as nn
import torch.nn.functional as F
import torch_geometric.nn as gnn
import pickle
from torch_geometric.loader import DenseDataLoader
from diffpool3 import GNN, Net

from sys import argv

if len(argv) < 4:
    print("please input:python diffpool_model.py epochs data_path name")
    exit()

epochs = int(argv[1])
data_path = argv[2]
name = argv[3]

path = 'data/IsingGraph/' + data_path
# 加载数据
data_file = open(path, 'rb')
data = pickle.load(data_file)
data_file.close()

n = 2000

test_loader = DenseDataLoader(data[:n], batch_size=200)
valid_loader = DenseDataLoader(data[n:2 * n], batch_size=200)
train_loader = DenseDataLoader(data[2 * n:], batch_size=200)

# 定义设备
device = torch.device('cuda:1' if torch.cuda.is_available() else 'cpu')
# 获取 nums_nodes 与 classes
nums_nodes = 64
classes = 3
features = 1
model = Net(max_nodes=nums_nodes, features=features, classes=classes).to(device)
optimizer = torch.optim.Adam(model.parameters(), lr=0.01, weight_decay=0.001)
print(model )

def train(loader):
    model.train()
    loss_all = 0
    print("--------------------------")
    for batch in loader:
        batch = batch.to(device)
        optimizer.zero_grad()
        output, l, e = model(batch.x, batch.adj)
        """
        损失函数方向可能还需要继续修改
        """
        # print(output.type())
        # print(batch.y.view(-1).type())
        # print(batch.y.view(-1).long().to(device).type())
        loss = F.nll_loss(output, batch.y.view(-1).long().to(device)) + l + e
        loss.backward()
        loss_all += batch.y.size(0) * float(loss)
        optimizer.step()
    return loss_all / len(data[2 * n:])


@torch.no_grad()
def test(loader):
    model.eval()
    correct = 0
    for batch in loader:
        batch = batch.to(device)
        pred = model(batch.x, batch.adj)[0].max(dim=1)[1]
        correct += int(pred.eq(batch.y.view(-1)).sum())

    return correct / n


best_val_acc = test_acc = 0
for epoch in range(epochs):
    train_loss = train(train_loader)
    val_acc = test(valid_loader)
    if val_acc > best_val_acc:
        test_acc = test(test_loader)
        best_val_acc = val_acc
    print(f'Epoch: {epoch:03d}, Train Loss: {train_loss:.4f}, '
          f'Val Acc: {val_acc:.4f}, Test Acc: {test_acc:.4f}')

torch.save({'epochs': epochs, 'state_dict': model.state_dict(), 'best_val_acc': best_val_acc,
            'optimizer': optimizer.state_dict(), 'batch_size': 200}, 'model/{}.pkl'.format(name))
