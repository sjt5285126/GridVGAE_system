import torch
import torch.nn as nn
import torch.nn.functional as F
import torch_geometric.nn as gnn
import pickle
from torch_geometric.loader import DataLoader, DenseDataLoader
from mincut_pool import Net

from sys import argv

if len(argv) < 4:
    print("please input:python mincut_pool_model.py epochs data_path name")
    exit()

epochs = int(argv[1])
data_path = argv[2]
name = argv[3]

path = 'data/IsingGraph/' + data_path

# 加载数据
data_file = open(path, 'rb')
data = pickle.load(data_file)
data_file.close()

n = 1000
trainLength = len(data[2 * n:])

test_loader = DataLoader(data[:n], batch_size=200)
valid_loader = DataLoader(data[n:2 * n], batch_size=200)
train_loader = DataLoader(data[2 * n:], batch_size=200)

# 定义设备
device = torch.device('cuda:3' if torch.cuda.is_available() else 'cpu')

# 获取nums_nodes 与 classes
nums_nodes = 8 * 8
classes = 3
features = 1
model = Net(nums_nodes).to(device)
optimizer = torch.optim.Adam(model.parameters(), lr=0.01, weight_decay=0.001)


def train():
    model.train()
    loss_all = 0

    for data in train_loader:
        data = data.to(device)
        optimizer.zero_grad()
        out, mc_loss, o_loss = model(data.x, data.edge_index, data.batch)
        loss = F.nll_loss(out, data.y.view(-1).long().to(device)) + mc_loss + o_loss
        loss.backward()
        loss_all += data.y.size(0) * loss.item()
        optimizer.step()

    return loss_all / trainLength


@torch.no_grad()
def test(loader):
    model.eval()
    correct = 0

    for data in loader:
        data = data.to(device)
        pred, mc_loss, o_loss = model(data.x, data.edge_index, data.batch)
        loss = F.nll_loss(pred, data.y.view(-1).long().to(device)) + mc_loss + o_loss
        correct += pred.max(dim=1)[1].eq(data.y.view(-1)).sum().item()

    return loss, correct


best_val_acc = test_acc = 0
best_val_loss = float('inf')
patience = start_patience = 50
for epoch in range(epochs):
    train_loss = train()
    _, train_acc = test(train_loader)
    train_acc = train_acc / trainLength
    val_loss, val_acc = test(valid_loader)
    val_acc = val_acc / n
    if val_loss < best_val_loss:
        test_loss, test_acc = test(test_loader)
        test_acc /= n
        best_val_acc = val_acc
        patience = start_patience
    else:
        patience -= 1
        if patience == 0:
            break
    print('Epoch: {:03d}, '
          'Train Loss: {:.3f}, Train Acc: {:.3f}, '
          'Val Loss: {:.3f}, Val Acc: {:.3f}, '
          'Test Loss: {:.3f}, Test Acc: {:.3f}'.format(epoch, train_loss,
                                                       train_acc, val_loss,
                                                       val_acc, test_loss,
                                                       test_acc))

torch.save({'epochs': epochs, 'state_dict': model.state_dict(), 'best_val_acc': best_val_acc,
            'optimizer': optimizer.state_dict(), 'batch_size': 200}, 'model/{}.pkl'.format(name))
