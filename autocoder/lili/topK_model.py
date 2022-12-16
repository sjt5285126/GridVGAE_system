import torch
import torch.nn as nn
import torch.nn.functional as F
import torch_geometric.nn as gnn
import pickle
from torch_geometric.loader import DataLoader
from TopK import Net

from sys import argv

if len(argv) < 3:
    print("please input:python topK_model.py epochs data_path name")
    exit()

epochs = int(argv[1])
data_path = argv[2]
name = argv[3]

path = 'data/IsingGraph/' + data_path

# 加载数据
data_file = open(path, 'rb+')
data = pickle.load(data_file)
data_file.close()

n = 1000

trainLength = len(data[2 * n:])
testLength = n

test_loader = DataLoader(data[:n], batch_size=200)
valid_loader = DataLoader(data[n:2 * n], batch_size=200)
train_loader = DataLoader(data[2 * n:], batch_size=200)

device = torch.device('cuda:1' if torch.cuda.is_available() else 'cpu')

num_features = 1
num_classes = 3

model = Net(num_features=num_features, num_classes=num_classes).to(device)
optimizer = torch.optim.Adam(model.parameters(), lr=0.01, weight_decay=0.001)


def train():
    model.train()
    loss_all = 0
    for data in train_loader:
        data = data.to(device)
        optimizer.zero_grad()
        output = model(data)
        # print(output.dtype)
        # print(data.y.dtype)
        loss = F.nll_loss(output, data.y.long().to(device))
        loss.backward()
        loss_all += data.num_graphs * loss.item()
        optimizer.step()
    return loss_all / trainLength

@torch.no_grad()
def test(loader):
    model.eval()

    correct = 0
    for data in loader:
        data = data.to(device)
        pred = model(data).max(dim=1)[1]
        correct += pred.eq(data.y).sum().item()

    return correct


for epoch in range(epochs):
    loss = train()
    train_acc = test(train_loader) / trainLength
    test_acc = test(test_loader) / testLength
    print(f'Epoch: {epoch:03d}, Loss: {loss:.5f}, Train Acc: {train_acc:.5f}, '
          f'Test Acc: {test_acc:.5f}')

torch.save({'epochs': epochs, 'state_dict': model.state_dict(),
            'optimizer': optimizer.state_dict(), 'batch_size': 200}, 'model/{}.pkl'.format(name))
