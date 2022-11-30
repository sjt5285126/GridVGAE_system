import os.path as osp
from math import ceil

import torch
import torch.nn.functional as F

import torch_geometric.transforms as T
from torch_geometric.datasets import TUDataset
from torch_geometric.loader import DenseDataLoader
from torch_geometric.nn import DenseSAGEConv, dense_diff_pool

max_nodes = 150


class MyFilter(object):
    def __call__(self, data):
        return data.num_nodes <= max_nodes


path = osp.join(osp.dirname(osp.realpath(__file__)), '..', 'data',
                'PROTEINS_dense')
dataset = TUDataset(path, name='PROTEINS', transform=T.ToDense(max_nodes),
                    pre_filter=MyFilter())
dataset = dataset.shuffle()
# print('dataset shape:\n{}'.format(dataset))
n = (len(dataset) + 9) // 10
test_dataset = dataset[:n]
val_dataset = dataset[n:2 * n]
train_dataset = dataset[2 * n:]
test_loader = DenseDataLoader(test_dataset, batch_size=20)
# print('test_loader:{}'.format(test_loader))
val_loader = DenseDataLoader(val_dataset, batch_size=20)
# print('val_loader:{}'.format(val_loader))
train_loader = DenseDataLoader(train_dataset, batch_size=20)
# print('train_loader:{}'.format(train_loader))


class GNN(torch.nn.Module):
    def __init__(self, in_channels, hidden_channels, out_channels,
                 normalize=False, lin=True):
        super().__init__()

        self.conv1 = DenseSAGEConv(in_channels, hidden_channels, normalize)
        self.bn1 = torch.nn.BatchNorm1d(hidden_channels)
        self.conv2 = DenseSAGEConv(hidden_channels, hidden_channels, normalize)
        self.bn2 = torch.nn.BatchNorm1d(hidden_channels)
        self.conv3 = DenseSAGEConv(hidden_channels, out_channels, normalize)
        self.bn3 = torch.nn.BatchNorm1d(out_channels)

        if lin is True:
            self.lin = torch.nn.Linear(2 * hidden_channels + out_channels,
                                       out_channels)
        else:
            self.lin = None

    def bn(self, i, x):
        batch_size, num_nodes, num_channels = x.size()

        x = x.view(-1, num_channels)
        x = getattr(self, f'bn{i}')(x)
        x = x.view(batch_size, num_nodes, num_channels)
        return x

    def forward(self, x, adj, mask=None):
        batch_size, num_nodes, in_channels = x.size()

        x0 = x
        x1 = self.bn(1, self.conv1(x0, adj, mask).relu())
        x2 = self.bn(2, self.conv2(x1, adj, mask).relu())
        x3 = self.bn(3, self.conv3(x2, adj, mask).relu())

        x = torch.cat([x1, x2, x3], dim=-1)

        if self.lin is not None:
            x = self.lin(x).relu()

        return x


class Net(torch.nn.Module):
    def __init__(self):
        super().__init__()

        num_nodes = ceil(0.25 * max_nodes)
        # print(dataset.num_features)
        # print(num_nodes)
        self.gnn1_pool = GNN(dataset.num_features, 64, num_nodes)
        self.gnn1_embed = GNN(dataset.num_features, 64, 64, lin=False)

        num_nodes = ceil(0.25 * num_nodes)
        self.gnn2_pool = GNN(3 * 64, 64, num_nodes)
        self.gnn2_embed = GNN(3 * 64, 64, 64, lin=False)

        self.gnn3_embed = GNN(3 * 64, 64, 64, lin=False)

        self.lin1 = torch.nn.Linear(3 * 64, 64)
        self.lin2 = torch.nn.Linear(64, dataset.num_classes)

    def forward(self, x, adj, mask=None):
        s = self.gnn1_pool(x, adj, mask)
        x = self.gnn1_embed(x, adj, mask)
        # print('pool1\ns.shape:{}\nx.shape:{}'.format(s.shape, x.shape))
        """
        x^(l+1) = s^(l^T) x^l 
        adj^(l+1) = s^(l^T) adj^l s^l 
        """
        x, adj, l1, e1 = dense_diff_pool(x, adj, s, mask)
        # print('diffpool1\nx.shape{}'.format(x.shape))
        s = self.gnn2_pool(x, adj)
        x = self.gnn2_embed(x, adj)
        # print('pool2\ns.shape:{}\nx.shape:{}'.format(s.shape, x.shape))
        x, adj, l2, e2 = dense_diff_pool(x, adj, s)
        # print('diffpoll2\nx.shape:{}'.format(x.shape))
        x = self.gnn3_embed(x, adj)
        # print('embed3\nx.shape:{}'.format(x.shape))
        x = x.mean(dim=1)
        # print('mean\nx.shape:{}'.format(x.shape))
        x = self.lin1(x).relu()
        # print('lin1\nx.shape:{}'.format(x.shape))
        x = self.lin2(x)
        # print('lin2\nx.shape:{}'.format(x.shape))
        return F.log_softmax(x, dim=-1), l1 + l2, e1 + e2


device = torch.device('cuda:1' if torch.cuda.is_available() else 'cpu')
model = Net().to(device)
optimizer = torch.optim.Adam(model.parameters(), lr=0.001)


def train(epoch):
    model.train()
    loss_all = 0
    print("-----------------")
    for data in train_loader:
        data = data.to(device)
        optimizer.zero_grad()
        # print('data:\n{}'.format(data))
        # print('data.x:\n{}'.format(data.x))
        # print('data.adj:\n{}'.format(data.adj))
        # print('data.mask:\n{}'.format(data.mask))
        # print('data.y:\n{}'.format(data.y))
        output, _, _ = model(data.x, data.adj, data.mask)
        # print(output.type())
        # print(data.y.view(-1).type())
        loss = F.nll_loss(output, data.y.view(-1))
        loss.backward()
        loss_all += data.y.size(0) * float(loss)
        optimizer.step()
        # exit()
    print("end------------")
    return loss_all / len(train_dataset)


@torch.no_grad()
def test(loader):
    model.eval()
    correct = 0

    for data in loader:
        data = data.to(device)
        pred = model(data.x, data.adj, data.mask)[0].max(dim=1)[1]
        correct += int(pred.eq(data.y.view(-1)).sum())
    return correct / len(loader.dataset)


best_val_acc = test_acc = 0
for epoch in range(1, 151):
    train_loss = train(epoch)
    val_acc = test(val_loader)
    if val_acc > best_val_acc:
        test_acc = test(test_loader)
        best_val_acc = val_acc
    print(f'Epoch: {epoch:03d}, Train Loss: {train_loss:.4f}, '
          f'Val Acc: {val_acc:.4f}, Test Acc: {test_acc:.4f}')
