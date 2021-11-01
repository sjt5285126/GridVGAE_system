import os.path as osp
import argparse

import torch
import torch.nn.functional as F
# 导入引文网络数据集,节点代表文档,边代表引文链接
from torch_geometric.datasets import Planetoid
import torch_geometric.transforms as T
from torch_geometric.nn import GCNConv, ChebConv  # noqa

# 编写一个命令行接口，可以使用 --user_gdc命令
parser = argparse.ArgumentParser()
# action命名参数指定了这个命令行参数应当如何处理.
parser.add_argument('--use_gdc', action='store_true',
                    help='Use GDC preprocessing.')
args = parser.parse_args()

dataset = 'Cora'
# join函数 表示将目录和文件名合成一个路径.
# __file__ 得到当前文件的源文件路径
path = osp.join(osp.dirname(osp.realpath(__file__)),  'data', dataset)
# 得到我们的数据集并将其归一化
dataset = Planetoid(path, dataset, transform=T.NormalizeFeatures())
'''
处理之后的数据可以通过属性 .data 获得，它将返回一个数据对象，包括如下几部分：
* x: 节点的特征，维度为 2708 * 1433，类型为 np.ndarray
* y: 节点的标签，总共包括7个类别，类型为 np.ndarray
* adjacency: 邻接矩阵，维度为 2708 * 2708，类型为 scipy.sparse.coo.coo_matrix
* train_mask: 训练集掩码向量，维度为 2708，当节点属于训练集时，相应位置为True，否则False
* val_mask: 验证集掩码向量，维度为 2708，当节点属于验证集时，相应位置为True，否则False
* test_mask: 测试集掩码向量，维度为 2708，当节点属于测试集时，相应位置为True，否则False
'''
data = dataset[0]

# 预处理与模型定义
if args.use_gdc:
    '''
    PyG提供了一个名为GDC(Graph diffusion convolution , 图扩散卷积)的图数据预处理方法，
    其结合了message passing和spectral methods优点，可以减少图中噪音的影响，
    可以在有监督和无监督任务的各种模型以及各种数据集上显着提高性能，并且GDC不仅限于GNN，
    还可以与任何基于图的模型或算法（例如频谱聚类）轻松组合，
    而无需对其进行任何更改或影响其计算复杂性
    '''
    # 通过图扩散卷积(GDC)对图进行处理
    gdc = T.GDC(self_loop_weight=1, normalization_in='sym',
                normalization_out='col',
                diffusion_kwargs=dict(method='ppr', alpha=0.05),
                sparsification_kwargs=dict(method='topk', k=128,
                                           dim=0), exact=True)
    data = gdc(data)


class Net(torch.nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        # GCNConv模型输出参数
        # 输入节点维度特征,输出节点维度特征,是否cahced,是否normalize
        '''
        cache参数会缓存模型第一层中的GCN传播公式中正则化邻接矩阵的值
        GCN是转导学习模型, GAT和GraphSAGE属于归纳模型
        '''
        self.conv1 = GCNConv(dataset.num_features, 16, cached=True,
                             normalize=not args.use_gdc)
        self.conv2 = GCNConv(16, dataset.num_classes, cached=True,
                             normalize=not args.use_gdc)

    def forward(self):
        x, edge_index, edge_weight = data.x, data.edge_index, data.edge_attr
        x = F.relu(self.conv1(x, edge_index, edge_weight))
        x = F.dropout(x, training=self.training)  # dropout操作,避免过拟合
        x = self.conv2(x, edge_index, edge_weight)
        # softmax 是一个非线性的函数，它接受了一个实数向量并返回了一个概率分布
        return F.log_softmax(x, dim=1) # 模型最后一层接上一个softmax


# 模型训练与评估
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model, data = Net().to(device), data.to(device)
# ? 什么是L2正则
# 对于不同的参数,我们用不同的optimizer实例,第一层模型参数需要权重衰减(L2正则).第二层模型参数不需要
# GCN论文中做了如下要求
optimizer = torch.optim.Adam([
    dict(params=model.conv1.parameters(), weight_decay=5e-4),
    dict(params=model.conv2.parameters(), weight_decay=0)
], lr=0.01)  # Only perform weight-decay on first convolution.


def train():
    model.train() # 模型处于train模式
    optimizer.zero_grad()
    F.nll_loss(model()[data.train_mask], data.y[data.train_mask]).backward()
    optimizer.step()


# torch.no_grad()作为一个装饰器，
# 被该语句wrap起来的部分将不会追踪梯度。
# 可以节约显存和内存
@torch.no_grad()
def test():
    model.eval() # 指明模型处于eval模式，因为模型存在dropout层，因此这一步是必须的
    logits, accs = model(), []
    for _, mask in data('train_mask', 'val_mask', 'test_mask'):
        pred = logits[mask].max(1)[1]
        acc = pred.eq(data.y[mask]).sum().item() / mask.sum().item()
        accs.append(acc)
    return accs


best_val_acc = test_acc = 0
for epoch in range(1, 201):
    train()
    train_acc, val_acc, tmp_test_acc = test()
    if val_acc > best_val_acc:
        best_val_acc = val_acc
        test_acc = tmp_test_acc
    log = 'Epoch: {:03d}, Train: {:.4f}, Val: {:.4f}, Test: {:.4f}'
    print(log.format(epoch, train_acc, best_val_acc, test_acc))
