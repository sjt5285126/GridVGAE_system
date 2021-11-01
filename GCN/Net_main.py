import torch
import torch.nn as nn
import Net
from torch_geometric.data import Data


device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model = Net.Net().to(device)


y_list = [1,1,0,0]

x_list = [[[-1],[-1]],[[1],[1]],[[1],[-1]],[[-1],[1]]]

edge_index_test = torch.tensor([
    [1,2],
    [2,1]
],dtype=torch.long)

data = []
for x,y in zip(x_list,y_list):
    y = torch.tensor([y])
    x = torch.tensor(x,dtype=torch.float)
    data.append(Data(x=x,edge_index=edge_index_test,y=y))


optimizer = torch.optim.Adam(model.parameters(), lr=0.01, weight_decay=5e-4)

model.train()
for epoch in range(200):
    for i in range(4):
        data[i] =data[i].to(device)
        out = model(data[i])
        optimizer.zero_grad()
        loss =  nn.MSELoss(out,data[i].y)
        loss.backward()
        optimizer.step()


# 生成测试集
data = Data(
    x = torch.tensor(
        [-1],
        [-1],dtype=torch.float
    ),
    edge_index = torch.tensor(
        [1,2],
        [2,1],
        dtype= torch.long
    ),
    y = torch.tensor([1])
)

#  进行测试
model.eval()
pred = model(data)
correct = pred-1
acc = int(correct) / int(data.test_mask.sum())
print('Accuracy: {:.4f}'.format(acc))


