import torch
import torch.nn as nn
import torch.nn.functional as F
from GCN_graphclass import Net
from torch_geometric.data import Data
import torch_geometric.data as gdata

device = torch.device('cpu')
model = Net().to(device)


y_list = [[1],[1],[0],[0],[1],[1],[0],[0]]

x_list = [[[-1],[-1]],[[1],[1]],[[1],[-1]],[[-1],[1]],[[-1],[-1]],[[1],[1]],[[1],[-1]],[[-1],[1]]]

edge_index_test = torch.tensor([
    [0,1],
    [1,0]
],dtype=torch.long)

data = []
for x,y in zip(x_list,y_list):
    y = torch.tensor(y)
    x = torch.tensor(x,dtype=torch.float)
    data.append(Data(x=x,edge_index=edge_index_test,y=y))

data_train = data[0:4]
data_test = data[4:8]
data_trainloader = gdata.DataLoader(data_train,batch_size=2,shuffle=True)
data_testloader = gdata.DataLoader(data_test,batch_size=2,shuffle=True)

optimizer = torch.optim.Adam(model.parameters(), lr=0.01, weight_decay=5e-4)


print(model)
for epoch in range(100):
    for one_train_batch in data_trainloader:
        model.train()
        optimizer.zero_grad()
        one_train_batch = one_train_batch.to(device)
        logit = model(one_train_batch.x, one_train_batch.edge_index, one_train_batch.batch)
        loss = F.nll_loss(logit, one_train_batch.y)
        loss.backward()
        optimizer.step()
    for one_test_batch in data_testloader:
        model.eval()
        one_test_batch = one_test_batch.to(device)
        pred = model(one_test_batch.x, one_test_batch.edge_index, one_test_batch.batch)
        pred = pred.max(1)[1]
        acc = pred.eq(one_test_batch.y).sum().item() / len(one_test_batch.y)
    print("epoch: {}\t, loss: {:.4f}, test_acc: {:.4f}".format(epoch, loss, acc))



#  进行测试
#model.eval()
#pred = model(data)
#correct = pred-1
#print(correct)
#acc = int(correct) / int(data.test_mask.sum())
#print('Accuracy: {:.4f}'.format(acc))


