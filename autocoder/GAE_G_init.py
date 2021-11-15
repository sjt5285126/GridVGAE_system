import torch
import torch.nn as nn
import torch_geometric.nn as gnn
import torch_geometric.loader as gloader
import torch_geometric.data as gdata
from GAE_G import gae
import dataset
channels = 4

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

model = gae(1,8).to(device)
data = dataset.init_data(4,dataset.init_p(10),100)

dataTrain = data[0:100]
dataTest = data[100:1000]
data_trainloader = gloader.DataLoader(dataTrain, batch_size=10, shuffle=True)
#print(data_trainloader)
data_testloader = gloader.DataLoader(dataTest, batch_size=10, shuffle=True)
#print(data_testloader)

print(model.parameters())
optimizer = torch.optim.Adam(model.parameters(),lr=0.01)

cost = nn.MSELoss()

print(model)

for epoch in range(50):
    for train_batch in data_trainloader:
        model.train()
        optimizer.zero_grad()
        train_batch = train_batch.to(device)
        x,x_ = model(train_batch.x,train_batch.edge_index,train_batch.batch)
        loss = cost(x,x_)
        loss.backward()
        optimizer.step()

    for test_batch in data_testloader:
        model.eval()
        test_batch = test_batch.to(device)
        with torch.no_grad():
            x,x_ = model(test_batch.x,test_batch.edge_index,test_batch.batch)
            loss = cost(x,x_)

    print("epoch: {}\t, loss: {:.4f}".format(epoch, loss))

torch.save(model.state_dict(),'GAE_G.pkl')
