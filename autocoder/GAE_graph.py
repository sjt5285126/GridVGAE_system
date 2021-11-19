import torch
import torch.nn as nn
import torch_geometric.nn as gnn
import torch_geometric.loader as gloader
import dataset
import torch.nn.functional as F
# 问题在于

class GAE_G(nn.Module):
    def __init__(self,in_channels,out_channels):
        super(GAE_G, self).__init__()
        # 沿用GAE的思路来做图级别
        self.encoder1 = gnn.Sequential('x,edge_index',[
            (gnn.GraphConv(in_channels,2*out_channels),'x,edge_index -> x'),
            nn.ReLU(),
            (gnn.GraphConv(2*out_channels,out_channels),'x,edge_index -> x'),
            nn.ReLU()
        ])
        self.conv1 = nn.Conv1d(in_channels=out_channels, out_channels=32, kernel_size=3, padding=1)
        self.pool = nn.MaxPool1d(kernel_size=4,return_indices=True)
        self.conv2 = nn.Conv1d(in_channels=32,out_channels=64,kernel_size=3,padding=1)
        self.conv3 = nn.Conv1d(in_channels=64,out_channels=64,kernel_size=3,padding=1)
        self.unConv1 = nn.ConvTranspose1d(in_channels=64,out_channels=64,kernel_size=3,padding=1)
        self.unConv2 = nn.ConvTranspose1d(in_channels=64, out_channels=64, kernel_size=3, padding=1)
        self.unpool = nn.MaxUnpool1d(kernel_size=4)
        self.unConv3 = nn.ConvTranspose1d(in_channels=64,out_channels=32,kernel_size=3,padding=1)
        self.unConv4 = nn.ConvTranspose1d(in_channels=32,out_channels=out_channels,kernel_size=3,padding=1)
        self.lin1 = nn.Sequential(
            nn.Flatten(start_dim=1),
            nn.Linear(1024, 512),
            nn.ReLU(),
            nn.Linear(512, 256),
            nn.ReLU(),
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Linear(128, 2)
        )
        self.lin2 = nn.Sequential(
            nn.Linear(2,128),
            nn.ReLU(),
            nn.Linear(128,256),
            nn.ReLU(),
            nn.Linear(256,512),
            nn.ReLU(),
            nn.Linear(512,1024),
            nn.Unflatten(1,(64,16))
        )
        # 思路没问题，但是代码编写仍存在一些毛病
        self.decoder2 = gnn.InnerProductDecoder()
        self.relu = nn.ReLU()
        self.unflatten = nn.Unflatten(0,(5,1024))
        self.flatten = nn.Flatten(0,1)
    def encoder2(self,x)->(torch.Tensor,list):
        index = []
        x = self.conv1(x)
        x,indices = self.pool(x)
        index.append(indices)
        x = self.conv2(x)
        x,indices = self.pool(x)
        index.append(indices)
        x = self.conv3(x)
        x,indices = self.pool(x)
        index.append(indices)
        x = self.lin1(x)
        return x,index

    def decoder1(self,x:torch.Tensor,index:list):
        x = self.lin2(x)
        x = self.unConv1(x)
        x = self.unpool(x,index[2])
        x = self.unConv2(x)
        x = self.unpool(x,index[1])
        x = self.unConv3(x)
        x = self.unpool(x,index[0])
        x = self.unConv4(x)
        x = torch.transpose(x,1,2)
        x = self.flatten(x)
        return x
    def forward(self,x,edge_index):
        x = self.encoder1(x,edge_index)
        x = self.unflatten(x)
        x = torch.transpose(x,1,2)
        x,index = self.encoder2(x)
        x = self.decoder1(x,index)
        x = self.decoder2(x,edge_index)
        return x



data = dataset.init_data(32,dataset.init_p(5),5)

device = torch.device('cpu')
model = GAE_G(1,2).to(device)

batchs = gloader.DataLoader(data,batch_size=5,shuffle=True)

for batch in batchs:
    print(batch.x.shape)
    z = model(batch.x, batch.edge_index)
    print(z.shape)



