import pickle
from sys import argv
import torch_geometric.loader as gloader
import torch

from VGAE_spin import SVGAE, EncoderSpin, DecoderSpin

if len(argv) < 3:
    print("please input: python model_init.py epochs name")
    exit()

epochs = int(argv[1])
name = argv[2]

# 先对单温度单尺寸进行训练,多温度单尺寸进行训练

# 定义设备
device = torch.device('cuda:1' if torch.cuda.is_available() else 'cpu')
model = SVGAE(EncoderSpin(), DecoderSpin()).to(device)
print(model)

# 　读取数据文件
datafile = open('data/IsingGraph/data16.pkl', 'rb')
data = pickle.load(datafile)
datafile.close()
# 读取温度在2.25的构型
batch_size = 5000
data_train_batchs = gloader.DataLoader(data, batch_size=batch_size)
optim = torch.optim.Adam(model.parameters(), lr=0.01, weight_decay=0.001)

lossMIN = 9999999

for epoch in range(epochs):
    model.train()
    for d in data_train_batchs:
        d = d.to(device)
        d.x = d.x.float()
        z = model.encode(d.x, d.edge_index, d.edge_attr, d.batch)
        x_ = model.decode(z)
        loss = model.recon_loss(d.x, x_) + model.kl_loss()
        lossMIN = lossMIN if loss > lossMIN else loss
        print('loss:{}'.format(loss))
        optim.zero_grad()
        loss.backward()
        optim.step()

# 保存模型
mu, log = model.get_mu_logstd()
torch.save({'epoch': epochs, 'state_dict': model.state_dict(), 'best_loss': lossMIN,
            'optimizer': optim.state_dict(), 'batch_size': batch_size, 'mu': mu, 'datalog': log}, '{}.pkl'.format(name))
