'''
pre3 是 在VGAE_Origin3的基础上更新的 相较于2  增加了卷积层 因为发现2在64尺寸的生成上效果太差,所以进行改变
'''

import pickle
from sys import argv
import torch_geometric.loader as gloader
import torch

from VGAE_IsingXY_new import SVGAE, EncoderSpin, DecoderSpin

if len(argv) < 3:
    print("please input: python model_init_pre.py epochs name")
    exit()

epochs = int(argv[1])
name = argv[2]

# 先对单温度单尺寸进行训练,多温度单尺寸进行训练

# 定义设备
device = torch.device('cuda:1' if torch.cuda.is_available() else 'cpu')
model = SVGAE(EncoderSpin(), DecoderSpin()).to(device)
print(model)

# 　读取数据文件1
datafile = open('data/IsingXYGraph/dataXY_IsingXYA0.32L16_0711.pkl', 'rb')
data = pickle.load(datafile)
datafile.close()
# # # 读取数据文件2
# datafile2 = open('data/IsingGraph/data_32_T_PTP.pkl', 'rb')
# data2 = pickle.load(datafile2)
# datafile2.close()
# data.extend(data2[:1000])
# # 读取数据文件3
# datafile3 = open('data/IsingGraph/data_16_PTP.pkl', 'rb')
# data3 = pickle.load(datafile3)
# datafile3.close()
# data.extend(data3[:1000])
batch_size = 2000
data_train_batchs = gloader.DataLoader(data, batch_size=batch_size, shuffle=True)
optim = torch.optim.Adam(model.parameters(), lr=0.01, weight_decay=0.001)

lossMIN = 9999999

for epoch in range(epochs):
    model.train()
    print("epoch:{}".format(epoch))
    for d in data_train_batchs:
        d = d.to(device)
        d.x = d.x.float()
        z = model.encode(d.x, d.edge_index, d.batch)
        x_ = model.decode(z)
        loss = model.recon_loss(d.x, x_) + model.kl_loss() / (16 * (d.num_nodes / batch_size))
        lossMIN = lossMIN if loss > lossMIN else loss
        print('loss:{}'.format(loss))
        optim.zero_grad()
        loss.backward()
        optim.step()

# 保存模型
mu, log = model.get_mu_logstd()
torch.save({'epoch': epochs, 'state_dict': model.state_dict(), 'best_loss': lossMIN,
            'optimizer': optim.state_dict(), 'batch_size': batch_size, 'mu': mu, 'datalog': log}, '{}.pkl'.format(name))
