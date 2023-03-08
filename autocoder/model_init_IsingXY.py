import pickle
from sys import argv
import torch_geometric.loader as gloader
import torch

from VGAE_spin_origin2 import SVGAE, EncoderSpin, DecoderSpin

if len(argv) < 3:
    print("please input: python model_init_IsingXY.py epochs name")
    exit()

epochs = int(argv[1])
name = argv[2]

device1 = torch.device('cuda:1' if torch.cuda.is_available() else 'cpu')
device2 = torch.device('cuda:2' if torch.cuda.is_available() else 'cpu')
model_Ising = SVGAE(EncoderSpin(), DecoderSpin()).to(device1)
model_XY = SVGAE(EncoderSpin(), DecoderSpin()).to(device2)

# load Ising
datafile = open('data/IsingXYGraph/dataIsing_IsingXYA0.32L8_XYattr.pkl', 'rb')
data_Ising = pickle.load(datafile)
datafile.close()

# load XY
datafile = open('data/IsingXYGraph/dataXY_IsingXYA0.32L8_XYattr.pkl', 'rb')
data_XY = pickle.load(datafile)
datafile.close()

batch_size = 5000
batchs_Ising = gloader.DataLoader(data_Ising, batch_size=batch_size, shuffle=True)
optim_Ising = torch.optim.Adam(model_Ising.parameters(), lr=0.01, weight_decay=0.001)

batchs_XY = gloader.DataLoader(data_XY, batch_size=batch_size, shuffle=True)
optim_XY = torch.optim.Adam(model_XY.parameters(), lr=0.01, weight_decay=0.001)

for epoch in range(epochs):
    model_Ising.train()
    model_XY.train()
    print("epoch:{}".format(epoch))
    for ising, xy in zip(batchs_Ising, batchs_XY):
        ising = ising.to(device1)
        print(ising.x.shape)
        ising.x = ising.x.float()
        # print(ising.num_nodes)
        z_ising = model_Ising.encode(ising.x, ising.edge_index, ising.edge_attr, ising.batch)
        x_ = model_Ising.decode(z_ising)
        loss_Ising = model_Ising.recon_loss(ising.x, x_) + model_Ising.kl_loss() / (16 * (ising.num_nodes / batch_size))
        print("Ising_loss:{}".format(loss_Ising))
        optim_Ising.zero_grad()
        loss_Ising.backward()
        optim_Ising.step()
        xy = xy.to(device2)
        print(xy.x.shape)
        z_xy = model_XY.encode(xy.x, xy.edge_index, xy.edge_attr, xy.batch)
        x_ = model_XY.decode(z_xy)
        loss_xy = model_XY.recon_loss(xy.x, x_) + model_XY.kl_loss() / (64 * (xy.num_nodes / batch_size))
        print("XY_loss:{}".format(loss_xy))
        optim_XY.zero_grad()
        loss_xy.backward()
        optim_XY.step()

Ising_mu, Ising_log = model_Ising.get_mu_logstd()

XY_mu, XY_log = model_XY.get_mu_logstd()

torch.save({'epoch': epochs, 'state_dict_ising': model_Ising.state_dict(), 'optim_ising': optim_Ising.state_dict(),
            'state_dict': model_XY.state_dict(), 'optim_xy': optim_XY.state_dict(), 'batch_size': batch_size,
            'mu_ising': Ising_mu, 'log_ising': Ising_log, 'mu_xy': XY_mu, 'log_xy': XY_log}, 'model/{}.pkl'.format(name))
