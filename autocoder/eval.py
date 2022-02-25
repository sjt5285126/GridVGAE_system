import torch
from VGAE_spin import EncoderSpin, DecoderSpin, SVGAE
import dataset
from dataset import reshapeIsing_MSE
import pickle
import torch_geometric.loader as gloader
import h5py

# 我们在知道mu与log的情况下，可以通过model的reparametrize()函数来得到z,再将z进行decode解码

def load_checkpoint(model, checkpoint_PATH, optimizer):
    if checkpoint_PATH != None:
        model_CKPT = torch.load(checkpoint_PATH)
        model.load_state_dict(model_CKPT['state_dict'], False)
        print("mu:\n{}".format(model_CKPT['mu']))
        print("log:\n{}".format(model_CKPT['log']))
        print('loading checkpoint!')
        optimizer.load_state_dict(model_CKPT['optimizer'])
        mu = model_CKPT['mu']
        log = model_CKPT['log']
    # 返回模型，优化器
    return model, optimizer,mu,log

device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
model = SVGAE(EncoderSpin(), DecoderSpin()).to(device)
optim = torch.optim.Adam(model.parameters(), lr=0.01)
PATH = 'model_16_220224.pkl'

checkpoint = torch.load(PATH)

model, optim,mu,log = load_checkpoint(model, PATH, optim)

def reparametrize(mu,log):
    # 返回重采样的z
    return mu + torch.randn_like(log) + torch.exp(log)

epochs = 1000

for epoch in range(epochs):
    model.eval()
    z = reparametrize(mu,log)
    print(z.shape)