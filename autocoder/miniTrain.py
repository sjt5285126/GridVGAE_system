import h5py
import torch
from VGAE_spin_origin2 import EncoderSpin, DecoderSpin, SVGAE
from dataset import reshapeIsing_MSE, acc, reshapeIsing, reshapeTorch, calculate
import pickle
import torch_geometric.loader as gloader
import time

# 定义设备
device = torch.device('cuda:3' if torch.cuda.is_available() else 'cpu')

# 再次小批量训练一组数据

datafile = open('data/IsingGraph/data_32_T_PTP.pkl', 'rb')
data = pickle.load(datafile)
batch_size = 500
data = data[:batch_size]
data_train_batchs = gloader.DataLoader(data, batch_size=batch_size)


# 加载模型
def load_checkpoint(model, checkpoint_PATH, optimizer):
    if checkpoint_PATH != None:
        model_CKPT = torch.load(checkpoint_PATH, map_location=device)
        model.load_state_dict(model_CKPT['state_dict'], False)
        print("mu:\n{}".format(model_CKPT['mu']))
        print("log:\n{}".format(model_CKPT['log']))
        print('loading checkpoint!')
        optimizer.load_state_dict(model_CKPT['optimizer'])
    # 返回模型，优化器
    return model, optimizer


# 模型地址
PATH = 'model_pre2_16mix32_PTP_0617.pkl'

# 加载模型
model = SVGAE(EncoderSpin(), DecoderSpin()).to(device)
optim = torch.optim.Adam(model.parameters(), lr=0.01, weight_decay=0.001)  # 增加L2正则化
model, optim = load_checkpoint(model, PATH, optim)

# 小批量训练
epochs = 5

for epoch in range(epochs):
    model.train()
    for d in data_train_batchs:
        d = d.to(device)
        d.x = d.x.float()
        z = model.encode(d.x, d.edge_index, d.edge_attr, d.batch)
        x_ = model.decode(z)
        loss = model.recon_loss(d.x, x_, ) + model.kl_loss() / (4 * (d.num_nodes / batch_size))
        preConfig = reshapeIsing_MSE(d.x, batch_size)
        afterConfig = reshapeIsing_MSE(x_, batch_size)
        print("acc:{}%".format(acc(preConfig, afterConfig)))
        optim.zero_grad()
        loss.backward()
        optim.step()

# 获取mu与logstd
mu, log = model.get_mu_logstd()


# 重采样
def reparametrize(mu, log):
    # 返回重采样的z
    return mu + torch.randn_like(log) + torch.exp(log)

def initData(path,epochs):
    TotalM = []
    TotalE = []
    AvrM = []
    AvrE = []
    model.eval()
    for e in range(epochs):
        z = reparametrize(mu,log)
        x_ = model.decode(z)
        configs = reshapeIsing_MSE(x_,batch_size)
        print(configs.shape)
        evalTotalM, evalTotalE, evalAvrM, evalAvrE = calculate(configs)
        TotalE.extend(list(evalTotalE))
        TotalM.extend(list(evalTotalM))
        AvrE.extend(list(evalAvrE))
        AvrM.extend(list(evalAvrM))
    f_gen = h5py.File(path,'w')
    print(len(AvrM))
    f_gen['TotalM'] = TotalM
    f_gen['TotalE'] = TotalE
    f_gen['AvrM'] = AvrM
    f_gen['AvrE'] = AvrE
    f_gen.close()
    print("complete")
# 生成数据
# f_gen = h5py.File('T_PTP_mix64_32_16_64.hdf5', 'w')
# model.eval()
# z = reparametrize(mu, log)
# x_ = model.decode(z)
# configs = reshapeIsing_MSE(x_, batch_size)
# print(configs.shape)
# evalTotalM, evalTotalE, evalAvrM, evalAvrE = calculate(configs)
# f_gen['TotalM'] = evalTotalM
# f_gen['TotalE'] = evalTotalE
# f_gen['AvrM'] = evalAvrM
# f_gen['AvrE'] = evalAvrE
# f_gen.close()
#
# print('complete')

initData('16mix32_32_PTP_0617.hdf5',1)
