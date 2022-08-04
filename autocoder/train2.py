import torch
from VGAE_spin_origin2 import EncoderSpin, DecoderSpin, SVGAE
from dataset import reshapeIsing_MSE, acc, acc_loss
import pickle
import torch_geometric.loader as gloader
import time
import matplotlib.pyplot as plt
import matplotlib as mpl


# 画出Ising构型图
def plotIsing(config, filename):
    config = config.cpu().numpy()
    print(config)
    plt.xticks([])
    plt.yticks([])
    plt.imshow(config, cmap='gray')
    plt.savefig('image/{}.eps'.format(filename))
    plt.savefig('image/{}.png'.format(filename))


# 导入model
def load_checkpoint(model, checkpoint_PATH, optimizer):
    if checkpoint_PATH != None:
        model_CKPT = torch.load(checkpoint_PATH, map_location=device)
        model.load_state_dict(model_CKPT['state_dict'], False)
        print("mu:\n{}".format(model_CKPT['mu']))
        print("log:\n{}".format(model_CKPT['log']))
        print('loading checkpoint!')
        optimizer.load_state_dict(model_CKPT['optimizer'])
    # 返回模型，优化器
    return model, optimizer, model_CKPT['batch_size']


# 构建模型

# 读取数据
device = torch.device('cuda:1' if torch.cuda.is_available() else 'cpu')
datafile = open('data/IsingGraph/data_16_T_PTPQuick.pkl', 'rb')
data = pickle.load(datafile)
test_batch = gloader.DataLoader(data, batch_size=200, shuffle=True)
datafile.close()
model = SVGAE(EncoderSpin(), DecoderSpin()).to(device)
optim = torch.optim.Adam(model.parameters(), lr=0.01)

# PATH = argv[1]

PATH = 'model_pre2_16_PTP_0706True2withPkl.pkl'

checkpoint = torch.load(PATH, map_location=device)
# 模型的测试
model, optim, batch_size = load_checkpoint(model, PATH, optim)

trigger = False

for epoch in range(1000):
    model.eval()
    with torch.no_grad():
        for batch in test_batch:
            batch = batch.to(device)
            z = model.encode(batch.x, batch.edge_index, batch.edge_attr, batch.batch)
            # z = model.encode(batch.x, batch.edge_index,batch.batch)
            x_ = model.decode(z)
            print("loss:{}".format(model.recon_loss(batch.x, x_, ) + model.kl_loss()))
            preConfig = reshapeIsing_MSE(batch.x, 200)
            afterConfig = reshapeIsing_MSE(x_, 200)
            if trigger:
                plotIsing(preConfig[0], 'MCMC16T3')
                plotIsing(afterConfig[0], 'GEN16T3')
            trigger = False
            # print("测试构型:{}".format(reshapeIsing_MSE(batch.x, 2)))
            # print("重构后的构型:{}".format(reshapeIsing_MSE(x_, 2)))
            print("acc:{}%".format(acc(preConfig, afterConfig)))
            # print("acc_totalM:{}, acc_totalE:{}, acc_AvrM:{}, acc_AvrE:{}".format(acc_loss(preConfig,afterConfig)))
            # print(acc_loss(preConfig,afterConfig))
            # print(acc_loss(preConfig, afterConfig))
            time.sleep(10)
