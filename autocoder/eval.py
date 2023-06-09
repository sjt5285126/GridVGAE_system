import torch
from VGAE_spin_GAT import EncoderSpin, DecoderSpin, SVGAE
import dataset
from dataset import calculate, acc_loss
from dataset import reshapeIsing_MSE, reshapeIsingHdf5, reshapeIsing
import pickle
import torch_geometric.loader as gloader
import h5py

# 我们在知道mu与log的情况下，可以通过model的reparametrize()函数来得到z,再将z进行decode解码

# 　增加在训练中观察训练出的模型相似度的代码
'''
想法1:
    1. acc = func(x_ - x) / nums_node  # 相当于是误差率
        准确率为 1 - func(x_ - x) / nums_node
    example:
        func(x): return sum(abs(x))
    2. 将得到的结果进行归一化 mean(acc)
    3. print('准确率{}'.format(acc * 100))
'''


# 加载模型的函数
def load_checkpoint(model, checkpoint_PATH, optimizer):
    if checkpoint_PATH != None:
        model_CKPT = torch.load(checkpoint_PATH, map_location=device)
        model.load_state_dict(model_CKPT['state_dict'], False)
        print(model_CKPT['mu'].shape)
        print(model_CKPT['datalog'].shape)
        print("mu:\n{}".format(model_CKPT['mu']))
        print("datalog:\n{}".format(model_CKPT['datalog']))
        print('loading checkpoint!')
        optimizer.load_state_dict(model_CKPT['optimizer'])
        mu = model_CKPT['mu']
        log = model_CKPT['datalog']
        batch_size = model_CKPT['batch_size']
    # 返回模型，优化器
    return model, optimizer, mu, log, batch_size


# 定义测试所需要的设备，模型，优化器
device = torch.device('cuda:3' if torch.cuda.is_available() else 'cpu')
model = SVGAE(EncoderSpin(), DecoderSpin()).to(device)
optim = torch.optim.Adam(model.parameters(), lr=0.01)
PATH = 'model/modelGAT_32_T_PTP.pkl'
name = PATH.split('/')[-1][:-4]

checkpoint = torch.load(PATH, map_location=device)
model, optim, mu, log, batch_size = load_checkpoint(model, PATH, optim)
print(batch_size)


def reparametrize(mu, log):
    # 返回重采样的z
    return mu + torch.randn_like(log) + torch.exp(log)


# 归一化计算  (f - f.mean()) / f.std()
# testData = h5py.File('16evalFeatures.hdf5','r')

'''
testTotalM = testData['TotalM'][:]
testTotalE = testData['TotalE'][:]
testAvrM = testData['AvrM'][:]
testAvrE = testData['AvrE'][:]

'''

epochs = 1

with torch.no_grad():
    model.eval()
    for epoch in range(epochs):
        f_gen = h5py.File('data/modelData/{}_config_{}.hdf5'.format(name, epoch), 'w')
        model.eval()
        z = reparametrize(mu, log)
        x_ = model.decode(z)
        configs = reshapeIsing_MSE(x_, batch_size)
        print(configs.shape)
        evalTotalM, evalTotalE, evalAvrM, evalAvrE = calculate(configs)
        f_gen['TotalM'] = evalTotalM
        f_gen['TotalE'] = evalTotalE
        f_gen['AvrM'] = evalAvrM
        f_gen['AvrE'] = evalAvrE
        f_gen.close()

print("hello world")

# 得到configs
