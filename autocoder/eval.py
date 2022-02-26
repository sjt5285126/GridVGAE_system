import torch
from VGAE_spin import EncoderSpin, DecoderSpin, SVGAE
import dataset
from dataset import calculate
from dataset import reshapeIsing_MSE
import pickle
import torch_geometric.loader as gloader
import h5py

# 我们在知道mu与log的情况下，可以通过model的reparametrize()函数来得到z,再将z进行decode解码

#　增加在训练中观察训练出的模型相似度的代码
'''
想法1:
    1. acc = func(x_ - x) / nums_node  # 相当于是误差率
        准确率为 1 - func(x_ - x) / nums_node
    example:
        func(x): return sum(abs(x))
    2. 将得到的结果进行归一化 mean(acc)
    3. print('准确率{}'.format(acc * 100))
'''
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
        batch_size = model_CKPT['batch_size']
    # 返回模型，优化器
    return model, optimizer,mu,log,batch_size

device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
model = SVGAE(EncoderSpin(), DecoderSpin()).to(device)
optim = torch.optim.Adam(model.parameters(), lr=0.01)
PATH = 'model_16_220224.pkl'

checkpoint = torch.load(PATH)

model, optim,mu,log,batch_size = load_checkpoint(model, PATH, optim)

def reparametrize(mu,log):
    # 返回重采样的z
    return mu + torch.randn_like(log) + torch.exp(log)

# 填装测试数据,用来和生成模型的特征进行比对
testData = h5py.File('16eval.hdf5','r')
for key in testData.keys():
    testConfigs = testData[key][:batch_size]

testConfigs = reshapeIsing_MSE(testConfigs,batch_size)
testFeatures = calculate(testConfigs)
# 归一化计算  (f - f.mean()) / f.std()



epochs = 1

for epoch in range(epochs):
    model.eval()
    z = reparametrize(mu,log)
    x_ = model.decode(z)
    configs = reshapeIsing_MSE(x_,batch_size)
    print(configs)
    features = calculate(configs)
    # 得到configs