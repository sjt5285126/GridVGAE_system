import torch
from VGAE_spin import EncoderSpin, DecoderSpin, SVGAE
import dataset
from dataset import reshapeIsing_MSE,acc,reshapeIsing,reshapeTorch
import pickle
import torch_geometric.loader as gloader
import time


# 导入model
def load_checkpoint(model, checkpoint_PATH, optimizer):
    if checkpoint_PATH != None:
        model_CKPT = torch.load(checkpoint_PATH,map_location=device)
        model.load_state_dict(model_CKPT['state_dict'], False)
        print("mu:\n{}".format(model_CKPT['mu']))
        print("log:\n{}".format(model_CKPT['log']))
        print('loading checkpoint!')
        optimizer.load_state_dict(model_CKPT['optimizer'])
    # 返回模型，优化器
    return model, optimizer,model_CKPT['batch_size']


# 构建模型

# 读取数据
device = torch.device('cuda:1' if torch.cuda.is_available() else 'cpu')
datafile = open('data/IsingGraph/data16.pkl', 'rb')
data = pickle.load(datafile)
test_batch = gloader.DataLoader(data,batch_size=200,shuffle=True)
datafile.close()
model = SVGAE(EncoderSpin(), DecoderSpin()).to(device)
optim = torch.optim.Adam(model.parameters(), lr=0.01)

# PATH = argv[1]

PATH = 'model_16_0228.pkl'

checkpoint = torch.load(PATH,map_location=device)
# 模型的测试
model, optim ,batch_size= load_checkpoint(model, PATH, optim)

for epoch in range(1000):
    model.eval()
    with torch.no_grad():
        for batch in test_batch:
            batch = batch.to(device)
            z = model.encode(batch.x,batch.edge_index,batch.edge_attr,batch.batch)
            x_ = model.decode(z)
            print("loss:{}".format(model.recon_loss(batch.x, x_) + model.kl_loss()))
            preConfig = reshapeIsing_MSE(batch.x, 200)
            afterConfig = reshapeIsing(x_, 200)
            #print("测试构型:{}".format(reshapeIsing_MSE(batch.x, 2)))
            #print("重构后的构型:{}".format(reshapeIsing_MSE(x_, 2)))
            print("acc:{}%".format(acc(preConfig, afterConfig)))
            time.sleep(10)




