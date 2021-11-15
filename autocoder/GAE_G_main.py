import torch
import torch.nn as nn
import torch_geometric.nn as gnn
import torch_geometric.loader as gloader
import torch_geometric.data as gdata
from GAE_G import gae
import dataset

if __name__ == '__main__':
    # 加载设备
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    # 生成数据
    data =  dataset.init_data(4,dataset.init_p(10),100)
    data_loader_test = gloader.DataLoader(data,batch_size=20,shuffle=True)
    # 加载模型
    model = gae(1,8)
    model.load_state_dict(torch.load('GAE_G.pkl'))
    model.to(device)
    print(model)
    #　开始测试
    for batch in data_loader_test:
        batch = batch.to(device)
        x,z=model.encoder(batch.x,batch.edge_index,batch.batch)
        print(z)

