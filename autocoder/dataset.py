import random
import math
import torch
from torch_geometric.data import Data
import numpy as np
import IsingGrid
import h5py
import pickle

# 生成概率
def init_p(n: int) -> np.ndarray:
    '''
    :param n: 需要生成多少[0,1]之间的概率
    :return:  返回一个概率列表
    '''
    return np.linspace(0, 1, num=n)


# edge_index：四方晶格, 每个顶点有四条边;三角晶格，每个定点有六条边
def neighbor(location: tuple, k: int, N: int, c: int):
    """location:顶点二维坐标；k:k接邻居；N：晶格一行的定点数；c：2（四方晶格）3（三角晶格）"""
    nn = []
    x = location[0]
    y = location[1]
    t = np.math.ceil(k / 2)  # 向上取整
    if c == 2:
        nn.append([(x - t) % N, (y + t) % N])
        nn.append([(x + t) % N, (y + t) % N])
        nn.append([(x - t) % N, (y - t) % N])
        nn.append([(x + t) % N, (y - t) % N])
    else:
        nn.append([(x - t) % N, (y - t) % N])
        nn.append([(x - t) % N, y % N])
        nn.append([x % N, (y - t) % N])
        nn.append([x % N, (y + t) % N])
        nn.append([(x + t) % N, y % N])
        nn.append([(x + t) % N, (y + t) % N])
    result = []
    for i in nn:
        result.append([x * N + y, i[0] * N + i[1]])
    return result


def init_data(n: int, p_list: list, graph_num: int, c:int=2) -> list:
    '''

    :param n: 生成构型的规格 n*n
    :param p_list: 生成构型的概率序列
    :param graph_num: 每种概率生成的数量
    :param c: 几个晶格之间会有相互作用力
    :return: 一组晶格的图结构
    '''
    total_node = n * n
    x = []

    for p in p_list:
        for num in range(graph_num):
            index = np.random.choice(range(total_node), size=(int(total_node * p)), replace=False)
            # print(index)
            x_list = np.zeros((total_node, 1))
            for i in index:
                x_list[i] = 1
            # print(x_list)
            x.append(x_list)
    print('x_shape:{}'.format(np.array(x).shape))
    # print(np.array(x).shape)
    edge_index = []
    for i in range(n):
        for j in range(n):
            edge_index.extend(neighbor((i, j), 1, n, c))
    edge_index = torch.tensor(edge_index, dtype=torch.long)
    edge_index = edge_index.t().contiguous()
    print('edge_index_shape:{}'.format(np.array(edge_index).shape))
    # print(edge_index)
    # print(np.array(edge_index).shape)
    # print(len(edge_index[0]))

    # edge_attr:若两个节点值都为为1，则其相连的边属性值为1
    # 方法一：遍历边集：若两顶点都为1，则该边属性赋值为1
    edge_attr = []
    # 200张图,遍历每张图
    for num in range(graph_num * len(p_list)):
        edge_attr_graph = np.zeros((len(edge_index[0]), 1))
        for i in range(len(edge_index[0])):
            edge_1 = edge_index[0][i]
            edge_1_index = edge_1.item()
            edge_2 = edge_index[1][i]
            edge_2_index = edge_2.item()
            if x[num][edge_1_index] == 1 and x[num][edge_2_index] == 1:
                edge_attr_graph[i] = 1
        edge_attr.append(edge_attr_graph)
    # print(edge_attr)
    # print(np.array(edge_attr).shape)

    # y的值：概率大于0.6的赋值为1，小于0.6的赋值为0
    y_list = []
    for p in p_list:
        for i in range(graph_num):
            if p >= 0.6:
                y_list.append([1])
            else:
                y_list.append([0])
    # print(y_list)
    # print(np.array(y_list).shape)

    data = []
    for x, y, z in zip(x, y_list, edge_attr):
        y = torch.tensor(y)
        x = torch.tensor(x, dtype=torch.float)
        z = torch.tensor(z, dtype=torch.float)
        data.append(Data(x=x, edge_index=edge_index, edge_attr=z, y=y))

    random.shuffle(data)
    print(len(data))
    return data

def init_ising(n: int, T_list: list, config_nums: int, c: int=2):
    '''
    需要对数据进行持久化层处理，并且保存图结构与普通结构两种状态
    :param n:
    :param T_list:
    :param config_num:
    :param c:
    :return:
    '''
    total_node = n * n

    # 保存未被转化为图结构的构型文件
    config_file = h5py.File('data/Ising/{}size.hdf5'.format(n),'w')
    #config_map = {}
    # 生成节点
    x = []
    count = 0
    for T in T_list:
        for num in range(config_nums):
            gird = IsingGrid.Grid(n,1)
            gird.randomize()
            # 得到热浴平衡下的构型
            for i in range(1):
                gird.clusterFlip(T)
            #将构型平整为1维
            #print(gird.canvas)
            x_list = gird.canvas.reshape((n*n,1))
            #print(x_list)
            # 取构型的绝对值
            x_list = np.where(np.sum(x_list)>0,x_list,-x_list)
            # 将-1转换1
            x_list = np.where(x_list>0,x_list,0)
            x.append(x_list)
        # 数据以一维形式存放在hdf5文件中
        config_file.create_dataset('T={}'.format(T),data=np.array(x[count*config_nums:(count+1)*config_nums]))
        count += 1
        #config_map['T={}'.format(T)] = gird_list
    config_file.close()
    # 生成边
    edge_index = []
    for i in range(n):
        for j in range(n):
            # 对于每个顶点,添加他的四条边
            edge_index.extend(neighbor((i,j),1,n,c))
    edge_index = torch.tensor(edge_index,dtype=torch.long)
    edge_index = edge_index.t().contiguous()
    #print(edge_index.shape)
    ### 生成edge_attr:若两个节点值都为1,则其相连的边属性值为1
    # 遍历边集:如果两顶点都为1,则该边属性赋值为1
    edge_attr = []
    for num in range(config_nums*len(T_list)):
        # 对每一个顶点都生成一个 权值矩阵
        edge_attr_graph = np.zeros((len(edge_index[0]),1))
        for i in range(len(edge_index[0])):
            edge_1 = edge_index[0][i]
            edge_1_index = edge_1.item()
            edge_2 = edge_index[1][i]
            edge_2_index = edge_2.item()
            if x[num][edge_1_index] == 1 and x[num][edge_2_index] == 1:
                edge_attr_graph[i] = 1
        edge_attr.append(edge_attr_graph)

    # 生成图标签y与生成节点标签y只能有一个存在

    # 生成图标签y的值

    # 生成节点标签y的值

    data = []
    for x,z in zip(x,edge_attr):
        x = torch.tensor(x,dtype=torch.float)
        z = torch.tensor(z,dtype=torch.float)
        data.append(Data(x=x,edge_index=edge_index,edge_attr=z))

    # 将数据集进行打乱
    # 数据生成
    random.shuffle(data)
    # 将生成的graph数据集放在磁盘上
    file = open('data/IsingGraph/data{}.pkl'.format(n),'wb')
    pickle.dump(data,file)
    file.close()
    print(len(data))

def reshape_Ising(gird):
    '''
    将一维的ising模型解压为二维
    :param gird:
    :param size:
    :return:
    '''
    size = int(math.sqrt(len(gird)))
    gird = np.where(gird>0,1,-1)
    return size,gird.reshape((size,size))

#init_ising(3,[1,2,3],10)

