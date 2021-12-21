import random
import math
import torch
from torch_geometric.data import Data
import numpy as np
from IsingGrid_gpu import Grid_gpu
import IsingGrid
import h5py
import pickle
import time
from Ising import Config

'''
验证出 数据集在cpu上运行速度要大于在gpu上的运行速度，因为生成的构型非常小
在GPU上数据全部都用作数据流的交换 严重影响了数据生成的速度，所以应该将数据集的生成放在CPU上处理

可能在大规模计算上 仍需要放在gpu上去处理
'''


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
    t = np.math.ceil(k / 2)
    # 向上取整
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


def init_data(n: int, p_list: list, graph_num: int, c: int = 2):
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


def init_ising(n: int, T_list: list, config_nums: int, c: int = 2):
    '''
    需要对数据进行持久化层处理，并且保存图结构与普通结构两种状态到不同的文件
    :param n: 构型的尺寸大小
    :param T_list: 温度列表，表示生成不同温度下的构型
    :param config_num: 每种温度下的稳态的构型数
    :param c: 构型的结构 默认是2体相互作用晶格
    :return: 无返回项
    '''
    begin = time.time()
    # 保存未被转化为图结构的构型文件
    config_file = h5py.File('data/Ising/{}size.hdf5'.format(n), 'w')
    # config_map = {}
    # 生成节点
    x = []
    count = 0
    for T in T_list:
        for num in range(config_nums):
            gird = IsingGrid.Grid(n, 1)
            gird.randomize()
            # 得到热浴平衡下的构型
            for i in range(1000):
                gird.clusterFlip(T)
            # 将构型平整为1维
            # print(gird.canvas)
            x_list = gird.canvas.reshape((n * n, 1))
            # print(x_list)
            # 取构型的绝对值
            x_list = np.where(np.sum(x_list) > 0, x_list, -x_list)
            # 将-1转换1
            x_list = np.where(x_list > 0, x_list, 0)
            x.append(x_list)
        # 数据以一维形式存放在hdf5文件中
        config_file.create_dataset('T={}'.format(T), data=np.array(x[count * config_nums:(count + 1) * config_nums]))
        count += 1
        # config_map['T={}'.format(T)] = gird_list
    config_file.close()
    # 生成边
    edge_index = []
    for i in range(n):
        for j in range(n):
            # 对于每个顶点,添加他的四条边
            edge_index.extend(neighbor((i, j), 1, n, c))
    edge_index = torch.tensor(edge_index, dtype=torch.long)
    edge_index = edge_index.t().contiguous()
    # print(edge_index.shape)
    ### 生成edge_attr:若两个节点值都为1,则其相连的边属性值为1
    # 遍历边集:如果两顶点都为1,则该边属性赋值为1
    edge_attr = []
    for num in range(config_nums * len(T_list)):
        # 对每一个顶点都生成一个 权值矩阵
        edge_attr_graph = np.zeros((len(edge_index[0]), 1))
        for i in range(len(edge_index[0])):
            edge_1 = edge_index[0][i]
            edge_1_index = edge_1.item()
            edge_2 = edge_index[1][i]
            edge_2_index = edge_2.item()
            if x[num][edge_1_index] == x[num][edge_2_index]:
                edge_attr_graph[i] = -1
            else:
                edge_attr_graph[i] = 1
        edge_attr.append(edge_attr_graph)

    # 生成图标签y与生成节点标签y只能有一个存在

    # 生成图标签y的值
    y_list = []
    for id_t, t in enumerate(T_list):
        for i in range(config_nums):
            y_list.append([id_t])
    # 生成节点标签y的值

    data = []
    for x, y, z in zip(x, y_list, edge_attr):
        y = torch.tensor(y)
        x = torch.tensor(x, dtype=torch.float)
        z = torch.tensor(z, dtype=torch.float)
        data.append(Data(x=x, edge_index=edge_index, edge_attr=z, y=y))

    # 将数据集进行打乱
    # 数据生成
    random.shuffle(data)
    # 将生成的graph数据集放在磁盘上
    file = open('data/IsingGraph/data{}.pkl'.format(n), 'wb')
    pickle.dump(data, file)
    file.close()
    print(len(data))
    end = time.time()
    print('cost time:{}'.format(end - begin))


'''
GPU代码可能需要的流处理影响了速度，实际应用并没有cpu版本的速度快需要进行改进

大量的I/O处理将降低了代码的运行时间，如果想要在GPU上获得高效率运算就需要尽量避免这些I/O流处理
所以 我们需要在GPU上将这些数据都生成完，再统一放回到CPU进行保存
'''


def init_Ising_gpu(n: int, T_list: list, config_nums: int, c: int = 2):
    '''
    需要对数据进行持久化层处理，并且保存图结构与普通结构两种状态到不同的文件
    在gpu上运行
    :param n: 构型的尺寸大小
    :param T_list: 温度列表，表示生成不同温度下的构型
    :param config_nums: 每种温度下的稳态的构型数
    :param c: 构型的结构 默认是2体相互作用晶格
    :return: 无返回项
    '''
    begin = time.time()
    dev = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(dev)
    config_map = {}
    # 生成节点
    x = []
    count = 0
    for T in T_list:
        for num in range(config_nums):
            gird = Grid_gpu(n, 1)
            # 得到热浴平衡下的构型
            for i in range(1000):
                gird.clusterFlip(T)
            # 将构型平整为1维
            # print(gird.canvas)
            x_list = gird.getCanvas().reshape((n * n, 1))
            # print(x_list)
            # 取构型的绝对值
            x_list = np.where(np.sum(x_list) > 0, x_list, -x_list)
            # 将-1转换1
            x_list = np.where(x_list > 0, x_list, 0)
            x.append(x_list)
        config_map['T={}'.format(T)] = x[config_nums * count:]
        count += 1
    # config_file.close()
    # 生成边
    edge_index = []
    for i in range(n):
        for j in range(n):
            # 对于每个顶点,添加他的四条边
            edge_index.extend(neighbor((i, j), 1, n, c))
    edge_index = torch.tensor(edge_index, dtype=torch.long, device=dev)
    edge_index = edge_index.t().contiguous()
    # print(edge_index.shape)
    ### 生成edge_attr:若两个节点值都为1,则其相连的边属性值为1
    # 遍历边集:如果两顶点都为1,则该边属性赋值为1
    edge_attr = []
    for num in range(config_nums * len(T_list)):
        # 对每一个顶点都生成一个 权值矩阵
        edge_attr_graph = torch.zeros((len(edge_index[0]), 1), device=dev)
        for i in range(len(edge_index[0])):
            edge_1 = edge_index[0][i]
            edge_1_index = edge_1.item()
            edge_2 = edge_index[1][i]
            edge_2_index = edge_2.item()
            if x[num][edge_1_index] == x[num][edge_2_index]:
                edge_attr_graph[i] = -1
            else:
                edge_attr_graph[i] = 1
        edge_attr.append(edge_attr_graph)

    # 生成图标签y与生成节点标签y只能有一个存在

    # 生成图标签y的值
    y_list = []
    for id_t, t in enumerate(T_list):
        for i in range(config_nums):
            y_list.append([id_t])
    # 生成节点标签y的值

    data = []
    for x, y, z in zip(x, y_list, edge_attr):
        y = torch.tensor(y, device=dev)
        x = torch.tensor(x, dtype=torch.float, device=dev)
        data.append(Data(x=x, edge_index=edge_index, edge_attr=z.float(), y=y))

    # 将数据集进行打乱
    # 数据生成
    random.shuffle(data)
    # 将生成的graph数据集放在磁盘上
    file = open('data/IsingGraph/data{}.pkl'.format(n), 'wb')
    file_config = open('data/Ising/dataSize{}.pkl'.format(n), 'wb')
    pickle.dump(data, file)
    pickle.dump(config_map, file_config)
    file.close()
    file_config.close()
    print(len(data))
    end = time.time()
    print('cost time:{}'.format(end - begin))


'''
IsingInit 为优化的生成数据集机器 
优化的方向为：
直接生成tensor 减少numpy或list转换为tensor的时间
将只使用一次的数据 不进行存储并及时清理内存
'''


def IsingInit(size, T_list, nums):
    begin = time.time()

    # 数据存放位置
    data = []
    # 形成对应的边

    edge_nums = size * size * 4
    edge_index = torch.zeros((2, edge_nums), dtype=torch.long)
    location = 0
    for x in range(size):
        for y in range(size):
            edge_index[0, location:location + 4] = x * size + y
            edge_index[1][location] = (x - 1) % size * size + y
            edge_index[1][location + 1] = x * size + (y + 1) % size
            edge_index[1][location + 2] = (x + 1) % size * size + y
            edge_index[1][location + 3] = x * size + (y - 1) % size
            location += 4

    # 形成图的对应标签
    y_list = torch.zeros((len(T_list) * nums, 1), dtype=torch.int8)
    for id_t, t in enumerate(T_list):
        for i in range(nums):
            y_list[id_t * nums + i] = id_t

    config_file = h5py.File('data/Ising/{}size.hdf5'.format(size), 'w')
    count = 0
    for T in T_list:
        configs = Config(size, 1, nums, False)
        configs.wollfAll(T)
        # 一个温度下的所有构型都产生完毕
        config = configs.canvas.reshape((nums, size * size, 1))
        del configs
        config = np.where(np.sum(config) > 0, config, -config)
        config = np.where(config > 0, config, 0)
        config_file.create_dataset('T={}'.format(T), data=config)
        for canvas, y in zip(config, y_list[count * nums:(count + 1) * nums]):
            x = torch.tensor(canvas, dtype=float)
            edge_attr_graph = torch.ones((edge_nums, 1))
            for i in range(edge_nums):
                edge_1 = edge_index[0][i]
                edge_2 = edge_index[1][i]
                if x[edge_1] == x[edge_2]:
                    edge_attr_graph[i] = -1
            data.append(Data(x=x, edge_index=edge_index, y=y, edge_attr=edge_attr_graph))
        print('存入温度{}'.format(T))
        count += 1
    config_file.close()
    file = open('data/IsingGraph/data{}'.format(size), 'wb')
    pickle.dump(data, file)
    file.close()
    end = time.time()
    print(end - begin)


def reshape_Ising(gird):
    '''
    将一维的ising模型解压为二维
    :param gird: 需要解压的构型
    :return: 构型的尺寸，解压完的构型
    '''
    size = int(math.sqrt(len(gird)))
    gird = np.where(gird > 0, 1, -1)
    return size, gird.reshape((size, size))


init_ising(3, [1,2,3], 3)
# init_Ising_gpu(3,[2],3)
IsingInit(3, [1,2,3], 3)
