import random

import torch
from torch_geometric.data import Data
import numpy as np


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


def init_data(n: int, p_list: list, graph_num: int) -> list:
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

    print(np.array(x).shape)
    edge_index = []
    for i in range(n):
        for j in range(n):
            edge_index.extend(neighbor((i, j), 1, n, c=3))
    edge_index = torch.tensor(edge_index, dtype=torch.long)
    edge_index = edge_index.t().contiguous()
    # print(edge_index)
    print(np.array(edge_index).shape)
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
    print(np.array(edge_attr).shape)

    # y的值：概率大于0.6的赋值为1，小于0.6的赋值为0
    y_list = []
    for p in p_list:
        for i in range(100):
            if p >= 0.6:
                y_list.append([1])
            else:
                y_list.append([0])
    # print(y_list)
    print(np.array(y_list).shape)

    data = []
    for x, y, z in zip(x, y_list, edge_attr):
        y = torch.tensor(y)
        x = torch.tensor(x, dtype=torch.float)
        z = torch.tensor(z, dtype=torch.float)
        data.append(Data(x=x, edge_index=edge_index, edge_attr=z, y=y))

    random.shuffle(data)
    print(data)
    return data


p_list = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0]

init_data(4, p_list, 100)
