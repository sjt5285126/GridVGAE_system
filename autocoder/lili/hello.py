import torch
import torch_geometric
from torch_geometric.utils import to_dense_adj
import torch_geometric.loader as gloader
import pickle


# data_file = open('../data/IsingGraph/data_16_T_PTPQuick.pkl', 'rb')
# data = pickle.load(data_file)
# data_file.close()
#
# print('data.y:{}'.format(data[0].y))
#
# data[0].y = torch.tensor([5], dtype=torch.int8)
#
# print('new data.y:{}'.format(data[0].y))
#
# data = to_dense_adj(data[0].edge_index)
#
# print(data.shape)

def get_dataset_attr(path):
    data_file = open(path, 'rb')
    data = pickle.load(data_file)
    data_file.close()

    dataset = gloader.DenseDataLoader(data, batch_size=20, shuffle=True)
    for data in dataset:
        # print('data:\n{}'.format(data))
        # print('data.x:\n{}'.format(data.x.shape))
        # print('data.adj:\n{}'.format(data.adj.shape))
        # adj = data.adj
        # adj = adj.reshape(20, 16, 16)
        # print('data.adj:\n{}'.format(adj[0]))
        # print('data.y\n{}'.format(data.y.shape))
        print('data.y\n{}'.format(data.y))
    print(dataset)
    # print("dataset.num_features:{}\ndataset.num_classes:{}\n".format(dataset.num_features, dataset.num_classes))


get_dataset_attr('data/IsingGraph/data_4size3T.pkl')
