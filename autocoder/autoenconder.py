import pickle
import h5py

'''
datafile = open('data/IsingGraph/data16.pkl','rb')
data = pickle.load(datafile)
print(len(data))
print(data[0].x)
print(data[0].edge_attr)
'''

data_config = open('data/Ising/dataSize16.pkl','rb')
data_map = pickle.load(data_config)
print(data_map.keys())

for key in data_map.keys():
    print(len(data_map[key]))


#hdf = h5py.File('data/Ising/16size.hdf5','r')



#print(list(hdf.keys()))
#for key in list(hdf.keys()):
    #print('{}\n{}'.format(key,hdf[key][:]))
    #print(hdf[key][1])
    #size,config=dataset.reshape_Ising(hdf[key][1])
    #gird = Grid_gpu(size,1,config)
    #print(gird.canvas)

