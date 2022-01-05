import pickle
import h5py


datafile = open('data/IsingGraph/data16.pkl','rb')
data = pickle.load(datafile)
print(len(data))
print(data[0].x)
print(data[0].edge_attr)

