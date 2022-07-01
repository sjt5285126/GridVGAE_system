import h5py
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# 获取生成构型的物理特征值
# 本程序中所采用的是demo_1.py 与 eval.py中所生成的hdf5数据格式
# 需要的时候请及时修改
f = h5py.File('data/Ising/T_PTP_64mix32_32.hdf5', 'r')
f1 = h5py.File('data/Ising/32_T_PTPFeatures.hdf5', 'r')
print(f.keys())
print(f1.keys())
if len(f['AvrM']) < len(f1['T=2.269_AvrM']):
    length = len(f['AvrM'])
else:
    length = len(f1['T=2.269_AvrM'])

AvrM = f['AvrM'][:length]
AvrE = f['AvrE'][:length]
df1 = pd.DataFrame({"AvrM": AvrM, "AvrE": AvrE, "classfiy": "VGAE"})
print("eval AvrM: mu:{},log:{}".format(np.mean(AvrM), np.std(AvrM) ** 2))
print("eval AvrE: mu:{},log:{}".format(np.mean(AvrE), np.std(AvrE) ** 2))

print(f1.keys())
AvrM1 = f1['T=2.269_AvrM'][:length]
AvrE1 = f1['T=2.269_AvrE'][:length]
df2 = pd.DataFrame({"AvrM": AvrM1, "AvrE": AvrE1, "classfiy": "MCMC"})
'''
f1 = h5py.File('data/Ising/16_PTP.hdf5', 'r')
print(f1.keys())
TotalM1 = f1['T=2.269_TotalM'][:1000]
TotalE1 = f1['T=2.269_TotalE'][:1000]
AvrM1 = f1['T=2.269_AvrM'][:1000]
AvrE1 = f1['T=2.269_AvrE'][:1000]
df2 = pd.DataFrame({"TotalM": TotalM1, "TotalE": TotalE1, "AvrM": AvrM1, "AvrE": AvrE1, "classfiy": "MCMC"})
'''
res = df1.append(df2, ignore_index=True)


# 先进行归一化处理
# x_d = (x - x.mean())/x.std()
def Normalization(x):
    return (x - x.mean()) / x.std()


sns.histplot(data=res, x='AvrM', element="step", kde=True, hue='classfiy')
plt.title('M')
plt.legend(['MCMC', 'VGAE '])
plt.show()
# plt.savefig('./image/T=PTP_SIZE=16_M2.png')
plt.cla()
sns.histplot(data=res, x='AvrE', element="step", kde=True, hue='classfiy')
plt.title('Energy')
plt.legend(['MCMC', 'VGAE '])
plt.show()
# plt.savefig('./image/T=2.25_SIZE=16_Energy2.png')
'''
plt.cla()
sns.scatterplot(data=res, x='AvrM', y=np.linspace(0, 1, num=2000), hue='classfiy')
plt.savefig('./image/T=2.25_SIZE=16_M散点图2.png')
plt.cla()
sns.scatterplot(data=res, x='AvrE', y=np.linspace(0, 1, num=2000), hue='classfiy')
plt.savefig('./image/T=2.25_SIZE=16_Energy散点图2.png')
'''

# 混合模型的泛化性能更好
