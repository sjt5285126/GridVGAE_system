import numpy as np

temp = []
x = []

for i in  range(3):
    for j in range(3):
        temp.append(j)
    x.append(temp)
    temp = []

x = np.array(x)
print(x)
print(x.sum(axis=0))