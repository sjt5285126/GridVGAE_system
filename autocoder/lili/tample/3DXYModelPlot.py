import matplotlib.pyplot as plt
import numpy as np
from mpl_toolkits import mplot3d

fig = plt.figure()

X, Y, Z = np.meshgrid(np.arange(0, 4), np.arange(0, 4), np.arange(0, 4))

# Z = np.linspace(0, 7, num=8)
'''
print("x:")
print(X)
print(X.shape)
print("y")
print(Y)
print(Y.shape)
print("z")
print(Z)
print(Z.shape)
#
ax = plt.axes(projection='3d')
ax.quiver(X, Y, Z, 0, 0, 0.5)
ax.grid(True)
ax.axis("off")

plt.show()

'''

# 有正有负的3D
size = 4
x_ = np.zeros((size, size, size))
xl = []
y_ = np.zeros((size, size, size))
yl = []
z_ = np.zeros((size, size, size))
zl = []
for num in range(0, 6):
    i = np.random.randint(size)
    j = np.random.randint(size)
    k = np.random.randint(size)
    xl.append((i, j, k))
    i = np.random.randint(size)
    j = np.random.randint(size)
    k = np.random.randint(size)
    yl.append((i, j, k))
    i = np.random.randint(size)
    j = np.random.randint(size)
    k = np.random.randint(size)
    zl.append((i, j, k))

ax = plt.axes(projection='3d')

for num in range(0, 6):
    x_i, x_j, x_k = xl[num]
    x_[x_i][x_j][x_k] = X[x_i][x_j][x_k]
    i, j, k = yl[num]
    y_[i][j][k] = Y[i][j][k]
    i, j, k = zl[num]
    z_[i][j][k] = Z[i][j][k]


for num in range(0, 6):
    x_i, x_j, x_k = xl[num]
    X[x_i][x_j][x_k] = np.NaN
    i, j, k = yl[num]
    Y[i][j][k] = np.NaN
    i, j, k = zl[num]
    Z[i][j][k] = np.NaN

ax.quiver(X, Y, Z, 0, 0, 0.5)
# ax.quiver(x_, y_, z_, 0, 0, -0.5, colors='red')
# ax.axis("off")

plt.show()
