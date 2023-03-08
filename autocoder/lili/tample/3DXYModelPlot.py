import matplotlib.pyplot as plt
import numpy as np
import matplotlib.colors as colors
import matplotlib.cm as cmx

from mpl_toolkits import mplot3d

fig = plt.figure()

X, Y, Z = np.meshgrid(np.arange(0, 4), np.arange(0, 4), np.arange(0, 4))

# 分别在X,Y,Z坐标的分量
U = np.zeros((4, 4, 4))
V = np.zeros((4, 4, 4))
W = (np.random.randint(0, 2, [4, 4, 4]) * 2 - 1) / 2

top = np.ma.masked_where(W > 0, W)
bot = np.ma.masked_where(W < 0, W)

ax = plt.axes(projection='3d')
h1 = ax.quiver(X, Y, Z, U, V, 0.5, pivot='middle', colors='#4672C4', linewidths=4)
# ax.set(h1, 'maxheadsize', 1)
# h2 = ax.quiver(X, Y, Z, U, V, top, pivot='middle', colors='#EDBCB4',linewidths=4)
# ax.set(h2, 'maxheadsize', 1)
X, Y = np.meshgrid(np.arange(0, 4), np.arange(0, 4))

for i in range(0, 4):
    Z = np.zeros((4, 4)) + i
    ax.plot_wireframe(X, Y, Z, color='black', linewidths=1)

X, Z = np.meshgrid(np.arange(0, 4), np.arange(0, 4))
for i in range(0, 4):
    Y = np.zeros((4, 4)) + i
    ax.plot_wireframe(X, Y, Z, color='black', linewidths=1)
ax.axis("off")

plt.show()
