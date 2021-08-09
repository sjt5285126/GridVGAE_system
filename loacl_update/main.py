"""
This file runs Ising simulation
Created: Mar. 30, 2019
Last Edited: Apr. 6, 2019
By Bill
"""
import numpy as np
import IsingGrid
import matplotlib as mpl
import matplotlib.pyplot as plt
from copy import deepcopy


# Fundamental parameters

size = 8
temperature = np.linspace(0.1,5,10)
steps = 400
interval = 100
Jfactor = 1

# Generate grid

g = IsingGrid.Grid(size, Jfactor)
g.randomize()

# Animation parameters

fig, ax = plt.subplots()
data = []

# m/t plot
M = []
print("m/t begin")
for t in temperature:
    m = 0
    for step in range(steps):
        g.clusterFlip(t)  # balance
    for step in range(steps):
        for i in range(8):
            g.clusterFlip(t) # compute
        m += abs(g.avrM())
    M.append(m/steps)
plt.xlabel('T')
plt.ylabel('m')
plt.title('wolff')
plt.plot(temperature, abs(np.array(M)), "ob")
plt.show()

# Simulation
'''
print("Simulation begins.")

for step in range(steps):

    # Single/cluster Filp

    # clusterSize = g.singleFlip(temperature)
    clusterSize = g.clusterFlip(temperature)

    if (step + 1) % interval == 0:
        data.append(deepcopy(g.canvas))

    if (step + 1) % (10 * interval) == 0:
        print("Step ", step + 1, "/", steps, ", Cluster size ", clusterSize, "/", size * size)

print("Simulation completes.")
'''


# Animation
'''
print("Animation begins.")

for frame in range(0, len(data)):
    ax.cla()
    ax.imshow(data[frame], cmap=mpl.cm.winter)
    ax.set_title("Step {}".format(frame * interval))
    plt.pause(1)

print("Animation completes.")
'''

