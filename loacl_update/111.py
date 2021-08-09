import numpy as np
import math
import wolff
from numpy.random import rand




def init_state(N):
    state = 2 * np.random.randint(2, size=(N, N)) - 1
    return state


def nn(grid, location, k, N):
    nn = []
    x = location[0]
    y = location[1]
    t = math.ceil(k / 2)
    if k % 2 == 0:
        nn.append(grid[(x - t) % N][(y + t) % N])
        nn.append(grid[(x + t) % N][(y + t) % N])
        nn.append(grid[(x - t) % N][(y - t) % N])
        nn.append(grid[(x + t) % N][(y - t) % N])
    else:
        nn.append(grid[(x - t) % N][y])
        nn.append(grid[x][(y + t) % N])
        nn.append(grid[(x + t) % N][y])
        nn.append(grid[x][(y - t) % N])
    return nn


def flipping_wolff(grid, beta):
    N = len(grid)
    for i in range(N):
        for j in range(N):
            a = np.random.randint(0, N)
            b = np.random.randint(0, N)
            wolff.flip(grid, (a, b), beta)
    return grid


def flipping(grid, beta):
    N = len(grid)
    for i in range(N):
        for j in range(N):
            a = np.random.randint(0, N)
            b = np.random.randint(0, N)
            s = grid[a][b]
            E = np.array(nn(grid, (a, b), 1, N)).sum()
            cost = 2 * s * E
            if cost < 0:
                s *= -1
            elif rand() < np.exp(-cost * beta):
                s *= -1
            grid[a][b] = s
    return grid



def metropolis(eqsteps, mcsteps, N):
    T = np.linspace(0.1, 5, 2 ** 8)
    n1 = 1 / (N * N * mcsteps)
    M = []
    j = 0
    for t in T:
        m = 0
        config = init_state(N)

        for i in range(eqsteps):
            flipping(config, 1 / t)

        for i in range(mcsteps):
            flipping(config, 1 / t)
            m += np.sum(config)
        M.append(m * n1)
    j += 1
    if j % 10 == 0:
        print("%d" % j)

    f = open('metropolis.txt','w')
    f.write(str(abs(np.array(M))))
    f.write(str(T))
    f.close()

    '''
    plt.xlabel('T')
    plt.ylabel('m')
    plt.title('metropolis')
    plt.plot(T, abs(np.array(M)), "ob")
    plt.show()
    '''



def mc_wolff(eqsteps, mcsteps, N):
    T = np.linspace(0.1, 5, 2 ** 8)
    n1 = 1 / (N * N * mcsteps)
    M = []
    j = 0
    for t in T:
        m = 0
        config = init_state(N)
        for i in range(eqsteps):
            flipping_wolff(config, 1 / t)
        for i in range(mcsteps):
            flipping_wolff(config, 1 / t)
            m += np.sum(config)
        M.append(m * n1)
        j += 1
        if j % 10 == 0:
            print("%d" % j)
    '''
    plt.xlabel('T')
    plt.ylabel('m')
    plt.title('wolff')
    plt.plot(T, abs(np.array(M)), "ob")
    plt.show()
    '''
    f = open('wolff.txt','w')
    f.write(str(abs(np.array(M))))
    f.write(str(T))
    f.close()



if __name__ == '__main__':
    metropolis(50, 50, 8)
    mc_wolff(50, 50, 8)
