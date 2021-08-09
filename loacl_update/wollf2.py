from math import *
from random import *
import matplotlib.pyplot as plt
import numpy as np
L=8
N=L*L
step1=1000
T=np.linspace(0.1,5,10)
nbr = {i: ((i // L) * L + (i + 1) % L, (i + L) % N, (i // L) * L + (i - 1) % L, (i - L) % N) for i in range(N)}
S = [choice([-1, 1]) for k in range(N)]
#spinall=[]
#beta=1/T
#p=1-exp(-2*beta)
#fig,ax=plt.subplots()
def flip():
    m = 0
    for step in range(step1):
        for i in range(8):
            k = randint(0, N - 1)
            Pocket, Cluster = [k], [k]
            while Pocket != []:
                j = choice(Pocket)
                for l in nbr[j]:
                    if (S[l] == S[j]) and (l not in Cluster) and (random() < p):
                        Pocket.append(l)
                        Cluster.append(l)
                Pocket.remove(j)
            for j in Cluster:
                S[j] *= -1
        m += sum(S)
    return m/(N*step1)
M = []
for t in T:
    beta = 1/t
    p=1-exp(-2*beta)
    flip()
    M.append(flip())
plt.xlabel('T')
plt.ylabel('m')
plt.title('wolff')
plt.plot(T, abs(np.array(M)), "ob")
plt.show()