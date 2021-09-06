import numpy as np

class configuration:
    def __init__(self,N):
        self.config = 2 * np.random.randint(2, size=(N, N)) - 1
        self.N = N

    def getConfig(self):
        return self.config


