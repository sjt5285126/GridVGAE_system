import numpy as np
import torch
from IsingGrid import Grid

class Grid_gpu(Grid):
    '''
    将数据由numpy变为torch，可以在GPU上运行
    '''
    def __init__(self, size, Jfactor, canvas=None):
        super(Grid_gpu, self).__init__(size, Jfactor, canvas=None)
        self.dev = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        if canvas is None:
            self.canvas = torch.randint(0,2, [self.size, self.size],device=self.dev) * 2 - 1
        else:
            self.canvas = torch.tensor(canvas,dtype=torch.int,device=self.dev)

    def randomize(self):
        self.canvas = torch.randint(0,2, [self.size, self.size],device=self.dev) * 2 - 1

    def set_positive(self):
        self.canvas = torch.ones([self.size, self.size], dtype=torch.int,device=self.dev)

    def set_negative(self):
        self.canvas = -torch.ones([self.size, self.size], dtype=torch.int,device=self.dev)

    def totalM(self):
        return torch.sum(self.canvas)
    def getCanvas(self):
        return self.canvas.cpu().numpy()
'''
#测试用例
grid = Grid_gpu(3,1)
grid.clusterFlip(0.5)
print(grid.canvas.shape)
'''

