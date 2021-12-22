import numpy as np
import torch
import time

class Config():
    def __init__(self,size,Jfactor,nums,tensor = True):
        self.size = size
        self.Jfactor = Jfactor
        self.nums = nums
        self.tensor = tensor
        if self.tensor is True:
            self.canvas = torch.randint(0, 2, [self.nums,self.size, self.size],device='cuda') * 2 - 1
        else:
            self.canvas = np.random.randint(0, 2, [self.nums,self.size, self.size]) * 2 - 1
    def choiceGPU(self):
        self.canvas = self.canvas.to('cuda' if torch.cuda.is_available() else 'cpu')
    def left(self, x, y):
        if x < 0.5:
            return [self.size - 1, y]
        else:
            return [x - 1, y]

    def right(self, x, y):
        if x > self.size - 1.5:
            return [0, y]
        else:
            return [x + 1, y]

    def up(self, x, y):
        if y < 0.5:
            return [x, self.size - 1]
        else:
            return [x, y - 1]

    def down(self, x, y):
        if y > self.size - 1.5:
            return [x, 0]
        else:
            return [x, y + 1]

    def clusterFlip(self,canvas, temperature):
        """Cluster flip (Wolff method)"""

        # Randomly pick a seed spin

        x = np.random.randint(0, self.size)
        y = np.random.randint(0, self.size)

        sign = canvas[x, y]
        P_add = 1 - np.exp(-2 * self.Jfactor / temperature)
        stack = [[x, y]]
        lable = np.ones([self.size, self.size], int)
        lable[x, y] = 0

        while len(stack) > 0.5:

            # While stack is not empty, pop and flip a spin

            [currentx, currenty] = stack.pop()
            self.canvas[currentx, currenty] = -sign

            # Append neighbor spins

            # Left neighbor

            [leftx, lefty] = self.left(currentx, currenty)

            if canvas[leftx, lefty] * sign > 0.5 and \
                    lable[leftx, lefty] and np.random.rand() < P_add:
                stack.append([leftx, lefty])
                lable[leftx, lefty] = 0

            # Right neighbor

            [rightx, righty] = self.right(currentx, currenty)

            if canvas[rightx, righty] * sign > 0.5 and \
                    lable[rightx, righty] and np.random.rand() < P_add:
                stack.append([rightx, righty])
                lable[rightx, righty] = 0

            # Up neighbor

            [upx, upy] = self.up(currentx, currenty)

            if canvas[upx, upy] * sign > 0.5 and \
                    lable[upx, upy] and np.random.rand() < P_add:
                stack.append([upx, upy])
                lable[upx, upy] = 0

            # Down neighbor

            [downx, downy] = self.down(currentx, currenty)

            if canvas[downx, downy] * sign > 0.5 and \
                    lable[downx, downy] and np.random.rand() < P_add:
                stack.append([downx, downy])
                lable[downx, downy] = 0

    def wollfAll(self,temperature):
        count = 0
        for canva in self.canvas:
            for i in range(10 * self.size ** 3):
                self.clusterFlip(canva,temperature)
            if count%10 == 0:
                print('-----完成{}%-----'.format(count/self.nums))


'''
begin1 = time.time()
config1 = config(3,1,3,2)
config1.wollfAll()
end1 = time.time()
print(end1-begin1)



begin2 = time.time()
config2 = config(3,1,3,2,False)
config2.wollfAll()
end2 = time.time()
print(end2-begin2)

'''