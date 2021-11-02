import torch
from torch import nn

class AE(nn.Module):

    def __init__(self):
        super(AE, self).__init__()

        # [b,784] => [b,20]
        # 编码器部分
        '''
        nn.Sequential 是一个顺序容器,可以当作是一个小型的神经网络
        按顺序将参数进行传递,接受到input并将其传递给第一个模块
        将输出“链接”到每个后续模块的输入，最后返回最后一个模块的输出
        '''
        self.encoder = nn.Sequential(
            # nn.Linear 对传入数据进行线性变换 y = x * A@T + b
            nn.Linear(784,256), # 将784维 降维到256
            nn.ReLU(), # 使用RLU作为激活函数
            nn.Linear(256,64),
            nn.ReLU(),
            nn.Linear(64,20),
            nn.ReLU()
        )

        # [b,20] => [b,784]

        self.decoder = nn.Sequential(
            nn.Linear(20,64),
            nn.ReLU(),
            nn.Linear(64,256),
            nn.ReLU(),
            nn.Linear(256,784),
            nn.Sigmoid()
        )

    # forword 前馈神经网络
    # 继承nn.module之后 构建forward函数会自动的构建反向传播
    def forward(self,x):
        """

        :param x:
        :return:
        """
        bathsz = x.size(0)

        # flatten 平面化,embedding 为一个vector 28*28 = 784
        x = x.view(bathsz,784)
        # encoder
        x = self.encoder(x)
        # decoder
        x = self.decoder(x)
        # reshape
        x = x.view(bathsz,1,28,28)

        return x