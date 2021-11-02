import torch
from torch import nn
import numpy as np

class VAE(nn.Module):

    def __init__(self):
        super(VAE,self).__init__()

        # b[b,784] => b[2,20]
        # u:[b,10]  sigma:[b,10]
        self.encoder = nn.Sequential(
            nn.Linear(784,256),
            nn.ReLU(),
            nn.Linear(256,64),
            nn.ReLU(),
            nn.Linear(64,20),
            nn.ReLU(),
        )

        # [b,10] => [b,784]
        self.decoder = nn.Sequential(
            nn.Linear(10,64),
            nn.ReLU(),
            nn.Linear(64,256),
            nn.ReLU(),
            nn.Linear(256,784),
            nn.Sigmoid()
        )
        self.criterion = nn.MSELoss()

    def forward(self,x):
        bacth_size = x.size(0)
        # flatten
        x = x.view(bacth_size,784)
        # encoder
        h_ = self.encoder(x)
        # [b,20] => [b,10] and [b,10]
        mu,sigma = h_.chunk(2,dim=1) # 分块
        # reparametrize trick, epison~N(0,1)
        h = mu + sigma * torch.randn_like(sigma)

        # decoder
        x_hat = self.decoder(h)

        # reshape
        x_hat = x_hat.view(bacth_size,1,28,28)

        # KL divergence
        # 因为我们想要逼近的是 N~（0，1） 所以 mu2=0,sigma2 = 1
        kld = 0.5 * torch.sum(
            torch.pow(mu,2) +
            torch.pow(sigma,2) -
            torch.log(1e-8 + torch.pow(sigma,2))-1
        ) / (bacth_size*28*28)

        return x_hat,kld