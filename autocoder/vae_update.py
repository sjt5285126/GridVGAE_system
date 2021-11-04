import torch
import torch.nn as nn

# vae更新代码,更新点在于 mu与logvar 通过两个神经网络分别得出
class VAE_new(nn.Module):
    def __init__(self):
        super(VAE_new, self).__init__()
        self.fc1 = nn.Linear(784,256)
        self.fc2 = nn.Linear(256,64)
        self.fc31 = nn.Linear(64,10)
        self.fc32 = nn.Linear(64,10)
        self.relu = nn.ReLU()
        self.decoder = nn.Sequential(
            nn.Linear(10,64),
            nn.ReLU(),
            nn.Linear(64,256),
            nn.ReLU(),
            nn.Linear(256,784),
            nn.Sigmoid()
        )

    def encoder(self,x):
        h1 = self.relu(self.fc1(x))
        h2 = self.relu(self.fc2(h1))

        return self.relu(self.fc31(h2)),self.relu(self.fc32(h2))

    def reparamtrize(self,mu,logvar):
        return mu + logvar * torch.randn_like(logvar)

    def forward(self,x):
        batch_size = x.size(0)
        # flatten
        x = x.view(batch_size,784)
        mu,logvar = self.encoder(x)
        h_ = self.reparamtrize(mu,logvar)
        x_hat = self.decoder(h_)
        x_hat = x_hat.view(batch_size,1,28,28)

        # KL divergence
        # 因为我们想要逼近的是 N~（0，1） 所以 mu2=0,sigma2 = 1
        kld = 0.5 * torch.sum(
            torch.pow(mu, 2) +
            torch.pow(logvar, 2) -
            torch.log(1e-8 + torch.pow(logvar, 2)) - 1
        ) / (batch_size * 28 * 28)

        return x_hat,kld



