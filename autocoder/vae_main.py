import torch
from torch.utils.data import DataLoader
from torch import nn,optim
from torchvision import transforms,datasets
from vae import VAE
import visdom
import os

def main():
    # 训练集
    mnist_train = datasets.MNIST('mnist', True, transform=transforms.Compose([
        transforms.ToTensor()
    ]), download=True)
    mnist_train = DataLoader(mnist_train, batch_size=32, shuffle=True)

    # 测试集
    mnist_test = datasets.MNIST('mnist', True, transform=transforms.Compose([
        transforms.ToTensor()
    ]), download=True)
    mnist_test = DataLoader(mnist_test, batch_size=32, shuffle=True)
    '''
    下载的训练集和测试集 是以32张手写体数字照片为一组的 图片信息
    '''
    # 无监督学习 不需要label
    x,_ = iter(mnist_train).next() # 构建一个迭代器
    print('x:',x.shape)

    device = torch.device('cuda')
    model = VAE().to(device)
    criteon = nn.MSELoss() # 损失函数 计算x与x_hat之间的均方误差

    # 实现Adam算法,是一种梯度优化算法
    optimizer = optim.Adam(model.parameters(),lr=1e-3)
    print(model)

    viz = visdom.Visdom()

    for epoch in range(100): # 迭代步数

        '''
        enumerate() 函数用于将一个可遍历的数据对象(如列表、元组或字符串)组合为一个索引序列，
        同时列出数据下标和数据，一般用在 for 循环当中
        '''
        for batchidx,(x,_) in enumerate(mnist_train):

            # [b,1,28,28]
            x = x.to(device)

            #x_hat 表示重建后的x
            x_hat,kld = model(x)
            loss = criteon(x_hat,x)

            if kld is not None:
                elbo =  - loss - 1.0 * kld
                loss = - elbo

            #backprop
            # 将之前优化的梯度清空,对新一次的参数进行梯度下降
            # 不清零，那么使用的这个梯度就得同上一个batch有关，这不是我们需要的结果
            optimizer.zero_grad()
            # 获取该batch的梯度
            loss.backward()
            # 进行单个优化步骤
            optimizer.step()

        print('epoch:',epoch,'loss',loss.item(),'kld:',kld.item())

        x,_ = iter(mnist_test).next()
        x = x.to(device)
        '''
        with关键字相当于 try-finally的简写,
        但是 with没有捕获异常的能力,在发生异常以后终止块内代码的运行
        '''
        with torch.no_grad():
            x_hat,kld = model(x)
        viz.images(x,nrow=8,win='x',opts=dict(title='x'))
        viz.images(x_hat,nrow=8,win='x_hat',opts=dict(title='x_hat'))


if __name__ == '__main__':
    #os.environ['CUDA_LAUNCH_BLOCKING'] = 1
    main()
