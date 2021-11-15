# autocoder

#### 介绍
pytorch搭建各种自编码器的简易模型

#### VGAE
[VGAE代码](https://gitee.com/shiyuehua666/autocoder/blob/ba086b4fe46bc2b7e407a7cdda187ac4dec271dc/autocoder/VGAE.py)
使用PYG集成的包来搭建  
VGAE继承于torch_geometric.nn.models.autoencoder.GAE  
其中只需要构建好encoder,框架会自己定义decoder,
使用GAE中给定的loss来计算重构误差,同时加上KL散度

##### 后续想法
1. 可以换不同的图卷积模型来做测试,目前使用的卷积模型无法做到小批量训练
2. 使用dropout来缩小误差
3. 更换不同的损失函数
4. 自己构造数据集

##### 当前遇到的阻碍
1. 目前的VGAE与VAE都是node-level级别的,而我们要做的是graph-level的
2. 对于graph-level的分类需要应用池化,考虑什么样的池化函数
3. autocode的进步是否也可以做创新之一
4. GAN也是生成模型的一种,是否也可以来做应用

##### 创新点
1. 将图分类级别的图神经网络与自编码器结合
2. 使用合适的pooling池化与合适的图卷积核

##### GAE走不通的原因
1. 无法找到合适的将图池化后反池化的方法
2. 在得到隐藏特征*z*以后,对于不同晶格需要设计不同网络,例如4晶格情况下的网络模型就不适用于其他n晶格下的网络模型
3. 但是不同体晶格的网络模型,应该可以适用,例如4体,3体应该可以公用一个网络模型

### 综上所述
GAE无法应用到 无监督的图级别分类
