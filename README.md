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

