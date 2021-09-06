import numpy as np

def sigmoid(z):
    return 1 / (1+np.exp(-z))

class RBM:
    '''
    使用该RBM的要求是 传入相应的可见节点与不可见节点
    '''
    def __init__(self, n_visible, n_hidden):
        self.n_visible = n_visible
        self.n_hidden = n_hidden
        self.bias_a = np.zeros(self.n_visible)  # 可视层偏移量
        self.bias_b = np.zeros(self.n_hidden)  # 隐藏层偏移量
        self.weights = np.random.normal(0, 0.01, size=(self.n_visible, self.n_hidden))
        self.n_sample = None

    def encode(self, v):
        # 编码，即基于v计算h的条件概率：p(h=1|v)
        return sigmoid(self.bias_b + v @ self.weights) #返回一个矩阵

    def decode(self, h):
        # 解码(重构)：即基于h计算v的条件概率：p(v=1|h)
        return sigmoid(self.bias_a + h @ self.weights.T) #返回一个矩阵

    def gibbs_sample(self, v0, max_cd):
        # max_cd cd算法次数？
        # gibbs采样, 返回max_cd采样后的v以及h值
        v = v0
        for _ in range(max_cd):
            # 首先根据输入样本对每个隐藏层神经元采样。二项分布采样，决定神经元是否激活
            ph = self.encode(v)
            '''
            binomial(n,p,size) 从二项分布中抽取样本
            n代表试验次数,p代表概率,size表示输出的大小 返回成功的次数列表
            列表大小为size
            '''
            h = np.random.binomial(1, ph, (self.n_sample, self.n_hidden))
            # 根据采样后隐藏层神经元取值对每个可视层神经元采样
            pv = self.decode(h)
            v = np.random.binomial(1, pv, (self.n_sample, self.n_visible))
        return v

    def update(self, v0, v_cd, eta):
        '''
        更新RBM网络中的权值矩阵与偏移量
        :param v0: 表示初始的数据
        :param v_cd: 经过一次RMB运算后的数据
        :param eta: 学习率
        :return: null
        '''
        # 根据Gibbs采样得到的可视层取值(解码或重构)，更新参数
        ph = self.encode(v0)
        ph_cd = self.encode(v_cd)
        self.weights += eta * (v0.T @ ph - v_cd.T @ ph)  # 更新连接权重参数
        self.bias_b += eta * np.mean(ph - ph_cd, axis=0)  # 更新隐藏层偏移量b
        self.bias_a += eta * np.mean(v0 - v_cd, axis=0)  # 更新可视层偏移量a
        return

    def train(self, data, max_step, max_cd=2, eta=0.1):
        # 训练主函数,采用对比散度算法(CD算法)更新参数
        # 输入数据的维度不能和可见节点相同
        # 为什么不能相等？
        assert data.shape[1] == self.n_visible, "输入数据维度与可视层神经元数目不相等"
        self.n_sample = data.shape[0] #样例个数
        for i in range(max_step): #max_step 步数
            v_cd = self.gibbs_sample(data, max_cd) #传入矩阵,每一行代表一个样本
            self.update(data, v_cd, eta)
            error = np.sum((data - v_cd) ** 2) / self.n_sample / self.n_visible * 100
            if i == (max_step - 1):  # 将重构后的样本与原始样本对比计算误差
                print("可视层(隐藏层)状态误差比例:{0}%".format(round(error, 2)))

    def predict(self, v):
        # 输入训练数据，预测隐藏层输出
        ph = self.encode(v)[0] #得到第一个样本的隐藏层概率
        states = ph >= np.random.rand(len(ph)) #通过bool判断 得到ph的概率后 预计states的输出
        return states.astype(int) #将bool格式转换成int格式
