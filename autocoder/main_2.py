
import torch_geometric.data as gdata
from torch import nn
import torch.nn.functional as F
import matplotlib.pyplot as plt
from edge_conv_model import Net
from dataset import *


device = torch.device('cpu')

# 加载数据，划分数据
data_train = data[0:700]
data_test = data[700:1000]
data_trainloader = gdata.DataLoader(data_train, batch_size=50, shuffle=True)
print(data_trainloader)
data_testloader = gdata.DataLoader(data_test, batch_size=50, shuffle=True)
print(data_testloader)


# 加载模型
model = Net().to(device)
print(model)
# 定义优化器
optimizer = torch.optim.Adam(model.parameters(), lr=0.01, weight_decay=5e-4)

# 绘制图形
def image(output):
    output = output
    p = p_list
    return

# 设置网络训练参数
total_train_step = 0
total_test_step = 0
epoch = 100


# 开始训练
for epoch in range(epoch):
    print("----------------第{}轮训练开始了-----------------".format(epoch + 1))
    model.train()
    for one_train_batch in data_trainloader:
        optimizer.zero_grad()
        one_train_batch = one_train_batch.to(device)
        # print(one_train_batch)
        output = model(data=one_train_batch, batch=one_train_batch.batch)
        image(output)
        loss = F.nll_loss(output, one_train_batch.y)
        # print(loss)
        # print(loss.shape)
        loss.backward()
        optimizer.step()
        total_train_step += 1

        if total_train_step % 100 == 0:
            print("训练次数：{}, Loss: {}".format(total_train_step, loss.item()))

    # 测试步骤开始
    model.eval()
    total_test_loss = 0
    total_accuracy = 0
    for one_test_batch in data_testloader:
        one_test_batch = one_test_batch.to(device)
        output = model(data=one_test_batch, batch=one_test_batch.batch)
        loss = F.nll_loss(output, one_test_batch.y)
        total_test_loss = total_test_loss + loss.item()
        accuracy = (output.argmax(1) == one_test_batch.y).sum()
        total_accuracy = total_accuracy + accuracy

    # print("epoch: {}\t, loss: {:.4f}, test_acc: {:.4f}".format(epoch, loss, acc))
    # 输出在整个测试集上的total_test_loss
    print("整体测试集上平均loss: {}".format(total_test_loss / 300))
    # print(len(data_testloader))
    print("整体测试集上的正确率：{}".format(total_accuracy / 300))
    total_test_step += 1

# 保存网络模型
torch.save(model.state_dict(), "model_{}.pth".format(100))
print("模型已保存")


