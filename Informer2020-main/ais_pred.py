# 1.导入必要的模块

import torch
import torch.nn.functional as F   # F中包含很多函数比如激励函数
import matplotlib.pyplot as plt #用于绘图
import pandas as pd
import numpy as np

# 2.生成要拟合的数据点

# linspace(-1,1,100)，从-1～1之间取100个数
# 因为torch只能处理二维数据，使用unsqueeze函数将一维数据转换为二维数据
#x = torch.unsqueeze(torch.linspace(-1,1,100),dim=1)	
# torch.squeeze(a，N)就是在a中指定位置N加上一个维数为1的维度


df = pd.DataFrame(pd.read_csv("data/413521520.csv",encoding='gbk'))
df["updatetime"] = pd.to_datetime(df["updatetime"])

x = torch.nn.functional.normalize(torch.unsqueeze(torch.from_numpy(df["updatetime"].values.astype(np.float32)),dim=1),p=1,dim=0)
y1 = torch.nn.functional.normalize(torch.unsqueeze(torch.Tensor(df["经度"].values.astype(np.float32)),dim=1),p=1,dim=0)
y2 = torch.nn.functional.normalize(torch.unsqueeze(torch.Tensor(df["纬度"].values.astype(np.float32)),dim=1),p=1,dim=0)



# plt.scatter(x.data.numpy(), y1.data.numpy())
# plt.show()

# 3.搭建神经网络

class Net(torch.nn.Module):
    # 设置神经网络属性，定义各层的信息
    # n_feature代表输入数据的个数,n_hidden是隐藏层神经元的个数,n_output输出的个数
    
    def __init__(self, n_feature, n_hidden,n_output):
        super(Net, self).__init__()
        self.hidden_lon = torch.nn.Linear(n_feature, n_hidden)
        self.hidden_lat = torch.nn.Linear(n_feature, n_hidden)
        self.predict_lon = torch.nn.Linear(n_hidden, n_output)
        self.predict_lat = torch.nn.Linear(n_hidden, n_output)

    # 前向传递过程
    def forward(self, x):
        lon = F.relu(self.hidden_lon(x))
        lon = self.predict_lon(lon)
        lat = F.relu(self.hidden_lat(x))
        lat = self.predict_lat(lat)
        
        return lon,lat

# 4.定义网络

net = Net(1,8,1)  # 定义一个含有8个神经元的隐藏层，一次只输入1个数据，输出1个数据的网络
print(net)

# 设置一个实时打印的过程，显示在屏幕
plt.ion()
plt.show()

# 5.设置优化器

# 选择SGD优化器对神经网络的全部参数进行优化，学习率设置为0.5
optimizer = torch.optim.SGD(net.parameters(),lr=0.1)
# 选择均方差为计算误差
loss_func = torch.nn.MSELoss()

# 6.训练


for t in range(200):
    pred_lon,pred_lat = net(x)   #产生预测值
    loss = loss_func(pred_lon, y1)  # 计算预测值与真实值之间的误差
    loss += loss_func(pred_lat, y2)  # 计算预测值与真实值之间的误差
    optimizer.zero_grad()  # 将网络中所有参数的梯度降为0
    loss.backward()  # 误差反向传播，并对每个节点计算梯度
    optimizer.step()  # 以学习率位0.5对梯度进行优化

    # 每学习5步打印一次
    if t % 5 == 0:
        y1 = y2
        plt.cla()		# # Clear axis即清除当前图形中的当前活动轴。其他轴不受影响。
        plt.scatter(x.data.numpy(),y1.data.numpy())  # 打印原始数据散点图
        plt.plot(x.data.numpy(),pred_lon.data.numpy(),'r-',lw=5) # 打印目前拟合的曲线
        plt.text(0.5, 0, 'Loss=%.4f' % loss.data,
                 fontdict={'size':20,'color':'red'})   # 打印当前的loss值
        plt.pause(0.1)
        print(f"loss:{loss.data}")

# 实时打印结束
plt.ioff()
plt.show()
