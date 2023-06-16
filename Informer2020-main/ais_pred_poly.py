import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler
from datetime import datetime

df = pd.DataFrame(pd.read_csv("data/413521520.csv",encoding='gbk'))

df["updatetime"] = pd.to_datetime(df["updatetime"])


pred = np.datetime64("2022-06-23 10:46")
x = df["updatetime"].values[:19]
x = np.append(x,pred)
x = x.astype(np.float32)
x = x.reshape(len(x),1)

sca_x = MinMaxScaler()
x = sca_x.fit_transform(x)

y1 = df["经度"].values[:19].astype(np.float32)
y1 = y1.reshape(len(y1),1)

sca_y1 = MinMaxScaler()
y1 = sca_y1.fit_transform(y1)

y2 = df["纬度"].values[:19].astype(np.float32)




x = x.reshape(len(x))
y1 = y1.reshape(len(y1))

z1 = np.polyfit(x[:19], y1, 3)#用3次多项式拟合
p1 = np.poly1d(z1)
pred = p1(x[-1])
print(pred)
yvals=p1(x)#也可以使用yvals=np.polyval(z1,x)
plot1=plt.plot(x[:19], y1, '*',label='original values')
plot2=plt.plot(x[:19], yvals[:19], 'r',label='polyfit values')
plot3=plt.plot(x[-1], yvals[-1], '*',label='pred')
plt.xlabel('x axis')
plt.ylabel('y axis')
plt.legend(loc=4)
plt.title('polyfitting')
plt.show()
print(sca_y1.inverse_transform(yvals.reshape(20,1)))