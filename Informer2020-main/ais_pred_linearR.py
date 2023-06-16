from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import MinMaxScaler
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

df = pd.DataFrame(pd.read_csv("data/413521520.csv",encoding='gbk'))
df["updatetime"] = pd.to_datetime(df["updatetime"])

x = df["updatetime"].values[:10].astype(np.float32)

y1 = df["经度"].values[:10].astype(np.float32)

y2 = df["纬度"].values[:10].astype(np.float32)


x = x.reshape(len(x),1)
y1 = y1.reshape(len(x),1)
y2 = y2.reshape(len(x),1)

#x = range(20)

# 进行归一化
sca = MinMaxScaler()
x = sca.fit_transform(x)

sca_y = MinMaxScaler()
y1 = sca.fit_transform(y1)
# 进行模型训练和预测，由于预测值未进行归一化，
model = LinearRegression()
mo =model.fit(x,y1)
p = mo.score(x,y1)
pred = mo.predict(x)
plt.figure(figsize=(13,2))
plt.plot(x, y1,'r')
plt.plot(x, pred,'g')
plt.show()
print(np.sum(abs(pred-y1))/100)
# 如果要获取X的原型，使用如下方法即可
#X_Ori = sca.inverse_transform(X) 

