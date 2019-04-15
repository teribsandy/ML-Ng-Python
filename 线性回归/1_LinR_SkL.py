# coding: utf-8

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn import linear_model

'''准备数据'''
data=pd.read_csv('/Users/zhangying/Desktop/243-ML-吴恩达/ex1-linear regression/ex1data1.txt',header=None,names=['Population','Profit'])
data.insert(0,'Ones',1) #data.shape (97, 3)  type(data): DataFrame

X=data.iloc[:,0:2] #type(X): DataFrame
y=data.iloc[:,2] #type(y): Series    y=data.Profit

x=X.iloc[:,1] #type(x): Series     x=data.Population

'''拟合模型''' 
model=linear_model.LinearRegression()
model.fit(X,y)

fX=model.predict(X) #fX是ndarray, ndim=1,shape=(97,)

'''画图'''
fig,ax=plt.subplots(figsize=(8,5))
ax.plot(x,fX,'r',label='PredictionLine')
ax.scatter(x,y,label='TrainingData')
ax.legend(loc=2)
ax.set_xlabel('Population')
ax.set_ylabel('Profit')
ax.set_title('Predicted Profit vs. Population Size')
plt.show()

