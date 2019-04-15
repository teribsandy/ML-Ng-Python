# coding: utf-8

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

path ='/Users/zhangying/Desktop/243-ML-Ng/ex1-linear regression/ex1data1.txt'

data = pd.read_csv(path, header=None, names=['Population', 'Profit'])
data.plot(kind='scatter',x='Population',y='Profit',figsize=(8,5))


def computeCost(X,y,theta):
    inner=np.power(((X*theta.T)-y),2)
    return np.sum(inner)/(2*len(X))
    #return np.sum( np.array( ((X*theta.T)-y) )**2 ) / (2*len(X))

data.insert(0,'Ones',1)
cols=data.shape[1] #data.shape (97,3), cols=3,列数
X=data.iloc[:,0:cols-1]  #X是dataframe
y=data.iloc[:,cols-1:cols] #y是dataframe


#computeCost函数中X，y，theta都是numpy矩阵，所以要转换
X=np.matrix(X.values) #X.value是array，X是matrix
y=np.matrix(y.values) #X.value是array，y是matrix
theta=np.matrix([0,0]) #theta输出为matrix([[0, 0]]), (1,2)

print (X.shape, theta.shape, y.shape) #(97, 2) (1, 2) (97, 1)
#print computeCost(X,y,theta) # 初始成本32.072733877455676

'''
def normalEqn(X,y): #正规方程 θ = (XT X)−1 XT y
    theta=np.linalg.inv(X.T * X) * X.T * y
    return theta
finaltheta=normalEqn(X,y) 

#[[-3.89578088]\n[ 1.19303364]] #matrix, shape为(2,1)
#在computeCost(X,y,theta)中，出现shape不匹配的错误
'''

def gradientDescent(X,y,theta,alpha,epochs):
    temp=np.matrix(np.ones(theta.shape))
    cost=np.zeros(epochs)

    for i in range(epochs):
        error=X*theta.T -y 

        #也可以不用for 循环，直接利用向量化一步求解
        #temp=theta-(alpha/len(X))* error.T * X
        for j in range(0,X.shape[1]): #j=0,1
            Xj=X[:,j]
            # theta[0,j]-=(alpha/len(X))*np.sum(np.multiply(error,Xj)) #wrong,当更新theta1时，error中的theta的theta0已经被更新过了
            temp[0,j]=theta[0,j]-(alpha/len(X))*np.sum(np.multiply(error,Xj)) 

        theta=temp
        cost[i]=computeCost(X,y,theta)

    return theta, cost

alpha=0.01
epochs=1000
finaltheta,cost=gradientDescent(X,y,theta,alpha,epochs) 
#矩阵[[-3.24140214  1.1272942 ]]   #ndarray    (1000,)    索引值999即第1000条的成本为4.51595550308


print (computeCost(X,y,finaltheta)) #4.51595550308


x = np.linspace(data.Population.min(), data.Population.max(), 100)
f = finaltheta[0, 0] + (finaltheta[0, 1] * x)

fig=plt.figure('graphs',figsize=(8,10))
ax=fig.add_subplot(211)
ax.plot(x,f,'r',label='Prediction')
ax.scatter(data.Population,data['Profit'],label='TrainingSet')
ax.legend(loc=1)
ax.set_xlabel('Population')
ax.set_ylabel('Profit')
ax.set_title('predicted profit vs. population size')


ax2=fig.add_subplot(212)
ax2.plot(np.arange(epochs),cost)
ax2.set_xlabel('Iterations')
ax2.set_ylabel('Cost')
ax2.set_title('Error vs. Training Epochs')

plt.show()

