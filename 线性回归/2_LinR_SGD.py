# coding: utf-8
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

path ='/Users/zhangying/Desktop/243-ML-Ng/ex1-linear regression/ex1data2.txt'

data = pd.read_csv(path, header=None, names=['Size', 'Bedrooms','Price']) #data的shape为(47,3) 类型为DataFrame

x=data['Size'] #x的shape为(47,) x的类型为Series
y=data['Bedrooms']
z=data['Price']
ax=plt.subplot(211,projection='3d')
ax.scatter3D(x,y,z,cmap='rainbow') #scatter3D中xyz数据的类型为array或series
ax.set_xlabel('Size'),ax.set_ylabel('Bedrooms'),ax.set_zlabel('Price'),ax.set_title('MLR_Scatter')


# 特征归一化超级重要！由于我之前没有特征归一化，多处报错！
#.apply() 作用于dataframe上，用于对row或者column进行计算(根据axis指定，默认axis=0)
#.applymap()作用于dataframe上的每一个元素，是元素级别的操作
# #.map()作用于series上，是元素级别的操作
def normalize_feature(df):
    return df.apply(lambda column : (column-column.mean())/column.std())
data=normalize_feature(data)    
#或者采用👇也可以特征归一化
#data=(data-data.mean())/data.std() #特征归一化

data.insert(0,'Ones',1) #data的shape为(47,4)
cols=data.shape[1] #data.shape (47,4), cols=4,列数
X=data.iloc[:,0:cols-1]  #X是dataframe，代表前3列的数据，(47,3)
y=data.iloc[:,cols-1:cols] #y是dataframe，代表Price列的数据，(47,1)

X=np.matrix(X.values) #X.value是array，X是matrix     X.shape(47, 3)
y=np.matrix(y.values)  # y.shape  (47, 1)
theta=np.matrix([0,0,0]) #theta输出为matrix([[0,0,0]])    theta.shape(1, 3)


#computeCost函数中X，y，theta都是numpy矩阵
def computeCost(X,y,theta):
    #inner=np.power(((X*theta.T)-y),2)
    inner=np.array((X*theta.T)-y)**2 #如果用**2的方式表示a中的元素平方，那么a必须是square array
    return np.sum(inner)/(2*len(X))

#print (computeCost(X,y,theta)) # 特征归一化前的初始成本65591548106  特征归一化后的初始成本0.489361702128


def gradientDescent(X,y,theta,alpha,epochs):
    temp=np.matrix(np.ones(theta.shape))
    cost=np.zeros(epochs)

    for i in range(epochs):
        error=X*theta.T -y

        #用for 循环   Or    直接利用向量化一步求解
        temp=theta-(alpha/len(X))* error.T * X
        # for j in range(0,X.shape[1]): #j=0,1,2
        #     Xj=X[:,j]
        #     # theta[0,j]-=(alpha/len(X))*np.sum(np.multiply(error,Xj)) #wrong,当更新theta1时，error中的theta的theta0已经被更新过了
        #     temp[0,j]=theta[0,j]-(alpha/len(X)) * np.sum( np.multiply(error,Xj) ) 

        theta=temp
        cost[i]=computeCost(X,y,theta)

    return theta, cost

alpha=0.01
epochs=1000

finaltheta,cost=gradientDescent(X,y,theta,alpha,epochs) 


#print (finaltheta, computeCost(X,y,finaltheta))
#[[-1.10995657e-16  8.78503652e-01 -4.69166570e-02]] 
#0.130703369608

# fig2,ax2=plt.subplots(figsize=(12,8))
# ax2.plot(np.arange(epochs),cost,'r')
# ax2.set_xlabel('Iterations')
# ax2.set_ylabel('Cost')
# ax2.set_title('Error vs. Training Epochs')
# plt.show()

ftheta0=np.array(finaltheta)[0,0]
ftheta1=np.array(finaltheta)[0,1]
ftheta2=np.array(finaltheta)[0,2]

x1 = np.linspace(data.Size.min(), data.Size.max(), 100)
x2 = np.linspace(data.Bedrooms.min(), data.Bedrooms.max(), 100)
f = ftheta0 + (ftheta1 * x1) + (ftheta2 * x2)

ax3=plt.subplot(212,projection='3d')
ax3.plot(x1,x2,f,'r',label='Hypothesis')
ax3.scatter(data.Size,data.Bedrooms,data['Price'],label='TrainingSet')
ax3.legend(loc=1)
ax3.set_xlabel('Size'),ax3.set_ylabel('Bedrooms'),ax3.set_zlabel('Price'),ax3.set_title('predicted price vs. size&bedrooms')
plt.show()





