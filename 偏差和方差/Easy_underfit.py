#coding:UTF-8

import numpy as np
import matplotlib.pyplot as plt
plt.ion()
from scipy import io
from scipy.optimize import minimize

path='/Users/zhangying/Desktop/243-ML-吴恩达/ex5-bias vs variance/ex5data1.mat'
data=io.loadmat(path)
X=data['X'] #(12,1)
y=data['y'].ravel() #(12,)
Xval=data['Xval'] #(21,1)
yval=data['yval'].ravel() #(21,)
Xtest=data['Xtest'] #(21,1)
ytest=data['ytest'].ravel() #(21,)


def plot_trainset(X,y):
    _=plt.figure(0)
    plt.scatter(X,y,c='k',marker='*')
    plt.xlabel('Change in water level (x)')
    plt.ylabel('Water flowing out of the dam (y)')

plot_trainset(X,y)


X=np.insert(X,0,1,axis=1)  #(12,2)
# Xval=np.insert(Xval,0,1,axis=1)  #(21,2)
Xval=np.c_[np.ones((21,1)),Xval] #(21, 2)
Xtest=np.concatenate((np.ones((21,1)),Xtest),axis=1) #(21, 2)


theta=np.ones(X.shape[1]) #(2,)

def Jcost(theta,X,y,l):
    error=X@theta - y           #(12,)
    _theta=theta[1:]  #(1,)
    #注意哦！像这样一维数组，np.sum(error**2) == error@error 
    #error@error表示vector inner product to itselves
    return ( np.sum(error**2) + l*np.sum(_theta**2) ) / (2*len(X))


# print (Jcost(theta,X,y,1))  
# l=0 303.9515255535976; l=1 303.9931922202643

def gradient(theta,X,y,l):
    error=X@theta - y
    _theta=theta[1:]  #(1,)
    first=(X.T@error)/len(X) #(2,)
    second=np.r_[np.zeros(1),_theta*(l/len(X))] #(2,)
    return first+second
    
# print (gradient(theta,X,y,1))
# l=0 [-15.30301567 598.16741084] ; l=1 [-15.303016; 598.250744] 


def optimizetheta(X,y,l):
    theta=np.ones(X.shape[1])
    res=minimize(fun=Jcost,x0=theta,args=(X,y,l),method='TNC',jac=gradient)
    return res.x



def plotfit(X,y,l):
    #set regularization parameter λ to zero. 
    #because regularization will not be incredibly helpful for a θ of such low dimension.
    ftheta=optimizetheta(X,y,0) #[13.08790348  0.36777923]
    f=ftheta[0]+ftheta[1]*X[:,1]
    _=plt.figure(0) #或者采用 _=plt.gcf() 都可以使散点图和拟合直线出现在同一画布上
    plt.plot(X[:,1],f)

plotfit(X,y,0) #拟合直线与散点图的关系反映出模型欠拟合，高偏差！


#可视化数据有时不那么容易实现，特别是在特征较多时！
#更好的方法是在学习曲线上绘制训练误差和验证误差，以诊断bias-variance问题从而调试学习算法。

def vectorcost(X,y,Xval,yval,l):
    #训练样本X从1开始逐渐增加，训练出不同的参数向量θ。
    #通过theta计算训练代价和交叉验证代价，切记此时不要使用正则化，将 λ=0
    #计算训练代价需分为子集
    #计算交叉验证代价时记得整个交叉验证集来计算，无需分为子集
    Jtrain=np.zeros(len(X))
    Jcv=np.zeros(len(X))

    mlist=range(1,len(X)+1)

    for m in mlist:         # m=1,2,3,...,10,12
        print('m={}'.format(m))
        thetaopt=optimizetheta(X[:m,:],y[:m],l)
        tc=Jcost(thetaopt,X[:m,:],y[:m],0)
        cvc=Jcost(thetaopt,Xval,yval,0)
        Jtrain[m-1]=tc
        Jcv[m-1]=cvc
        print('tc={}, cv={}'.format(tc, cvc))

    return Jtrain,Jcv

vectorcost(X,y,Xval,yval,0)


def plotlearningcurve(X,y,Xval,yval,l):
    x=np.arange(1,len(X)+1)
    Jtrain,Jcv=vectorcost(X,y,Xval,yval,l)
    _=plt.figure(1)
    plt.plot(x,Jtrain,'r--',label='training cost')
    plt.plot(x,Jcv,'g-',label='cv cost')
    plt.legend()
    plt.xlabel('Number of training examples')
    plt.ylabel('Error')
    plt.title('Learning curve for Linear Regression')
    plt.grid(True)

plotlearningcurve(X,y,Xval,yval,0)


plt.ioff()
plt.show()


