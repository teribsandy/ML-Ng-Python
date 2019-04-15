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


def polyfeatures(X,p):
    originalX=X.copy()
    for i in range(2,p+1): #p=6 i=2,3,4,5,6
        PolyX=np.power(originalX,i)
        X=np.c_[X,PolyX]
    return X

X=polyfeatures(X,6) #(12,6)
Xval=polyfeatures(Xval,6) #(21,6)
Xtest=polyfeatures(Xtest,6)  #(21,6)


#关于归一化，所有数据集应该【都】用【训练集的均值和样本标准差】处理。
#把对数据归一化当作是训练模型的一个步骤， 模型训练好了之后，预测的时候当然要严格的遵循训练时的过程呀。
#预测的数据都是模型没有见过的，如果对这些新的数据用自己的均值和标准差进行处理，当然是不符合模型训练的过程的。

#注意这里是样本标准差而不是总体标准差
#使用np.std()时，将ddof=1则是样本标准差，默认=0是总体标准差。而pandas默认计算样本标准差。
trainmu=np.mean(X,axis=0)
trainsigma=np.std(X,axis=0,ddof=1)

def featureNormalize(X,mu,sigma):
    newX=np.zeros((X.shape[0],X.shape[1]))
    for i in range(X.shape[1]):
        newX[:,i]=(X[:,i]-mu[i])/sigma[i]
    return newX

X=featureNormalize(X,trainmu,trainsigma)  
Xval=featureNormalize(Xval,trainmu,trainsigma)
Xtest=featureNormalize(Xtest,trainmu,trainsigma)

X=np.insert(X,0,1,axis=1)  #(12,7)
Xval=np.insert(Xval,0,1,axis=1)  #(21,7)
Xtest=np.insert(Xtest,0,1,axis=1) #(21, 7)

theta=np.ones(X.shape[1]) #(7,)

# print (X[:3,:])
# print (Xval[:3,:])

def Jcost(theta,X,y,l):
    error=X@theta - y           #(12,)
    _theta=theta[1:]  #(7,)
    #注意哦！像这样一维数组，np.sum(error**2) == error@error 
    #error@error表示vector inner product to itselves
    return ( np.sum(error**2) + l*np.sum(_theta**2) ) / (2*len(X))


def gradient(theta,X,y,l):
    error=X@theta - y
    _theta=theta[1:]  #(6,)
    first=(X.T@error)/len(X) #(7,)
    second=np.r_[np.zeros(1),_theta*(l/len(X))] #(7,)
    return first+second


def optimizetheta(X,y,l):
    theta=np.ones(X.shape[1])
    res=minimize(fun=Jcost,x0=theta,args=(X,y,l),method='TNC',jac=gradient)
    return res.x


def plotfit(X,y,l):
    ftheta=optimizetheta(X,y,l) #(7,)
    x=np.linspace(-75,55,num=50)
    xx=x.reshape(-1,1) #(50,1)
    xx=polyfeatures(xx,6) #(50,6)
    xx=featureNormalize(xx,trainmu,trainsigma) #(50,6)
    xx=np.insert(xx,0,1,axis=1)  #(50,7)
    y_fit=xx@ftheta #(50,)
    _=plt.figure(0) #或者采用 _=plt.gcf() 都可以使散点图和拟合直线出现在同一画布上
    plt.plot(x,y_fit)

plotfit(X,y,0) #当lambda为0时，拟合曲线虽然很好地拟合了散点图，但complex and even drops off at the extremes. 
#indicates that the model is overfitting the training data and will not generalize well.

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


def plotlearningcurve(X,y,Xval,yval,l):
    x=np.arange(1,len(X)+1)
    Jtrain,Jcv=vectorcost(X,y,Xval,yval,l)
    _=plt.figure(1)
    plt.plot(x,Jtrain,'r--',label='training cost')
    plt.plot(x,Jcv,'g-',label='cv cost')
    plt.legend()
    plt.xlabel('Number of training examples')
    plt.ylabel('Error')
    plt.title('Learning curve for Polynomial Regression')
    plt.grid(True)

plotlearningcurve(X,y,Xval,yval,0)
#trainning error is too small，almost zero！And the gap between the training and cv errors also indicates a high variance problem.

plt.ioff()
plt.show()


