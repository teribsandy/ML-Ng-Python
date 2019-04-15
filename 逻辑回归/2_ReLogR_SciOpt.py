# https://blog.csdn.net/Cowry5/article/details/80247569#commentsedit
# 含正则化项的逻辑回归

# coding: utf-8

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

data = pd.read_csv('/Users/zhangying/Desktop/243-ML-吴恩达/ex2-logistic regression/ex2data2.txt', names=['test1', 'test2', 'passed'])

def plotdata():
    positive = data[data.passed.isin(['1'])]  # 1
    negetive = data[data.passed.isin(['0'])]  # 0
    ax=plt.subplot(111)
    ax.scatter(positive['test1'],positive['test2'],c='r',marker='o',label='passed')
    ax.scatter(negetive['test1'],negetive['test2'],color='k',marker='x',label='failed')
    ax.set_xlabel('test1 score')
    ax.set_ylabel('test2 score')
    ax.legend(loc=1)
    

#.as_matrix will be removed in a future version; Use .values instead.
x1=data['test1'].values #ndarray (118,)
x2=data['test2'].values
y=data['passed'].values

'''由散点图可知正负两类数据并没有线性的决策界限
只有获得更多的特征进行逻辑回归得到的决策界限才可以是高阶函数的形状
因此有必要通过特征映射为每个样本数据创建更多的特征,即扩大特征空间,得到新数据'''

def feature_mapping(x1,x2,power):
    data={}
    for i in np.arange(power+1):
        for p in np.arange(i+1):
            data['f{}{}'.format(i-p,p)]=np.power(x1,i-p)*np.power(x2,p)
    return pd.DataFrame(data)

Xdata=feature_mapping(x1,x2,power=6) #(118, 28) #特征数28的计算方式（（0+1）+（power+1））*（power+1） / 2
# print (X.head())
#    f00      f01      ...                f51           f60
# 0  1.0  0.69956      ...       2.477505e-07  1.815630e-08
# 1  1.0  0.68494      ...      -4.699318e-06  6.362953e-07
# 2  1.0  0.69225      ...      -3.085938e-04  9.526844e-05
# 3  1.0  0.50219      ...      -3.724126e-03  2.780914e-03
# 4  1.0  0.46564      ...      -1.658422e-02  1.827990e-02

X=Xdata.values
theta=np.zeros(X.shape[1])

#print (X.shape, y.shape, theta.shape) #(118, 28) (118,) (28,)

'''
特征数量过多，会出现过拟合问题
采用正则化，即保留所有的特征，在代价函数中加入theta1至thetaN的惩罚项（注意噢！theta0是不惩罚的）
由于是要最小化代价函数，如果相应参数的惩罚项lambda很大，该参数自然就很小甚至为0
不就达到了在假设函数中保留了对应特征但实际却不起什么影响的效果嘛～
'''

def sigmoid(z):
    return 1 / (1 + np.exp(- z))

def cost(theta, X, y):
    first = (-y) * np.log(sigmoid(X@theta))
    second = (1 - y)*np.log(1 - sigmoid(X@theta))
    return np.mean(first - second)

def costReg(theta,X,y,l):
    _theta=theta[1:] #(27,)
    reg=(l/2*len(X))*np.sum(np.power(_theta,2))
    return cost(theta,X,y)+reg

#print (costReg(theta, X, y,l=1)) # 0.6931471805599453

def gradient(theta,X,y): # <class 'numpy.ndarray'> (28,)
    error=sigmoid(X@theta)-y 
    return (X.T@error)/len(X)

def gradientReg(theta, X, y,l): # <class 'numpy.ndarray'> (28,)
    reg=(float(l)/len(X))*theta
    reg[0]=0
    return gradient(theta,X,y)+reg

# [8.47457627e-03 7.77711864e-05 3.76648474e-02 2.34764889e-02
#  3.93028171e-02 3.10079849e-02 3.87936363e-02 1.87880932e-02
#  1.15013308e-02 8.19244468e-03 3.09593720e-03 4.47629067e-03
#  1.37646175e-03 5.03446395e-02 7.32393391e-03 1.28600503e-02
#  5.83822078e-03 7.26504316e-03 1.83559872e-02 2.23923907e-03
#  3.38643902e-03 4.08503006e-04 3.93486234e-02 4.32983232e-03
#  6.31570797e-03 1.99707467e-02 1.09740238e-03 3.10312442e-02]

import scipy.optimize as opt
result=opt.minimize(fun=costReg, x0=theta, args=(X, y,2), method='TNC', jac=gradientReg)
#fun: 0.6931338399526238
# x: array([ 1.93620103e-04,  1.90650272e-04, -1.78287414e-04, -2.82106430e-05,
#        -1.76071099e-04, -8.56474729e-05, -1.53402592e-04,  8.29915905e-05,
#        -1.00170947e-04, -4.71876006e-05, -2.51592152e-05, -3.32937951e-05,
#        -9.91930049e-06, -3.17292996e-04, -4.64635135e-05, -9.10808671e-05,
#        -3.80481848e-05, -4.84158421e-05,  7.24123582e-06, -8.34613169e-06,
#        -1.31290587e-05,  2.89849567e-07, -2.31413740e-04, -2.91679650e-05,
#        -4.54035084e-05, -4.06345675e-05,  1.32689736e-06, -1.63216161e-04])

ftheta=result.x

def predict(theta,X):
    probability=sigmoid(X@theta) 
    return [1 if x>=0.5 else 0 for x in probability]

predictions=predict(ftheta,X)

'''评估预测的准确度'''
correct=[1 if a==b else 0 for (a,b) in zip(predictions,y)]
accuracy=sum(correct)/len(X)  #0.8050847457627118

# from sklearn.metrics import classification_report
# print (classification_report(y,predictions)) #0.82

'''事实上上面的建模过程在scikit-learn中的Linear_model中已经被封装
给定正则化因子lambda后就会建立相应的逻辑回归模型
引入测试集数据X，y，模型就会去拟合这些数据
然后根据拟合结果与y对比，从而评估预测的准确度
'''
from sklearn import linear_model
model=linear_model.LogisticRegression(penalty='l2',C=2.)
model.fit(X,y)
print(model.score(X,y)) #0.8220338983050848


'''画图：决策边界'''
#等高线特别适合高维决策边界的绘制
#为什么是利用等高线绘出决策边界呢？ 已知假设函数为theta.T@x，得到的立体的三维曲线
#不妨想象成一个不规则的碗扣在二维坐标轴的正上方，我们可以在碗上画出一圈圈的等高线
#高度为0的那一条线，与坐标轴平面相交，不就是theta.T@x等于0的那一条线吗！
#也就是能解方程把x2表示成x1的关系式，正是我们想要的决策边界呀！
#np.contour(X,Y,Z)的维度的要求：
#X and Y must both be 2-D with the same shape as Z
#or they must both be 1-D such that len(X) is the number of columns in Z and len(Y) is the number of rows in Z.

u=np.linspace(-1,1.5,250)                       #ndarray (250,)
a,b=np.meshgrid(u,u)                     #ndarray (250,250) #ndarray (250,250)
c=feature_mapping(a.ravel(),b.ravel(),6).values #ndarray (62500,28)
c=c@ftheta                                      #ndarray (62500,)
c=c.reshape(a.shape)                     #ndarray (250,250)

plotdata()
plt.contourf(a,b,c,0,alpha=0.3,cmap=plt.cm.coolwarm)
plt.contour(a,b,c,0,colors='green',linewidth=1.)
# C=plt.contour(a,b,c,0,colors='green',linewidth=1.)
# plt.clabel(C,inline=True,fontsize=10)
plt.show()
plt.contour()


