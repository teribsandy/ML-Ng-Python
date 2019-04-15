# coding: utf-8

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

data=pd.read_csv('/Users/zhangying/Desktop/243-ML-吴恩达/ex2-logistic regression/ex2data1.txt',header=None,names=['Exam1','Exam2','Class'])
#print data.head() #data是DataFrame #data.Class是Series

#data.Class.isin(['1'])是Series,布尔值False True (Class=1则为True，Class=0则为False)
positive=data[ data.Class.isin(['1']) ] #positive是DataFrame，只是把Class全为1的数据取出来了，行索引来自于data
#print positive.head()
negative=data[ data.Class.isin(['0']) ]

# fig,ax=plt.subplots(figsize=(8,5))
# ax.scatter(positive.Exam1,positive['Exam2'],color='black',marker='+',label='Admitted')
# ax.scatter(negative['Exam1'],negative.Exam2,c='y',marker='o',label='Not admitted')
# ax.set_xlabel('Exam1 Score')
# ax.set_ylabel('Exam2 Score')
# ax.set_title('scatter_binaryclassification')
# ax.legend(loc=1) #图例显示在右上角
# plt.show()
def normalize_feature(df):
    return df.apply(lambda column : (column-column.mean())/column.std())
data=normalize_feature(data)    
#或者采用👇也可以特征归一化
#data=(data-data.mean())/data.std() #特征归一化

data.insert(0,'Ones',1)
X=data.iloc[:,0:3]
y=data.iloc[:,3] #若y=data.iloc[:,3:4],则y=np.array(y.values)类型虽然也是ndarray，但形状是(100,1),在cost函数计算时会有问题

X=np.array(X.values) #ndarray, (100,3) 
y=np.array(y.values) #ndarray, (100,)  

theta=np.zeros(3) #ndarray, (3,)


def sigmoid(theta,X):
    z=X@theta
    return 1/(1+np.exp(-z))

originalhypothesis=sigmoid(theta,X) #ndarray，shape为（100,），初始值中元素值均为0.5
# print (type(originalhypothesis),originalhypothesis.shape)

def cost(theta,X,y):
    hypothesis=sigmoid(theta,X)
    a=y*np.log(hypothesis) + (1-y)*np.log(1-hypothesis) #ndarray (100,) 计算初始成本下，a中每个元素都是-0.6931
    return (-1./len(X))*np.sum(a)  

originalcost= cost(theta,X,y) #0.69314718056, <class 'numpy.float64'> () 
print (originalcost)
# print (type(originalcost),originalcost.shape)


def gradient(theta,X,y):
    error=sigmoid(theta,X)-y #(100,)
    return (1.0/len(X))*X.T@error #(3,)
#print (gradient(theta,X,y)) #[ -0.1  -12.00921659  -11.26284221]

import scipy.optimize as opt

res=opt.minimize(fun=cost,x0=theta,args=(X,y),method='TNC',jac=gradient)
ftheta=res.x
print (res)
#fun: 0.20349770158947456 即最小成本
#jac: array([9.12848998e-09, 9.69677383e-08, 4.84144623e-07]) 即求得finaltheta前的偏导
# x: array([-25.16131865,   0.20623159,   0.20147149]) 即finaltheta

'''
def predict(theta,X):
    probability=sigmoid(theta,X) 
    return [1 if x>=0.5 else 0 for x in probability]

predictions=predict(ftheta,X)
correct=[1 if a==b else 0 for (a,b) in zip(predictions,y)]
accuracy=sum(correct)/len(X)
#print (accuracy) #0.89
'''

'''也可以用skearn中的方法来检验预测精度
from sklearn.metrics import classification_report
print (classification_report(y,predictions))
'''

'''
x1=np.arange(130,step=0.1)
x2=-(ftheta[0]+ftheta[1]*x1)/ftheta[2]

fig,ax=plt.subplots(figsize=(8,5))
ax.scatter(positive.Exam1,positive['Exam2'],color='black',marker='+',label='Admitted')
ax.scatter(negative['Exam1'],negative.Exam2,c='y',marker='o',label='Not admitted')
ax.plot(x1,x2)
ax.set_xlim(20,110)
ax.set_ylim(20,110)
ax.set_xlabel('Exam1 Score')
ax.set_ylabel('Exam2 Score')
ax.set_title('Decision Boundary')
ax.legend(loc=1) #图例显示在右上角
plt.show()

'''