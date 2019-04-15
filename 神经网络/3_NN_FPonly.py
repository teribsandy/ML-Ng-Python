#coding: UTF-8
import numpy as np
import matplotlib.pyplot as plt
from scipy.io import loadmat
from sklearn.metrics import classification_report

def load_data(path):
    data=loadmat(path)
    return data['Theta1'],data['Theta2']

theta1,theta2=load_data('/Users/zhangying/Desktop/243-ML-Ng/ex3-neural network/ex3weights.mat')
#ndarray(25, 401) ndarray(10, 26)

data=loadmat('/Users/zhangying/Desktop/243-ML-Ng/ex3-neural network/ex3data1.mat')
X=data['X'] #(5000,400)
y=data['y'] #(5000,1)
y=y.ravel() #(5000,)


def g(z):
    return 1./(1+np.exp(-z))

a1=np.insert(X,0,1,axis=1) #(5000,401)

#法1
# z2=a1@theta1.T
# a2=np.insert(g(z2),0,1,axis=1) #a2的第一列的值为1

#法2
z2=np.insert(a1@theta1.T,0,1,axis=1)
a2=g(z2) #print (a2[:,0]) #a2的第一列的值为0.73105858

a3=g(a2@theta2.T) #a2(5000,26) a3(5000,10)

hargmax=np.argmax(a3,axis=1) #(5000,)
hargmax=hargmax+1

accuracy=np.mean(hargmax==y)
#或者 accu=np.mean([1 if a==b else 0 for (a,b) in zip(hargmax,y)])
print ('准确率为{}%'.format(accuracy*100)) #有个疑问，在隐藏层的前向传播中采用两种方法所得准确率都是97.52%

#OR采用sklearn中的分类评估方法给出精确率、召回率与F值等
#print (classification_report(y,hargmax))