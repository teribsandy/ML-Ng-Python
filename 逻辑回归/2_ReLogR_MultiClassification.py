#coding: UTF-8
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.io import loadmat

def load_data(path):
    data=loadmat(path)
    X=data['X']
    y=data['y']
    return X,y

X,y=load_data('/Users/zhangying/Desktop/243-ML-吴恩达/ex3-neural network/ex3data1.mat')
#print (np.unique(y)) #看下有几类标签 [ 1  2  3  4  5  6  7  8  9 10]
#print (X.shape,y.shape) #ndarray(5000, 400) ndarray(5000, 1)


def plot_two_images(X):
    index1=np.random.randint(0,5000)
    index2=np.random.randint(0,5000)
    image1=X[index1,:]
    image2=X[index2,:]
    fig,(ax1,ax2)=plt.subplots(1,2)
    ax1.matshow(image1.reshape((20,20)),cmap='gray_r')
    ax1.set_title('Target: {}'.format(y[index1][0]))
    ax1.set_xticks([])
    ax1.set_yticks([])
    ax2.matshow(image2.reshape((20,20)),cmap='rainbow')
    ax2.set_title('Target: {}'.format(y[index2,0]))
    ax2.set_xticks([])
    ax2.set_yticks([])
    plt.show()
plot_two_images(X)

def plot_100_images(X):
    indexs=np.random.choice(np.arange(5000),100)
    images=X[indexs,:] #(100,400)
    fig,ax_array=plt.subplots(10,10,sharex=True,sharey=True)
    for row in range(10):
        for column in range(10):
            ax_array[row,column].matshow(images[row*10+column,:].reshape((20,20)),cmap='gray_r')
            ax_array[row,column].set_title('{}'.format(y[indexs[row*10+column],0]))
    plt.xticks([])
    plt.yticks([])
    plt.show()

plot_100_images(X)


def sigmoid(z):
    return 1./(1+np.exp(-z))

def regularized_cost(theta,X,y,l):
    h=sigmoid(X@theta)
    cost=np.mean(-y*np.log(h)-(1-y)*np.log(1-h))

#   _theta=theta
#   _theta[0]=0
    _theta=theta[1:]

    reg=(l/(2*len(X)))*np.sum(np.power(_theta,2))
    reg_cost=cost+reg
    return reg_cost

def regularized_gradient(theta,X,y,l):
    h=sigmoid(X@theta)
    gradient=(1./len(X))*(X.T@(h-y))

    # _theta=theta
    # _theta[0]=0
    # regterm=(l/len(X))*_theta

    _theta=theta[1:]
    regterm=(l/len(X))*_theta
    regterm=np.insert(regterm,0,0)
    
    return gradient+regterm

from scipy.optimize import minimize

def one_vs_all(X,y,l):
    alltheta=np.zeros((10,401)) 
    for i in range(1,11):
        y_i=np.array([1 if label==i else 0 for label in y])
        res=minimize(fun=regularized_cost,x0=theta,args=(X,y_i,l),method='TNC',jac=regularized_gradient,options={'disp':True})
        alltheta[i-1]=res.x
    return alltheta

def predict_all(X,alltheta):
    allh=sigmoid(X@alltheta.T) #(5000,10)
    h_argmax=np.argmax(allh,axis=1) #(5000,1)
    h_argmax=h_argmax+1
    return h_argmax

X=np.insert(X,0,values=1,axis=1) #(5000, 401)
y=y.ravel() #(5000,)
theta=np.zeros(401) #(401,)

thetaall=one_vs_all(X,y,1)
prediction=predict_all(X,thetaall)

# right=[1 if a==b else 0 for (a,b) in zip(prediction,y)]
# accuracy=np.mean(right)
accuracy=np.mean(prediction==y)
print ('准确率为{}%'.format(accuracy*100)) #实际答案应该是94.46%，但前一种正则化方法跑出来准确率为90.72%