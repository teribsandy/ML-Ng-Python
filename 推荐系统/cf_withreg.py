#coding:utf-8

import numpy as np
from scipy import io

data=io.loadmat('/Users/zhangying/Desktop/243-ML-Ng/ex8-anomaly detection and recommendation/data/ex8_movies.mat')
Y,R=data['Y'],data['R'] #(1682,943) (1682,943)

# print (np.sum(R)) #100000 共有10w条评分记录，不过比起来总数158w多来可以知道用户电影评分矩阵有很多空白处，我们要做的就是利用低秩重构进行矩阵填充

params=io.loadmat('/Users/zhangying/Desktop/243-ML-Ng/ex8-anomaly detection and recommendation/data/ex8_movieParams.mat')
X,Theta=params['X'],params['Theta']
# (1682, 10) (943, 10)


def data(X,Theta,Y,R,nm=1682,nu=943,nf=10):
    X=X[:nm,:nf]
    Theta=Theta[:nu,:nf]
    Y=Y[:nm,:nu]
    R=R[:nm,:nu]
    return X,Theta,Y,R


def params_vector(X,Theta):
    return np.r_[X.ravel(),Theta.ravel()]


def params_seperate(XTvector,nm=1682,nu=943,nf=10):
    X=XTvector[:nm*nf].reshape(nm,nf)
    Theta=XTvector[nm*nf:].reshape(nu,nf)
    return X,Theta


def Cost(XTvector,Y,R,nm=1682,nu=943,nf=10,l=0):
    X,Theta=params_seperate(XTvector,nm,nu,nf)

    # ratedindex=np.where(R.ravel()==1)[0]
    # y=Y.ravel()[ratedindex]
    # h=(X@Theta.T).ravel()[ratedindex]
    # cost=.5*np.sum((h-y)**2)
    # reg=(l/2.)*(np.sum(X**2)+np.sum(Theta**2))
    

    #虽然结果是一样的，但是从简便性上来说，强推👇这种哇，谁让R这个矩阵的元素是二进制的呢，要好好利用这个特点
    H=(X@Theta.T)*R
    cost=0.5*np.sum((H-Y)**2)
    reg=(l/2.)*(np.sum(X**2)+np.sum(Theta**2))

    return cost+reg


def gradient(XTvector,Y,R,nm=1682,nu=943,nf=10,l=0):
    X,Theta=params_seperate(XTvector,nm,nu,nf)

    error=(X@Theta.T)*R -Y #(1682,943)
    Xgrad=error@Theta+l*X      #(1682,10)
    Thetagrad=error.T@X+l*Theta   #(943,10)

    gradvector=np.r_[Xgrad.ravel(),Thetagrad.ravel()]
    return gradvector      #(26250,)


Xsub,Thetasub,Ysub,Rsub=data(X,Theta,Y,R,5,4,3)
XTvectorsub=params_vector(Xsub,Thetasub)
costsub=Cost(XTvectorsub,Ysub,Rsub,5,4,3,1.5)  
gradvectorsub=gradient(XTvectorsub,Ysub,Rsub,5,4,3,1.5)

print (costsub,'\n',gradvectorsub)
#learningrate=0时cost为22.224603725685675;learningrate=1时cost为31.344056244274213


def checkGradient(XTvector,Y,R,nm=1682,nu=943,nf=10,l=0):
    gradvector=gradient(XTvector,Y,R,nm,nu,nf,l)

    l=len(XTvector)
    evec=np.zeros(l)

    idx=np.random.choice(l)
    evec[idx]=0.0001

    loss1=Cost(XTvector+evec,Y,R,nm,nu,nf,l)
    loss2=Cost(XTvector-evec,Y,R,nm,nu,nf,l)

    numgrad=(loss1-loss2)/(2*0.0001)
    difference=(numgrad-gradvector[idx])/(numgrad+gradvector[idx])

    return difference


print (checkGradient(XTvectorsub,Ysub,Rsub,5,4,3,1.5)) #随机值-1.4292438337096058e-09 \ 8.379931319177953e-09 \ ...

