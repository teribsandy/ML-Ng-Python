#coding: UTF_8

import numpy as np
import matplotlib.pyplot as plt
from scipy import io

'''获取样本'''
data=io.loadmat('/Users/zhangying/Desktop/243-ML-Ng/ex7-kmeans and PCA/data/ex7data2.mat')
X=data['X']  #(300,2)


'''可视化'''
def visualize(X,s,c,marker,fnum):
    x1=X[:,0]
    x2=X[:,1]
    _=plt.figure(fnum)
    plt.scatter(x1,x2,s=s,c=c,marker=marker,cmap='rainbow')

# 由散点图看出K=3（对于特征高维的数据，难以直接可视化，K可以通过elbow method处确定，但大部分情况下还是从实际需求出发确定K）
visualize(X,10,np.ones(300),'o',0)


'''随机初始化聚类中心'''
#随机初始化容易got bad luck！
#通常要进行50-1000次随机初始化，并分别进行聚类，最后选取令J最小的init_mu_k
def initcentroids(K,X):
    mu_k=[X[i] for i in np.random.choice(300,3)] 
    mu_k=np.array(mu_k)   #(3,2)
    return mu_k

init_mu_k=initcentroids(3,X)


def distance(a,b):
    squa_dist=(a-b)@(a-b).T #a和b的shape为(2,)
    return squa_dist


def findclosetcentriods(X,mu_k):
    m=len(X)
    K=len(mu_k)
    c_index=np.zeros(m)
    for i in range(m):
        temp_dist=1000
        temp_j=10
        for j in range(K):
            dist=distance(X[i],mu_k[j])
            if dist<temp_dist:
                temp_dist=dist
                temp_j=j
        c_index[i]=temp_j
    return c_index

'''可视化第一次聚类'''
visualize(init_mu_k,80,np.arange(3),'+',1)
visualize(X,10,findclosetcentriods(X,init_mu_k),'o',1)


def updatemuk(X,mu_k):
    class_of_sample=findclosetcentriods(X,mu_k)  #（300，） 取值为0或1或2
   
    samples_0=X[np.where(class_of_sample==0)] #返回属于mu_0的样本 ndarray,ndim=2
    samples_1=X[np.where(class_of_sample==1)] #返回属于mu_1的样本 ndarray,ndim=2
    samples_2=X[np.where(class_of_sample==2)] #返回属于mu_2的样本 ndarray,ndim=2

    new_mu_0=np.mean(samples_0,axis=0) #(2,)
    new_mu_1=np.mean(samples_1,axis=0) #(2,)
    new_mu_2=np.mean(samples_2,axis=0) #(2,)
    new_mu_k=np.r_[new_mu_0.reshape(1,-1),new_mu_1.reshape(1,2),new_mu_2.reshape(1,2)]
    return new_mu_k

new_mu_k=updatemuk(X,init_mu_k)


'''可视化第二次聚类'''
visualize(new_mu_k,80,np.arange(3),'+',2)
visualize(X,10,findclosetcentriods(X,new_mu_k),'o',2)

plt.show()