#coding:UTF-8
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.io import loadmat

def findClosestCentroids(X,centroids):
    """output a 1D array idx that holds the index of the closest centroid to every training example."""
    idx=[]
    max_dist=10000
    for i in range(len(X)):
        minus=X[i]-centroids #here use numpy's broadcasting
        #X[i] (2,) 
        #centroids (3,2) 
        #minus (3,2)
        dist=np.sum(np.power(minus,2),axis=1) #(3,)
        if dist.min()<max_dist:
            ci=np.argmin(dist)
            idx.append(ci)
    return np.array(idx)

'''ex7data2.mat实现'''
mat=loadmat('/Users/zhangying/Desktop/243-ML-Ng/ex7-kmeans and PCA/data/ex7data2.mat')
X=mat['X']
init_centroids=np.array([[3,3],[6,2],[8,5]])
idx=findClosestCentroids(X,init_centroids)
# print (idx[0:3]) #[0,2,1]

def computeCentroids(X,idx):
    centroids=[]
    for i in range(len(np.unique(idx))): #i=0,1,2
        u_k=X[idx==i].mean(axis=0) #X[idx==0] 假如idx中索引为4和5的元素为0，则X中行索引为4和5的那两行被取出
        centroids.append(u_k)
    return np.array(centroids)
# print (computeCentroids(X,idx))
#array([[2.42830111, 3.15792418],
    #    [5.81350331, 2.63365645],
    #    [7.11938687, 3.6166844 ]])

def plotData(X,Rcentroids,idx=None):
    """可视化数据，并自动分开着色。
    idx: 最后一次迭代生成的idx向量，存储每个样本分配的簇中心点的值
    Rcentroids: 包含每次中心点历史记录"""

    colors = ['b','g','gold','darkorange','salmon','olivedrab', 
              'maroon', 'navy', 'sienna', 'tomato', 'lightgray', 'gainsboro'
             'coral', 'aliceblue', 'dimgray', 'mintcream', 'mintcream']
    
    #Rcentroids[0] #第一次的中心点们 (3,2)

    assert len(Rcentroids[0]) <= len(colors) #测试语句：raise-if-not

    subX=[] #将属于同一类的样本打包存放进subX
    if idx is not None:
        for i in range(Rcentroids[0].shape[0]):
            x_i=X[idx==i]
            subX.append(x_i)
    else:
        subX=[X]
    
    plt.figure(figsize=(8,5))
    for i in range(len(subX)):
        xx=subX[i]
        plt.scatter(xx[:,0],xx[:,1],c=colors[i],label='Cluster %d'% i)
    plt.legend()
    plt.grid(True)
    plt.xlabel('x1',fontsize=14)
    plt.ylabel('x2',fontsize=14)
    plt.title('Plot of X Points',fontsize=16)

    xx,yy=[],[] #记录下每次 K个质心 的 横纵坐标
    for centroid in Rcentroids:
        xx.append(centroid[:,0])
        yy.append(centroid[:,1])
    
    plt.plot(xx,yy,'rx--',markersize=8)

# plotData(X,[init_centroids])

def runKmeans(X,init_centroids,max_iters):
    Rcentroids=[]
    Rcentroids.append(init_centroids)
    
    centroids=init_centroids
    for i in range(max_iters):
        idx=findClosestCentroids(X,centroids)
        centroids=computeCentroids(X,idx)
        Rcentroids.append(centroids)
    
    return idx,Rcentroids

idx,Rcentroids=runKmeans(X,init_centroids,20)
# plotData(X,Rcentroids,idx)

def initCentroids(X,K):
    '''Random Initialization'''
    m=X.shape[0]
    index=np.random.choice(m,K)
    centroids=X[index]
    return centroids

'''进行三次随机初始化，看下各自的效果'''
for i in range(3):
    init_centroids=initCentroids(X,3)
    idx,Rcentroids=runKmeans(X,init_centroids,10)
    plotData(X,Rcentroids,idx) 
    plt.show()
