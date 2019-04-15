#coding:UTF-8

from scipy import io 
import numpy as np
import matplotlib.pyplot as plt
import skimage 
from mpl_toolkits.mplot3d import Axes3D 

A=io.loadmat('/Users/zhangying/Desktop/243-ML-Ng/ex7-kmeans and PCA/data/bird_small.mat')['A'] #(128,128,3) 
A=A/255.

X=A.reshape(A.shape[0]*A.shape[1],A.shape[2])         #(16384,3) 
#共有128*128个像素，每个像素被表示为 3个 8位无符号整数(从0到255)，指定了红、绿和蓝色的强度值。这种编码通常被称为RGB编码。
#把原始图片的每个像素看作一个数据样本，然后利用K-means算法去找分组最好的16种颜色。

#从样本集X中选出K个初始质心
def initCentroids(X,K):
    m=X.shape[0]
    index=np.random.choice(m,K)
    centroids=X[index]
    return centroids                                   #（16，3）

#为每个样本找到离他最近的质心
def findClosestCentroids(X,centroids):
    idx=[]
    max_dist=1000000000 
    for i in range(len(X)):
        minus=X[i]-centroids #numpy's broadcasting!   X[i]的shape为(3,) centroids的shape为（16,3)
        dist=np.sum(np.power(minus,2),axis=1) 
        if dist.min()<max_dist:
            ci=np.argmin(dist)
            idx.append(ci)
    return np.array(idx)

#更新质心的位置
def computeCentroids(X,idx):
    centroids=[]
    for i in range(16): 
        u_k=X[idx==i].mean(axis=0) #(3,)
        centroids.append(u_k)
    return np.array(centroids)   #(16,3)

#找到最后的质心，为每个样本找到最终属于的质心，
def runKmeans(X,init_centroids,max_iters):
    Rcentroids=[]
    centroids=init_centroids

    for _ in range(max_iters+1):

        Rcentroids.append(centroids)

        idx=findClosestCentroids(X,centroids)
        centroids=computeCentroids(X,idx)
    
    return idx,Rcentroids


init_centroids=initCentroids(X,16)  #(16,3)
idx,Rcentroids=runKmeans(X,init_centroids,10)
centroids=Rcentroids[-1]


#🀄️可视化3D空间中最终像素分配。每个数据点都根据其分配的群集着色。因此这里我们需要知道 idx
ax=plt.subplot(121,projection='3d')
ax.scatter3D(X[:,0],X[:,1],X[:,2],s=5,c=idx,alpha=0.3,cmap='rainbow') #将scatter3D替换成scatter也是可以滴！


#🀄️由于在3维或更大维度上可视化数据集可能很麻烦，因此通常希望仅以丢失一些信息为代价以2D显示数据。
#在实践中，PCA通常用于减少数据的维度以用于可视化目的。PCA投影可以被认为是旋转，其选择最大化数据传播的视图，其通常对应于“最佳”视图。

means=X.mean(axis=0)        #(3,)
stds=X.std(axis=0,ddof=1)   #(3,)
Xnorm= (X-means)/stds       #(16384,3)

def getsvd(X):
    sigma=(X.T@X)/len(X)    #(3,3)
    U,S,V=np.linalg.svd(sigma)
    return U,S,V

#共有3个主成分，每一列作为一个主成分，每个主成分都是一个长度为3的向量
U,S,V=getsvd(Xnorm)


def projectData(X,U,K):
    Ureduce=U[:,:K]
    Z=X@Ureduce
    return Z

#从U中选出2个最重要的分量，即前2个特征向量作为Uredeuce
Z=projectData(Xnorm,U,2) #(16384,2)
ax=plt.subplot(122)
ax.scatter(Z[:,0],Z[:,1],s=5,c=idx,alpha=0.5,cmap='rainbow') #有个疑问🤔️平面图的横纵坐标范围和pdf对不上
plt.show()


#🀄️如何显示压缩后的图像呢？将只包含16种颜色的图片reshape并展示
#怎样才算压缩了呢？确定最终质心后，即存储16个选中颜色的RGB值后，对于图中的每个像素，将他所对应的质心指定好
newX=np.zeros(X.shape)                
for i in range(16):
    newX[idx==i]=centroids[i] #(16384,3) 

newA=newX.reshape(A.shape)    #(128,128,3) 
plt.imshow(newA)
plt.show()
