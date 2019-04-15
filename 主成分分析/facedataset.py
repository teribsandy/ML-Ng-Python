#coding:UTF-8

import numpy as np
import matplotlib.pyplot as plt
from scipy.io import loadmat
import skimage

data=loadmat('/Users/zhangying/Desktop/243-ML-Ng/ex7-kmeans and PCA/data/ex7faces.mat')
X=data['X'] #(5000,1024)

#load and visualize the first 100 images
def loadimages(X,rows,cols):
    images=X[0:rows*cols,:]  #前100张 (100,1024)  
    # images=X[np.random.choice(5000,100),:] #随机选出100张

    fig,ax_array=plt.subplots(rows,cols,sharex=True,sharey=True)
    
    for row in range(rows):
        for col in range(cols):
            ax_array[row][col].matshow(images[rows*row+col].reshape(32,32).T,cmap='gray')
    plt.xticks(())
    plt.yticks(())
    plt.show()
# loadimages(X,10,10)

means=X.mean(axis=0) #(1024,)
stds=X.std(axis=0,ddof=1) #(1024,)
Xnorm= (X-means)/stds #(5000,1024)

loadimages(Xnorm,10,10)

def getsvd(X):
    sigma=(X.T@X)/len(X) #(1024,1024)
    U,S,V=np.linalg.svd(sigma)
    return U,S,V

#共有1024个主成分，每个主成分都是一个长度为1024的向量
U,S,V=getsvd(Xnorm)
print(U.shape,S.shape,V.shape)
#(1024, 1024) (1024,) (1024, 1024)

#可视化主成分(每个主成分可以reshape为32*32的像素值矩阵，我们在这只取前36个主成分画出来看看)
#⚠️ 【每一列】代表一个主成分！
images=U[:,:36]  #前36个主成分 (1024,36)  
fig,ax_array=plt.subplots(6,6,sharex=True,sharey=True)
    
for row in range(6):
    for col in range(6):
        ax_array[row][col].matshow(images[:,6*row+col].reshape(32,32).T,cmap='gray')
plt.xticks(())
plt.yticks(())
plt.show()

#从U中选出K个最重要的分量，即前K个特征向量作为Uredeuce
def projectData(X,U,K):
    Ureduce=U[:,:K]
    Z=X@Ureduce
    return Z

Z=projectData(Xnorm,U,36) #(5000,36)


#reconstructed from only the top 36 principal components.
def recoverData(Z,U,K):
    Ureduce=U[:,:K]
    Xrec=Z@Ureduce.T
    return Xrec

Xrec=recoverData(Z,U,36) #(5000,1024)

loadimages(Xrec,10,10)