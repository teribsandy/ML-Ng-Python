#coding:UTF-8

from scipy import io 
import numpy as np
import matplotlib.pyplot as plt
import skimage 


#bird-small.png是一张像素为128*128的图片，两个128分别代表行与列的position，3代表RGB通道
#Eg. A(50, 33, 3) gives the blue intensity of the pixel at row 50 and column 33


# data=io.loadmat('/Users/zhangying/Desktop/243-ML-Ng/ex7-kmeans and PCA/data/bird_small.mat')
# A=data['A'] #(128,128,3) 

A=skimage.io.imread('/Users/zhangying/Desktop/ying/K-means/1.jpeg')/255.

plt.imshow(A)
plt.show()

def initCentroids(X,K):
    m=X.shape[0]
    index=np.random.choice(m,K)
    centroids=X[index]
    return centroids


def findClosestCentroids(X,centroids):
    idx=[]
    max_dist=1000000000 
    for i in range(len(X)):
        minus=X[i]-centroids #here use numpy's broadcasting
        dist=np.sum(np.power(minus,2),axis=1) 
        if dist.min()<max_dist:
            ci=np.argmin(dist)
            idx.append(ci)
    return np.array(idx)


def computeCentroids(X,idx):
    centroids=[]
    for i in range(16): 
        u_k=X[idx==i].mean(axis=0)
        centroids.append(u_k)
    return np.array(centroids)


def runKmeans(X,init_centroids,max_iters):
    Rcentroids=[]
    centroids=init_centroids

    for _ in range(max_iters+1):

        Rcentroids.append(centroids)

        idx=findClosestCentroids(X,centroids)
        centroids=computeCentroids(X,idx)
    
    return idx,Rcentroids



IV=A.reshape(A.shape[0]*A.shape[1],A.shape[2]) #(16384,3)

# print (len(set(tuple(each) for each in IV))) #13930 
#each的shape为(3,) 每个像素被表示为三个8位无符号整数(从0到255)，指定了红、绿和蓝色的强度值。这种编码通常被称为RGB编码。
#有13930个像素位置R、G、B通道的intensity不完全相同，说明有13930种颜色


'''
我们的目标是把颜色减少至16种！
具体地说，只需要存储16个选中颜色的RGB值，而对于图中的每个像素，现在只需要将该颜色的索引存储在该位置(只需要4 bits就能表示16种可能性)。
把原始图片的每个像素看作一个数据样本，然后利用K-means算法去找分组最好的16种颜色
Once you have computed the cluster centroids, you will then use the 16 colors to replace the pixels.
'''

init_centroids=initCentroids(IV,16)
print (init_centroids) #(16,3)

idx,Rcentroids=runKmeans(IV,init_centroids,10)
print (Rcentroids)

centroids=Rcentroids[-1]
print (centroids)

# image=np.zeros(IV.shape,dtype=int) #(16384,3) 
image=np.zeros(IV.shape) 
#np.zeros默认数据类型为浮点数，但imshow要求像素值必须为0-255内的整数或者是0-1内的浮点数。
#由于我没有对A进行A/255的归一化处理，所以我的像素值需要为0-255内的整数，所以我必须把image=np.zeros(IV.shape)的 dtype设置为整数类型。另外findClosestCentroids函数中的max_dist最好设大一点，以免有的样本被拦在条件外从而ci少记录了

#当然啦！最简便的方法还是一开始就令A=A/255. 那么就不需要操心这些东西啦！(注意255那有个. 浮点数！不然很容易出现nan)
#除以255的目的是做一个数据的归一化，把所有的像素值限定到(0，1)，这对计算距离的模型来说(比如K-means)是很重要的，不然会被离群值影响较大。 

for i in range(16):
    image[idx==i]=centroids[i]

image=image.reshape(A.shape)
print (image)

plt.imshow(image)
plt.show()