#coding: UTF_8

import numpy as np
import matplotlib.pyplot as plt
from scipy import io
from matplotlib.font_manager import FontProperties
plt.rcParams['font.family'] = ['Arial Unicode MS'] #用来正常显示中文标签 (有用的！！！亲测)
plt.ion()

'''获取样本'''
data=io.loadmat('/Users/zhangying/Desktop/243-ML-Ng/ex7-kmeans and PCA/data/ex7data2.mat')
X=data['X']  #(300,2)


'''可视化'''
def visualize(X,s,c,marker,fnum):
    x1=X[:,0]
    x2=X[:,1]
    _=plt.figure(fnum)
    plt.scatter(x1,x2,s=s,c=c,marker=marker,cmap='rainbow')
    plt.title('第{}次更新聚类结果图\n(第0次表示随机初始聚类结果)'.format(fnum))

# 由散点图看出K=3（对于特征高维的数据，难以直接可视化，K可以通过elbow method处确定，但大部分情况下还是从实际需求出发确定K）
# visualize(X,10,np.ones(300),'o',0)


'''随机初始化聚类中心'''
#随机初始化容易got bad luck！
def initcentroids(K,X):
    mu_k=[X[i] for i in np.random.choice(300,3)] 
    mu_k=np.array(mu_k)   #(3,2)
    return mu_k

# init_mu_k=initcentroids(3,X)



def distance(a,b):
    squa_dist=(a-b)@(a-b).T #a和b的shape为(2,)
    return squa_dist

def findclosetcentriods(X,mu_k):
    m=len(X)
    K=len(mu_k)
    c_index=np.zeros(m)
    # all_dist=np.zeros(m)
    for i in range(m):
        temp_dist=1000
        temp_j=10
        for j in range(K):
            dist=distance(X[i],mu_k[j])
            if dist<temp_dist:
                temp_dist=dist
                temp_j=j
        c_index[i]=temp_j
        # all_dist[i]=temp_dist
    return c_index#（300，）



'''初始聚类结果可视化'''
# visualize(init_mu_k,80,np.arange(3),'+',0)
# visualize(X,10,findclosetcentriods(X,init_mu_k),'o',0)



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




'''更新聚类中心并可视化'''
#每次运行都可能得到非常不同的聚类结果，迭代次数也不一样，很明显与初始化的聚类中心的位置好坏有关！
def iteration(X,now_mu_k):
    now_fnum=0
    new_mu_k=updatemuk(X,now_mu_k)
    while  not (new_mu_k==now_mu_k).all() :
        new_fnum=now_fnum+1
        visualize(new_mu_k,80,np.arange(3),'+',new_fnum) #每次更新的可视化
        visualize(X,10,findclosetcentriods(X,new_mu_k),'o',new_fnum) #每次更新的可视化
        now_fnum=new_fnum
        now_mu_k=new_mu_k
        new_mu_k=updatemuk(X,now_mu_k)
    return now_mu_k




'''计算Jcost'''
def computeJcost(X,init_mu_k):
    final_mu_k=iteration(X,init_mu_k)
    finalclass_of_samples=findclosetcentriods(X,final_mu_k) #(300,)

    samples_0=X[np.where(finalclass_of_samples==0)] #返回属于mu_0的样本 ndarray,ndim=2
    num_0=len(samples_0)
    samples_1=X[np.where(finalclass_of_samples==1)] #返回属于mu_1的样本 ndarray,ndim=2
    num_1=len(samples_1)
    samples_2=X[np.where(finalclass_of_samples==2)] #返回属于mu_2的样本 ndarray,ndim=2
    num_2=len(samples_2)

    all_dist=np.zeros(len(X)) #(300,)

    for i in range(num_0):
        dist=distance(samples_0[i],final_mu_k[0])
        all_dist[i]=dist

    for i in range(num_1):
        dist=distance(samples_1[i],final_mu_k[1])
        all_dist[i+num_0]=dist
    
    for i in range(num_2):
        dist=distance(samples_2[i],final_mu_k[2])
        all_dist[i+num_0+num_1]=dist
    
    return final_mu_k,np.mean(all_dist)





'''前面提到我们通常要进行50-1000次随机初始化，并分别进行聚类，最后选取 J 最小的 聚类过程
所以我们还需要设置一个循环'''

def loop(X,epochs=3):
    record_init_mu_k =[]
    record_final_mu_k=[]
    record_Jcost=[]

    for _ in range(epochs):
        init_mu_k=initcentroids(3,X)
        visualize(init_mu_k,80,np.arange(3),'+',0)
        visualize(X,10,findclosetcentriods(X,init_mu_k),'o',0)

        record_init_mu_k.append(init_mu_k)
        record_final_mu_k.append(computeJcost(X,init_mu_k)[0])
        record_Jcost.append(computeJcost(X,init_mu_k)[1])

        plt.ioff()
        plt.show()
    return record_init_mu_k,record_final_mu_k,record_Jcost


x,y,z=loop(X,epochs=3)
best_J=min(z)
Index=z.index(best_J)
best_init_mu_k=x[Index]
best_final_mu_k=y[Index]

print (best_J) 
#0.8888617321830645
print (best_init_mu_k)
# [[3.233947   1.08202324]
#  [3.20360621 0.7222149 ]
#  [1.67838038 5.26903822]]
print (best_final_mu_k)
# [[6.03366736 3.00052511]
#  [3.04367119 1.01541041]
#  [1.95399466 5.02557006]]

