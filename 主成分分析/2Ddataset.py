#coding:UTF-8

from scipy import io 
import numpy as np
import matplotlib.pyplot as plt
from scipy.io import loadmat

data=loadmat('/Users/zhangying/Desktop/243-ML-Ng/ex7-kmeans and PCA/data/ex7data1.mat')
X=data['X'] #(50,2)


means=X.mean(axis=0)
# print (means) #(2,) #[3.98926528 5.00280585]
stds=X.std(axis=0,ddof=1) 
# print (stds)  #(2,) #[1.17304991 1.02340778]
Xnorm= (X-means)/stds
# print (Xnorm)

plt.scatter(Xnorm[:,0],Xnorm[:,1],facecolors='none', edgecolors='blue',label='Processed Dataset')


def getsvd(X):
    sigma=(X.T@X)/len(X) #(2,2)
    U,S,V=np.linalg.svd(sigma)
    return U,S,V


U,S,V=getsvd(Xnorm)
print ('第一个主成分U1是{}'.format(U[:,0]))
# print (U) #(2, 2)
# [[-0.70710678 -0.70710678]
#  [-0.70710678  0.70710678]]
# print (S) #(2,) 
# [1.70081977 0.25918023]
# print (V) #(2, 2)
# [[-0.70710678 -0.70710678]
#  [-0.70710678  0.70710678]]


# plt.plot([means[0],means[0]+1.5*S[0]*U[0,0]],[means[1],means[1]+1.5*S[0]*U[0,1]],c='r',label='the 1st pc')
# plt.plot([means[0],means[0]+1.5*S[1]*U[1,0]],[means[1],means[1]+1.5*S[1]*U[1,1]],c='g',label='the 2nd pc')



def projectData(X,U,K):
    Ureduce=U[:,:K]
    Z=X@Ureduce
    return Z

Z=projectData(Xnorm,U,1) #(50,1)

def recoverData(Z,U,K):
    Ureduce=U[:,:K]
    Xrec=Z@Ureduce.T
    return Xrec

Xrec=recoverData(Z,U,1) #(50,2)

plt.scatter(Xrec[:,0],Xrec[:,1],facecolors='None',edgecolors='red',label='Recoverd Dataset')

for i in range(len(Xnorm)):
    plt.plot([Xnorm[i,0],Xrec[i,0]],[Xnorm[i,1],Xrec[i,1]],'k--') 

plt.axis('equal')

plt.legend()
plt.show()