#coding:UTF-8
import numpy as np
from scipy import io
from sklearn.svm import SVC
import matplotlib.pyplot as plt
from matplotlib.font_manager import FontProperties
plt.rcParams['font.family'] = ['Arial Unicode MS'] #用来正常显示中文标签 (有用的！！！亲测)


path='/Users/zhangying/Desktop/243-ML-Ng/ex6-SVM/data/ex6data2.mat'
data=io.loadmat(path)
#print (data.keys()) #dict_keys(['__header__', '__version__', '__globals__', 'X', 'y'])
X,y=data['X'],data['y']  #(863, 2) <class 'numpy.ndarray'>   #(863, 1) <class 'numpy.ndarray'>


'''画出正负样本的散点图'''
plt.scatter(X[:,0],X[:,1],c=y.flatten(),cmap='rainbow')


'''自定义求高斯核函数'''
def similarity(sample,landmark,gamma): #(1,2)
    return np.exp( (-gamma)*(sample-landmark)@(sample-landmark).T ) #gamma=1/(2*sigma**2)

#用吴恩达给的例子检测一下,x1,x2如下，sigma=2
# x1=np.array([[1,2,1]])
# x2=np.array([[0,4,-1]])
# print (similarity(x1,x2,1/8)) #[[0.32465247]]

def f_asmatrix(X):
    m=len(X)
    f=np.zeros((m,m)) #(863,863)
    for i in range(m): #0,1,...,862
        for j in range (m):
            f[i][j]=similarity(X[i,:],X[j,:],1/2)  #gamma=1/n_features #gamma=1/(n_features*X.std())

#print (f_asmatrix(X)) #   为啥得出来None呢 ???? 



'''用线性核函数SVM分类，得到权重和偏置，并给出假设函数'''
sigma=0.1
gamma=np.power(sigma,-2.)/2
clf=SVC(C=1,kernel='rbf',gamma=gamma)
clf.fit(X,y.ravel())


'''绘制决策边界 #https://blog.csdn.net/Cowry5/article/details/80261260 '''
x1=np.linspace(X[:,0].min()*1.1,X[:,0].max()*1.1,500)
x2=np.linspace(X[:,1].min()*1.1,X[:,1].max()*1.1,500)
xx1,xx2=np.meshgrid(x1,x2)
x=np.c_[xx1.ravel(),xx2.ravel()] 
z=clf.predict(x)
z=z.reshape(xx1.shape)
plt.contour(xx1,xx2,z)
plt.xlabel('特征1:x1')
plt.ylabel('特征2:x2')


plt.show()