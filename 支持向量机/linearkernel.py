#coding:UTF-8
import numpy as np
from scipy import io
from sklearn.svm import SVC
import matplotlib.pyplot as plt
from matplotlib.font_manager import FontProperties
plt.rcParams['font.family'] = ['Arial Unicode MS'] #用来正常显示中文标签 (有用的！！！亲测)
# plt.rcParams['axes.unicode_minus'] = False #用来正常显示负号 （没感觉出有啥用，没有该命令打负号也没啥问题啊～）


path='/Users/zhangying/Desktop/243-ML-Ng/ex6-SVM/data/ex6data1.mat'
data=io.loadmat(path)
X,y=data['X'],data['y']  #(51, 2) <class 'numpy.ndarray'>   #(51, 1) <class 'numpy.ndarray'>


'''画出正负样本的散点图'''
# yposindex=np.where(y==True)
# #np.where() 返回输入数组中满足给定条件的元素的索引,即y为1的元素的索引，以二元数组存放，数组中第一个元素是行索引的ndarray，第一个元素是列索引的ndarray。
# Xposrowindex=yposindex[0]
# Xpos=np.array( [X[each,:] for each in Xposrowindex] )

# ynegindex=np.where(y==False)
# Xnegrowindex=ynegindex[0]
# Xneg=np.array( [X[each,:] for each in Xnegrowindex] )

# plt.scatter(Xpos[:,0],Xpos[:,1],s=15,c='k',marker='+')
# plt.scatter(Xneg[:,0],Xneg[:,1],s=15,c='y',marker='o')

#👆的步骤其实可以用下面这一条语句就做到了！！！！
plt.scatter(X[:,0],X[:,1],c=y.flatten(),cmap='rainbow')


'''用线性核函数SVM分类，得到权重和偏置，并给出假设函数'''
clf=SVC(C=1,kernel='linear')
clf.fit(X,y.ravel())


w=clf.coef_ #[[1.40718563 2.13398052]]  #(1,2)
b=clf.intercept_ #[-10.34889778]        #(1,)

def h(x):
    return np.array(w)@x.T+np.array(b)


'''绘制决策边界 #https://blog.csdn.net/Cowry5/article/details/80261260 '''
#实施笨方法，根据w1x1+w2x2+b=0，写出x2关于x1的等式
# w1=w[0][0]
# w2=w[0][1]
# b=b[0] 
# def get_x2(x1):
#     return (-w1*x1-b)/w2
# x1=np.arange(0,4.5,0.1)
# plt.plot(x1,get_x2(x1),linewidth='1.5',color='red')


#利用等高线绘制高纬决策边界
x1=np.linspace(0,4.5,500)
x2=np.linspace(1.5,5,500)
xx1,xx2=np.meshgrid(x1,x2)
x=np.c_[xx1.ravel(),xx2.ravel()] 
z=h(x) 


#利用等高线绘制高纬决策边界,其实可以不用特意自己写函数算出hypothesis的,用拟合后的clf预测x就好了
#区别在于z=h(x)所得的决策边界是平滑的直线；而用clf.predict(x)所得是一个个点只是因为点的数量多，看上去像是一条拟合直线，实际上是一条条小线段
# z=clf.predict(x)


z=z.reshape(xx1.shape)
C=plt.contour(xx1,xx2,z,0,colors='black')
plt.clabel(C,inline=True,fontsize=10)
plt.xlabel('特征1:x1')
plt.ylabel('特征2:x2')


plt.show()