#coding:UTF-8
import numpy as np
from scipy import io
from sklearn.svm import SVC
import matplotlib.pyplot as plt
from matplotlib.font_manager import FontProperties
plt.rcParams['font.family'] = ['Arial Unicode MS'] #用来正常显示中文标签 (有用的！！！亲测)


path='/Users/zhangying/Desktop/243-ML-Ng/ex6-SVM/data/ex6data3.mat'
data=io.loadmat(path)
#print (data.keys())
#dict_keys(['yval', '__header__', 'X', 'y', '__globals__', '__version__', 'Xval']) 

X,y,Xval,yval=data['X'],data['y'],data['Xval'],data['yval']  
#print (X.shape,y.shape,Xval.shape,yval.shape)
#(211, 2) (211, 1) (200, 2) (200, 1)


'''画出正负样本的散点图'''
#plt.scatter(X[:,0],X[:,1],c=y.flatten(),cmap='rainbow') #缺点在于我并不知道哪些是正样本，哪些是负样本

# import pandas as pd
# pddata = pd.DataFrame(X, columns=['X1', 'X2'])
# pddata['y'] = y

# positive = pddata[ pddata['y'].isin([1]) ]
# negative = pddata[ pddata['y'].isin([0]) ]

# fig, ax = plt.subplots(figsize=(8,5))
# ax.scatter(positive['X1'], positive['X2'], s=50, marker='x', label='Positive')
# ax.scatter(negative['X1'], negative['X2'], s=50, marker='o', label='Negative')
# ax.legend()


'''确定C*与gamma*'''
C=[0.01,0.03,0.1,0.3,1,3,10,30] 
sigma=[0.01,0.03,0.1,0.3,1,3,10,30]
gamma=np.power(sigma,-2.)/2 #(10,)

#用高斯核函数训练多个模型，并通过cv集
# bestscore=0
# bestpair=(0,0)
# for c in C:
#     for gm in gamma:
#         model=SVC(C=c,kernel='rbf',gamma=gm)
#         model.fit(X,y.ravel())
#         score=model.score(Xval,yval)
#         if score>bestscore:
#             bestscore=score
#             bestpair=(c,gm)
# print ('bestpair={},bestscore={}'.format(bestpair,bestscore))
#bestpair=(1, 49.99999999999999),bestscore=0.965

#用sklearn提供的网格搜索法GridSearchCV（网络搜索交叉验证）
#用于系统地遍历模型的多种参数组合，通过交叉验证从而确定最佳参数，适用于小数据集。
#常用属性包括best_score_  best_params_  cv_results_  best_estimator_

from sklearn.model_selection  import GridSearchCV
parameters = {'C': C, 'gamma': list(gamma)}
newmodel=SVC()
clf = GridSearchCV(estimator=newmodel, param_grid=parameters,scoring='accuracy',cv=3)
clf.fit(X,y.ravel())
print (clf.best_params_) 
print (clf.best_score_)
# {'C': 3, 'gamma': 49.99999999999999}
# 0.8815165876777251


# '''确定C和gamma后用训练集拟合模型'''
# clf=SVC(C=bestpair[0],kernel='rbf',gamma=bestpair[1])
# clf.fit(X,y.ravel())


# '''绘制决策边界 #https://blog.csdn.net/Cowry5/article/details/80261260 '''
# x1=np.linspace(X[:,0].min()*1.1,X[:,0].max()*1.1,500)
# x2=np.linspace(X[:,1].min()*1.1,X[:,1].max()*1.1,500)
# xx1,xx2=np.meshgrid(x1,x2)
# x=np.c_[xx1.ravel(),xx2.ravel()] 
# z=clf.predict(x)
# z=z.reshape(xx1.shape)
# plt.contour(xx1,xx2,z)
# plt.axis('tight')
# plt.xlabel('特征1:x1')
# plt.ylabel('特征2:x2')


# plt.show()