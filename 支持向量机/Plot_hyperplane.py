import numpy as np
import matplotlib.pyplot as plt
from sklearn import svm

np.random.seed(0)

X = np.array([[3,3],[4,3],[1,1]])
Y = np.array([1,1,-1])

plt.scatter(X[:, 0], X[:, 1], c=Y, cmap=plt.cm.rainbow)

clf = svm.SVC(kernel='linear')
clf.fit(X, Y)

print (clf.coef_) #[[0.5 0.5]]
print (clf.intercept_) #[-2.]
w=clf.coef_[0] #[0.5 0.5]
b=clf.intercept_[0] #-2.

#画出HYPERPLANE
x1=np.linspace(-5,5)
x2=(-w[0]/w[1])*x1-b/w[1]
plt.plot(x1,x2,'k') 


#print (clf.support_vectors_) #[[1. 1.] [3. 3.]]

#画出下间隔
downsv=clf.support_vectors_[0] #[1,1]
x2_down=(-w[0]/w[1])*x1 + (w[0]*downsv[0]+w[1]*downsv[1])/w[1]
plt.plot(x1,x2_down,'b')

#画出上间隔
upsv=clf.support_vectors_[1] #[3,3]
x2_up=(-w[0]/w[1])*x1 + (w[0]*upsv[0]+w[1]*upsv[1])/w[1]
plt.plot(x1,x2_up,'g')

# 圈出支持向量
plt.scatter(clf.support_vectors_[:, 0], clf.support_vectors_[:, 1],facecolors='none',edgecolors='k')

plt.show()

#输出各数据点与超平面的距离（带正负）
print(clf.decision_function(X)) 
#[ 1.   1.5 -1. ]