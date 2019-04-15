'''梯度下降算法好像不是很适合用在逻辑回归的求解里
无论X，y，theta是采用matrix还是ndarray的形式都出现下面的现象，在本文件中，X，y，theta采用matrix
为啥会出现下面这些情况呢？？？？？不懂啊
现象1:执行a=y*np.log(hypothesis) + (1-y)*np.log(1-hypothesis)时报RuntimeWarning
错误包括divide by zero encountered in log, invalid value encountered in multiply
现象2:allcost里迭代记录每3次出现一次nan（not a number）
现象3:所得解与高级优化算法相差太多，最终画决策边界图的时候发现所求得的决策边界完全没有起到分类的效果
'''

# coding: utf-8
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

data=pd.read_csv('/Users/zhangying/Desktop/243-ML-Ng/ex2-logistic regression/ex2data1.txt',header=None,names=['Exam1','Exam2','Class'])

positive=data[ data.Class.isin(['1']) ]
negative=data[ data.Class.isin(['0']) ]

def normalize_feature(df):
    return df.apply(lambda column : (column-column.mean())/column.std())
data=normalize_feature(data)    
#或者采用👇也可以特征归一化
#data=(data-data.mean())/data.std() #特征归一化

data.insert(0,'Ones',1)
X=data.iloc[:,0:3]
y=data.iloc[:,3:4] 

X=np.matrix(X.values) # (100,3) 
y=np.matrix(y.values) #  (100,1)  
theta=np.matrix(np.zeros(3))   # (1,3)

# print (X.shape, y.shape, theta.shape)

def sigmoid(theta,X):
    z=X*theta.T 
    return 1/(1+np.exp(-z))

originalhypothesis=sigmoid(theta,X) #matrix，shape为（100,1）
# print (type(originalhypothesis),originalhypothesis.shape)

def cost(theta,X,y):
    hypothesis=sigmoid(theta,X)
    a=np.multiply(y,np.log(hypothesis)) + np.multiply((1-y),np.log(1-hypothesis)) 
    return (-1./len(X))*np.sum(a)  

originalcost= cost(theta,X,y) #0.69314718056
print (originalcost)


def gradientdescent(X,y,theta,alpha,epochs):
    temp=np.matrix(np.ones(3)) #(1,3)
    allcost=np.zeros(epochs)
    for i in range(epochs):
        error=sigmoid(theta,X)-y #matrix (100,1)  
        temp=theta-(alpha/len(X))* error.T*X
        theta=temp
        allcost[i]=cost(theta,X,y) 
    return theta,allcost

alpha=0.1
epochs=200000
finaltheta,allcost=gradientdescent(X,y,theta,alpha,epochs)
#finaltheta [[-0.76754116  0.35868899 -0.11012487]]
#allcost[-1]为4.76424858314 等价于print cost(X,y,finaltheta)
print (finaltheta)
print (allcost)

'''

def predict(theta,X):
    probability=sigmoid(theta,X) 
    return [1 if x>=0.5 else 0 for x in probability]

predictions=predict(finaltheta,X)
correct=[1 if a==b else 0 for (a,b) in zip(predictions,y)]
accuracy=sum(correct)/len(X)
#print (accuracy) #0.6

#也可以用skearn中的方法来检验预测精度，完全不知道为啥它测出来的精度总是奇高无比，无语了
# from sklearn.metrics import classification_report
# print (classification_report(predictions,y)) 


x1=np.arange(130,step=0.1)
x2=-(finaltheta[0,0]+finaltheta[0,1]*x1)/finaltheta[0,2]


fig,ax=plt.subplots(figsize=(8,5))
ax.scatter(positive.Exam1,positive['Exam2'],color='black',marker='+',label='Admitted')
ax.scatter(negative['Exam1'],negative.Exam2,c='y',marker='o',label='Not admitted')
ax.plot(x1,x2)
ax.set_xlim(20,110)
ax.set_ylim(20,110)
ax.set_xlabel('Exam1 Score')
ax.set_ylabel('Exam2 Score')
ax.set_title('Decision Boundary')
ax.legend(loc=1) #图例显示在右上角
plt.show()
'''