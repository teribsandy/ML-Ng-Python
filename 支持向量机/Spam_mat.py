# coding: utf-8
import numpy as np
from scipy import io

trainset=io.loadmat('/Users/zhangying/Desktop/243-ML-Ng/ex6-SVM/data/spamTrain.mat')
# print (trainset.keys()) #dict_keys(['__version__', 'X', 'y', '__globals__', '__header__'])

X_train,y_train=trainset['X'],trainset['y'].ravel()
# print(X_train.shape,y_train.shape) #(4000, 1899) (4000, )

testset=io.loadmat('/Users/zhangying/Desktop/243-ML-Ng/ex6-SVM/data/spamTest.mat')

X_test,y_test=testset['Xtest'],testset['ytest'].ravel()
#(1000, 1899) (1000, )
print (y_test)

from sklearn import svm

#训练多个模型，并通过test集确定 C 值
# C=[0.01,0.03,0.1,0.3,1,3,10,30] 
# bestscore=0
# bestC=0
# for c in C:
#     model=svm.SVC(C=c,kernel='linear') #采用linear是因为样本数和特征数的大小决定的
#     model.fit(X_train,y_train)
#     score=model.score(X_test,y_test)
#     if score>bestscore:
#         bestscore=score
#         bestC=c
# print ('bestC={},bestscore={}'.format(bestC,bestscore))
#bestC=0.03,bestscore=0.99


clf=svm.SVC(C=0.1,kernel='linear') 
clf.fit(X_train,y_train)   
score_train=clf.score(X_train,y_train) 
score_test=clf.score(X_test,y_test)
# print (score_train,score_test)
#C=0.1（Ng的练习里给的）:0.99825 0.989
#C=0.03（自测得到的）:   0.99425 0.99

import pandas as pd

#获取模型拟合后得到的权重，并转换成dataframe形式
weights=clf.coef_.reshape(1899,1)
df=pd.DataFrame(weights,columns=['Weight'])

#按权重降序排列，row index会按照权重原来的值移动
sorted_df=df.sort_values(by='Weight',ascending=False)
# print (sorted_df.head())
#        Weight
# 1190  0.500614
# 297   0.465916
# 1397  0.422869
# 738   0.383622
# 1795  0.367710


#将sorted_df的行索引存放进列表
indexs=sorted_df.index.tolist()
# print (indexs)
#[1190, 297, 1397, 738, 1795, 155, 476, 1851, 1298, 1263,.....1764, 1665, 1560]

path='/Users/zhangying/Desktop/243-ML-Ng/ex6-SVM/data/vocab.txt'
df=pd.read_table(path,sep=r'\s+',header=None,names=['from one','word'],usecols=['word'])
vocab=df.values.ravel()  #ndarray (1899,)
# print (vocab)
#['aa' 'ab' 'abil' ... 'zdnet' 'zero' 'zip']


#取top 15 权重最大的词()
toppredictions=[ vocab[indexs[i]] for i in range(15)]
print (toppredictions)
#['our', 'click', 'remov', 'guarante', 'visit', 'basenumb', 'dollar', 'will', 'price', 'pleas', 'most', 'nbsp', 'lo', 'ga', 'hour']








