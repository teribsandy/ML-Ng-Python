# coding: utf-8

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import re
import nltk, nltk.stem.porter


'''预处理email'''

def processEmail(email): #实现1-6
    email=email.lower()
    email=re.sub('>','',email)
    email=re.sub(r'(http|https)://[^\s]*','httpaddr',email)
    email=re.sub(r'[^\s]+@[^\s]+','emailaddr',email)
    email=re.sub('[$]+','dollar',email)
    email=re.sub('[0-9]+','number',email)
    return email


def email2Tokenlist(email): #实现7&8
    email=processEmail(email)
    stemmer=nltk.stem.porter.PorterStemmer()
    
    #把邮件分割为单词,不保留分隔符 #如果使用括号捕获分组，默认保留分割符  re.split(r"([,!.?:'\s])",email)  
    tokens =re.split(r"[,!.?:'\s+]", email) 
    
    tokenlist=[]
    for token in tokens:
        #删除任何非字母数字的字符
        token=re.sub('[^a-zA-Z0-9]','',token)  

        #跳过空字符
        if len(token)==0:
            continue

        #使用Porter stemmer提取词根
        stemmed=stemmer.stem(token)
        tokenlist.append(stemmed) 

    return tokenlist



'''对一封email来说，返回这些单词在vocab中的相应的index，从而为特征向量中的特征赋值'''

def email2VocabIndices(email,vocab):
    tokenlist=email2Tokenlist(email)
 
    vocab_indice=[]

    for i in range(len(vocab)): #i=0,1,...1898 
        if vocab[i] in tokenlist: 
            vocab_indice.append(i)

    return vocab_indice   #返回列表，该列表包含了【这些特征在特征向量中对应的索引】


'''为每封邮件形成一个特征向量，根据vocab_indice为特征向量中的一些特征赋值为1'''

def email2FeatureVector(email):
    
    path='/Users/zhangying/Desktop/243-ML-Ng/ex6-SVM/data/vocab.txt'
    df=pd.read_table(path,sep=r'\s+',header=None,names=['from one','word'],usecols=['word'])
    vocab=df.values.ravel()            #(1899,)

    vocab_indice=email2VocabIndices(email,vocab)

    featurevector=np.zeros(len(vocab)) #(1899,)

    for i in vocab_indice:
        featurevector[i]=1
    
    return featurevector




'''读取txt文件'''
paths=['/Users/zhangying/Desktop/243-ML-Ng/ex6-SVM/data/emailSample1.txt',
    '/Users/zhangying/Desktop/243-ML-Ng/ex6-SVM/data/emailSample2.txt',
    '/Users/zhangying/Desktop/243-ML-Ng/ex6-SVM/data/spamSample1.txt',
    '/Users/zhangying/Desktop/243-ML-Ng/ex6-SVM/data/spamSample2.txt']

emails=[]
for eachpath in paths:
    with open(eachpath,'r') as f:
        email=f.read()
    emails.append(email)

'''读取txt文件'''

featurevectors=[]

for i in range(len(paths)): #0,1,2,3
    email=emails[i]
    fv=email2FeatureVector(email)
    print('length of fv= {}\nnum of non-zero = {}'.format(len(fv), int(fv.sum())))
    #1899,45  1899,124  1899,46  1899,18
    featurevectors.append(fv)

X=np.array(featurevectors) #(4,1899)
y=np.array([0,0,1,1])

from sklearn import svm

clf=svm.SVC(C=1,kernel='linear') 
clf.fit(X,y)   
score=clf.score(X,y) 

print(score) # 1.0









