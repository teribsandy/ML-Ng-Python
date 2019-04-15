# coding: utf-8

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import re
import nltk, nltk.stem.porter

'''读取txt文件'''
path1='/Users/zhangying/Desktop/243-ML-Ng/ex6-SVM/data/emailSample1.txt'

with open(path1,'r') as f:
    email=f.read()



'''预处理email
# 1. Lower-casing: 把整封邮件转化为小写。
# 2. Stripping HTML: 移除所有HTML标签，只保留内容。
# 3. Normalizing URLs: 将所有的URL替换为字符串 “httpaddr”.
# 4. Normalizing Email Addresses: 所有的地址替换为 “emailaddr”
# 5. Normalizing Dollars: 所有dollar符号($)替换为“dollar”.
# 6. Normalizing Numbers: 所有数字替换为“number”
# 7. Word Stemming(词干提取): 将所有单词还原为词源。例如，“discount”, “discounts”, “discounted” and “discounting”都替换为“discount”。
# 8. Removal of non-words: 移除所有非文字类型，所有的空格(tabs, newlines, spaces)调整为一个空格.
并将email这个纯文本 切割为 单词 所组成的列表
'''

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
    # print (tokens)
    #['', 'anyone', 'knows', 'how', 'much', 'it', 'costs', 'to', 'host', 'a', 'web', 'portal', '', '', '', 'well', '', 'it', 'depends', 'on', 'how', 'many', 'visitors', 'you', 're', 'expecting', '', 'this', 'can', 'be', 'anywhere', 'from', 'less', 'than', 'number', 'bucks', 'a', 'month', 'to', 'a', 'couple', 'of', 'dollarnumber', '', '', 'you', 'should', 'checkout', 'httpaddr', 'or', 'perhaps', 'amazon', 'ecnumber', '', 'if', 'youre', 'running', 'something', 'big', '', '', '', 'to', 'unsubscribe', 'yourself', 'from', 'this', 'mailing', 'list', '', 'send', 'an', 'email', 'to', '', 'emailaddr', '', '']
    
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



'''vocab中的单词是实际中常使用的单词，我们以它构建我们的特征向量，特征数=len(vocab)
对一封email来说，它的tokenlist中那些也出现在vocab.txt中的单词代表了该样本的特征向量中这些特征取值为1
Specifically, the feature xi ∈ {0, 1} for an email corresponds to whether the i-th word in the dictionary occurs in the email. 
That is, xi = 1 if the i-th word is in the email and xi = 0 if the i-th word is not present in the email.
因此，我们需要得到返回这些单词在vocab中的相应的index，从而为特征向量中的特征赋值'''

def email2VocabIndices(email,vocab):
    tokenlist=email2Tokenlist(email)
    # print (tokenlist)
    #['anyon', 'know', 'how', 'much', 'it', 'cost', 'to', 'host', 'a', 'web', 'portal', 'well', 'it', 'depend', 'on', 'how', 'mani', 'visitor', 'you', 're', 'expect', 'thi', 'can', 'be', 'anywher', 'from', 'less', 'than', 'number', 'buck', 'a', 'month', 'to', 'a', 'coupl', 'of', 'dollarnumb', 'you', 'should', 'checkout', 'httpaddr', 'or', 'perhap', 'amazon', 'ecnumb', 'if', 'your', 'run', 'someth', 'big', 'to', 'unsubscrib', 'yourself', 'from', 'thi', 'mail', 'list', 'send', 'an', 'email', 'to', 'emailaddr']

    vocab_indice=[]

    for i in range(len(vocab)): #i=0,1,...1898 

        #对tokenlist中的单词进行筛选，在实际中不常使用的单词我们不把它作为特征
        if vocab[i] in tokenlist: 
            vocab_indice.append(i)

    return vocab_indice   #返回列表，该列表包含了【这些特征在特征向量中对应的索引】



'''为每封邮件形成一个长度为len(vocab)、初始元素均为0的特征向量，
再根据vocab_indice为特征向量中的一些特征赋值为1
（xi = 1 if the i-th word is in the email）'''

def email2FeatureVector(email):
    # read_csv() 读取以‘，’分割的文件到DataFrame   read_table()读取以‘/t’分割的文件到DataFrame
    # 实质上是通用的，在实际使用中可以通过对sep或delimiter参数的控制来对任何文本文件读取。

    path2='/Users/zhangying/Desktop/243-ML-Ng/ex6-SVM/data/vocab.txt'
    df=pd.read_table(path2,sep=r'\s+',header=None,names=['from one','word'],usecols=['word'])
    vocab=df.values.ravel()            #(1899,)

    vocab_indice=email2VocabIndices(email,vocab)
    # print (vocab_indice)
    #[70, 85, 88, 161, 180, 237, 369, 374, 430, 478, 529, 530, 591, 687, 789, 793, 798, 809, 882, 915, 944, 960, 991, 1001, 1061, 1076, 1119, 1161, 1170, 1181, 1236, 1363, 1439, 1476, 1509, 1546, 1662, 1675, 1698, 1757, 1821, 1830, 1892, 1894, 1895]

    featurevector=np.zeros(len(vocab)) #(1899,)

    for i in vocab_indice:
        featurevector[i]=1
    
    return featurevector

featurevector=email2FeatureVector(email)

print('length of featurevector = {}\nnum of non-zero = {}'.format(len(featurevector), int(featurevector.sum())))
#length of featurevector = 1899
#num of non-zero = 45

    

    








