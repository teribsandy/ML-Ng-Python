#coding:utf-8

import numpy as np
from scipy import io
from scipy.optimize import minimize

def getmovies():
    movies={}
    f=open('/Users/zhangying/Desktop/243-ML-Ng/ex8-anomaly detection and recommendation/data/movie_ids.txt','r',encoding='gbk')
    for line in f.readlines():
        words=line.strip().split(' ')
        index=int(words[0])-1
        name=' '.join(words[1:])
        movies[index]=name
    return movies



def getMyRatings(mymovies,myratings):

    myindexs=[] #[0, 6, 11, 53, 63, 65, 68, 97, 182, 225, 354]

    for each in mymovies:
        listindex=list(Movies.values()).index(each)
        dictindex=list(Movies.keys())[listindex] 
        #由于movies这个字典的key当时储存的就是0-1681这些数，所以dictindex和listindex的结果其实是一样的，都是一些电影名对应的索引。
        myindexs.append(dictindex)

    MyRatings=np.zeros((1682,1))

    i=0
    for each in myindexs:
        MyRatings[each]=myratings[i]

        #打印出来看看Movies和MyRatings是不是都对上了
        # print ('Rated {0} for {1}'.format( int(MyRatings[each]), Movies[each] ) ) 

        i+=1

    return MyRatings



def Normalize(Y):
    nm,nu=Y.shape[0],Y.shape[1]
    Mmean=np.zeros(nm)
    Ynorm=np.zeros((nm,nu))
    for i in range(nm):
        Mmean[i]=sum(Y[i])/sum(Y[i]!=0)
        userindex=np.where(Y[i]!=0)[0]
        Ynorm[i,userindex]=Y[i,userindex]-Mmean[i]
    return Mmean,Ynorm



def params_vector(X,Theta):
    return np.r_[X.ravel(),Theta.ravel()]



def params_seperate(XTvector,nm,nu,nf):
    X=XTvector[:nm*nf].reshape(nm,nf)
    Theta=XTvector[nm*nf:].reshape(nu,nf)
    return X,Theta


#计算成本和梯度
def Cost(XTvector,Y,R,nf,l=0):
    nm=Y.shape[0]
    nu=Y.shape[1]
    X,Theta=params_seperate(XTvector,nm,nu,nf)

    H=(X@Theta.T)*R
    cost=0.5*np.sum((H-Y)**2)
    reg=(l/2.)*(np.sum(X**2)+np.sum(Theta**2))

    return cost+reg


def Gradient(XTvector,Y,R,nf,l=0):
    nm=Y.shape[0]
    nu=Y.shape[1]
    X,Theta=params_seperate(XTvector,nm,nu,nf)

    error=(X@Theta.T)*R -Y     #(1682,943)
    Xgrad=error@Theta+l*X      #(1682,10)
    Thetagrad=error.T@X+l*Theta   #(943,10)

    gradvector=np.r_[Xgrad.ravel(),Thetagrad.ravel()]
    return gradvector      #(26250,)




#⚠️加载txt文本获取电影
Movies=getmovies() 


#⚠️根据pdf中给出的新用户对部分电影的评分，创建该用户的评分向量
mymovies=['Toy Story (1995)','Twelve Monkeys (1995)','Usual Suspects, The (1995)','Outbreak (1995)','Shawshank Redemption, The (1994)','While You Were Sleeping (1995)','Forrest Gump (1994)','Silence of the Lambs, The (1991)','Alien (1979)','Die Hard 2 (1990)','Sphere (1998)']

myratings=[4,3,5,4,5,3,5,2,4,5,5]

MyRatings=getMyRatings(mymovies,myratings)


#⚠️准备数据(加入新用户)
data=io.loadmat('/Users/zhangying/Desktop/243-ML-Ng/ex8-anomaly detection and recommendation/data/ex8_movies.mat')
Y,R=data['Y'],data['R'] #(1682,943) (1682,943)

Ynew=np.c_[Y,MyRatings]      #(1682,944)
Rnew=np.c_[R,MyRatings!=0]   #(1682,944)

MoviesMean,Ynorm=Normalize(Ynew) #(1682,) (1682,944)

nm=Ynew.shape[0]
nu=Ynew.shape[1]
nf=10
l=10

X=np.random.random(size=(nm, nf))        #(1682,10)
Theta=np.random.random(size=(nu, nf))    #(944,10)
XTvector=params_vector(X,Theta)          #(26260)

#⚠️高级优化算法获得最佳参数
res=minimize(fun=Cost,x0=XTvector,args=(Ynorm,Rnew,nf,l),method='TNC',jac=Gradient)

#      fun: 38951.847559994734
#      jac: array([ 9.76108171e-06, -2.17310042e-05,  6.00794443e-06, ...,
#        -1.03016156e-06, -8.27640298e-07, -2.06891274e-07])
#  message: 'Converged (|f_n-f_(n-1)| ~= 0)'
#     nfev: 497
#      nit: 38
#   status: 1
#  success: True
#        x: array([ 0.23379786,  0.29751594,  0.93747413, ...,  0.08347551,
#        -0.03076913, -0.10729576])

finalX, finalTheta = params_seperate(res.x,nm,nu,nf)

#⚠️根据最佳参数进行预测
Predictions=finalX@finalTheta.T+MoviesMean.reshape(1682,1)       #(1682,944)      
myPred=Predictions[:,-1]    #(1682,)

# Predictions=finalX@finalTheta.T         #(1682,944)      
# myPred=Predictions[:,-1]+MoviesMean     #(1682,)


#⚠️获得前10得分与对应电影索引，并打印出来
rates=np.sort(myPred)
rates_des=rates[::-1]
rates_top10=rates_des[:10]

idxs=np.argsort(myPred)
idxs_des=idxs[::-1]
idxs_top10=idxs_des[:10]

for i in range(10):
    j=idxs_top10[i]
    print ('电影{0}的预测评分为{1}'.format(Movies[j],rates_top10[i]) )

# 电影Someone Else's America (1995)的预测评分为5.000000038816171
# 电影Santa with Muscles (1996)的预测评分为5.00000002415254
# 电影Star Kid (1997)的预测评分为5.000000017763759
# 电影Prefontaine (1997)的预测评分为5.000000013131249
# 电影Saint of Fort Washington, The (1993)的预测评分为5.000000008669887
# 电影Aiqing wansui (1994)的预测评分为5.0000000000943245
# 电影Marlene Dietrich: Shadow and Light (1996)的预测评分为5.000000000003238
# 电影Entertaining Angels: The Dorothy Day Story (1996)的预测评分为4.999999999994927
# 电影They Made Me a Criminal (1939)的预测评分为4.999999994716483
# 电影Great Day in Harlem, A (1994)的预测评分为4.999999984863214