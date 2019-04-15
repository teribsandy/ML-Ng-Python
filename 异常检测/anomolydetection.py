# coding:utf-8
import numpy as np
from scipy import io
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from matplotlib.font_manager import FontProperties
plt.rcParams['font.family'] = ['Arial Unicode MS'] #用来正常显示中文标签 


def plotdata(X):
    plt.figure()
    plt.scatter(X[:,0],X[:,1],marker='+',c='b')


'''status为True代表多元高斯模型，为False代表一元高斯模型'''
def getparams(X,status=False):
    mu=np.mean(X,axis=0) #(2,)
    if status==False:
        sigma2=np.mean((X-mu)**2,axis=0) #(2,)
        # sigma2=X.var(axis=0,ddof=0)
    else:
        sigma2=((X-mu).T@(X-mu))/len(X) #(2,2)
    return mu,sigma2


'''计算高斯分布概率的方法'''

#法1⃣️原模型 p(x)=p(x1;mu1,sigma2_1)*p(x2;mu2,sigma2_2)*...
#其中sigma2是向量

def P(X,mu,sigma2):
    a=(1./np.sqrt(2*np.pi*sigma2))*np.exp(-((X-mu)**2)/(2*sigma2))
    n=a.shape[1] 
    i=0
    p=1
    while i<n: 
        p*=a[:,i] 
        i+=1
    return p                   #(m,)

#法2⃣️ 多变量高斯模型 
# p(x)=p(x;mu,sigma2) 
# 其中sigma2是形状为（n，n）的协方差矩阵，所以当传入的sigma2为向量时要变换成协方差矩阵
# 采用矩阵相乘求解exp()中的项
# ⚠️当矩阵过大时，numpy矩阵相乘会出现内存错误。
# eg. gaussian(T,mu,sigma2) T的形状为（94294，2），在程序执行到该步骤时，报Killed: 9。

def gaussian(X,mu,sigma2):
    n=X.shape[1]

    #原始模型是多元高斯模型在sigma2上是对角矩阵而已
    if sigma2.ndim==1: #(n,)
        sigma2=np.diag(sigma2)     #(n,n)

    #如果想用矩阵相乘求解exp()中的项，一定要注意维度的变换。
    first=1./( np.power(2*np.pi,n/2) * np.sqrt(np.linalg.det(sigma2)) ) #constant
    e=(X-mu)@np.linalg.inv(sigma2)@(X-mu).T   #(m,m)
    #事实上我们只需要取对角线上的元素即可!（类似于方差而不是想要协方差）
    #最后得到一个（m，）的向量，包含每个样本的概率，而不是想要一个（m,m）的矩阵
    second=np.exp(-.5*np.diag(e))             #(m,)
    return first*second                       #(m,)


#法3⃣️ 多变量高斯模型 p(x)=p(x;mu,sigma2)
# 采用矩阵相乘的方法计算高斯分布概率进行画图时不能生成太多数据！！！！
# 可是遗憾的是，为了使等高线是原数据的等高线，我就是要这么多数据啊，怎么办呢～～～
# 那就不用矩阵呀！把每行数据输入进去，就不会出现内存错误。

def Gaussian(X,mu,sigma2):
    m,n=X.shape

    #当传入的sigma2为向量时要变换成协方差矩阵
    if sigma2.ndim==1: #(n,)
        sigma2=np.diag(sigma2)     #(n,n)


    first=1./( np.power(2*np.pi,n/2) * np.sqrt(np.linalg.det(sigma2)) ) #constant
    # second=[]
    second=np.zeros((m,1))
    for i in range(m):
        e=(X[i]-mu).T@np.linalg.inv(sigma2)@(X[i]-mu)
        # second.append(np.exp(-.5*e)) 
        second[i]=np.exp(-.5*e)

    # second=np.array(second)         #(m,)
    return first*second             #(m,)


'''画图'''
# 一元高斯模型仅在横向和纵向上有变化，而多元高斯模型在斜轴上也有相关变化，对应着特征间的相关关系。
# 一元高斯模型就是多元高斯模型中协方差矩阵为对角矩阵的结果，即协方差都为0，不考虑协方差，只考虑方差，故一元高斯模型不会有斜轴上的变化。

def plot3dp(X,mu,sigma2):
    print (sigma2.shape)

    #原数据概率分布三维散点图
    x1=X[:,0]
    x2=X[:,1]

    '''三选一'''
    p=Gaussian(X,mu,sigma2)

    plt.figure(figsize=(8,5))
    ax=plt.subplot(121,projection='3d')
    ax.scatter3D(x1,x2,p)

    #🤔️其实不是很明白为什么不是x1，x2给meshgrid，而且用x1，x2画出来的等高线出来的图像乱七八糟的
    # 换新数据，用x，y来画等高线图案就正确了，但是为啥呢？？？
    # 而且x和y明明与原数据X都不是一份数据啊怎么二维散点图和等高线就对上了呢？？？
    x = np.arange(0,30,.3)
    y = np.arange(0,30,.3)
    xx,yy=np.meshgrid(x,y)

    T=np.c_[xx.ravel(),yy.ravel()]  
    
    '''二选一（因为zz=gaussian(T,mu,sigma2).reshape(xx.shape) 会报内存错误）'''
    zz=Gaussian(T,mu,sigma2).reshape(xx.shape) 

    #概率分布三维曲面图
    ax2=plt.subplot(122,projection='3d')
    ax2.plot_surface(xx,yy,zz)

    #原数据二维平面散点图和新数据的等高线图
    plotdata(X)
    cont_levels = [10**h for h in range(-20,0,3)]
    plt.contour(xx,yy,zz,cont_levels) # 这个levels是作业里面给的参考,或者通过求解的概率推出来。
    if sigma2.ndim==1:
        plt.title('原模型',fontsize=16)
    if sigma2.ndim==2:
        plt.title('多变量高斯分布',fontsize=16)


'''select threshold epsilion via f1 score through cv dataset'''

def selectthreshold(yval,pval):

    def getf1(yval,ypred):
        tp=sum(np.logical_and(ypred==1,yval==1))
        fp=sum(np.logical_and(ypred==1,yval==0))
        fn=sum(np.logical_and(ypred==0,yval==1))

        #之所以要有这些条件语句，是因为e最开始取值为pval的最小值，那么没有条件能满足pval<e
        #因此，ypred中的元素将全部为0，则sum(ypred==1)为0，tp为0，recall为0
        #由于分母不能为0，所以我们必须要对precise和f1做些处理，否则第一个f1会是nan值
        if sum(ypred==1):
            precise=tp/sum(ypred==1)  
        else:
            precise= 0 

        recall=tp/sum(yval==1)

        if precise+recall: 
            f1=2*precise*recall/(precise+recall)
        else:
            f1=0

        return f1

    allf1=[]
    epsilions=np.linspace(min(pval),max(pval),1000)
    # bestf1,beste=0,0

    for e in epsilions:
        ypred=np.zeros(len(yval))
        ypred[np.where(pval<e)]=1

        f1=getf1(yval,ypred)
        allf1.append(f1)
        # if f1>bestf1:
        #     bestf1=f1
        #     beste=e

    # 其实我们很容易就能知道阈值越大，f1越小，所以e最开始取值为pval的最小值时，所得的f1就是bestf1（是个nan值），该e就是beste）
    # 但是结合👆条件语句那些分析，我得到的ypred中的元素将全部为0，这样对我异常检测有什么意义呢.....
    # 所以呀，我必须得把最开始得到的f1设置为0，让程序认为比pval最小值大一丢丢的那个e才是我们真正需要的
    allf1=np.array(allf1)
    print (allf1) 
    bestf1=allf1[np.argmax(allf1)]
    beste=epsilions[np.argmax(allf1)]
    return bestf1,beste


'''低维数据集'''

# data=io.loadmat('/Users/zhangying/Desktop/243-ML-Ng/ex8-anomaly detection and recommendation/data/ex8data1.mat')
# X,Xval,yval=data['X'],data['Xval'],data['yval']   #(307, 2) (307, 2) (307, 1)

# plot3dp(X,*getparams(X,False))  # *表示解元组

# mu,sigma2=getparams(X,status=False)

# yval=yval.ravel()
# pval=gaussian(Xval,mu,sigma2)
# bestf1,threshold=selectthreshold(yval,pval) #0.8750000000000001 8.999852631901397e-05

# anomalyX=X[np.where(gaussian(X,mu,sigma2)<threshold)]
# plt.scatter(anomalyX[:,0],anomalyX[:,1],s=50,facecolors='None',edgecolors='red')
# plt.show()



'''高维数据集'''

mat=io.loadmat('/Users/zhangying/Desktop/243-ML-Ng/ex8-anomaly detection and recommendation/data/ex8data2.mat')

X2,Xval2,yval2=mat['X'],mat['Xval'],mat['yval']
#(1000, 11) (100, 11) (100, 1)

Mu,Sigma2=getparams(X2,status=False)

yval2=yval2.ravel()
pval2=gaussian(Xval2,Mu,Sigma2)
Bestf1,Threshold=selectthreshold(yval2,pval2)
#0.6153846153846154 1.3786074982000245e-18

p=gaussian(X2,Mu,Sigma2)
print ('异常样本数量为%d'%np.sum(p<Threshold))
print ('异常样本如下\n',X2[np.where(p<Threshold)])