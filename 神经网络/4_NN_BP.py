#coding: UTF-8
import numpy as np
import matplotlib.pyplot as plt
from scipy.io import loadmat

data=loadmat('/Users/zhangying/Desktop/243-ML-Ng/ex4-NN back propagation/ex4data1.mat')
X=data['X'] #(5000,400)
y=data['y'].flatten() #(5000,)

def plot_100_images(X):
    index=np.random.choice(range(5000),100) 
    images=X[index]
    _,axs=plt.subplots(10,10,sharex=True,sharey=True)
    for i in np.arange(10):
        for j in np.arange(10):
            axs[i,j].matshow(images[10*i+j,:].reshape((20,20)),cmap='gray_r')
    plt.xticks([])
    plt.yticks([])
    plt.show()
            
# plot_100_images(X) #太慢了

X=np.insert(X,0,np.ones(len(X)),axis=1) #(5000,401)

Y=np.zeros((len(y),10)) #(5000,10)
for each in np.arange(5000):
    labelvalue=y[each]
    Y[each][labelvalue-1]=1

'''
result=[]
for value in y: #y (5000,)
    yarr=np.zeros(10)
    yarr[int(value)-1]=1
    result.append(yarr)
Y=np.array(result) #(5000,10)

print (Y[:5]) # 效果和 Y[:5,:] 一样
'''

params=loadmat('/Users/zhangying/Desktop/243-ML-Ng/ex4-NN back propagation/ex4weights.mat')
theta1=params['Theta1'] #ndarray(25, 401)
theta2=params['Theta2'] #ndarray(10, 26)

def serialize(a,b):
    return np.r_[a.ravel(),b.ravel()]  

thetavector=serialize(theta1,theta2)  #(10285,) 

def deserialize(seq):
    return seq[:10025].reshape(25,401),seq[10025:].reshape(10,26)


def g(z):
    return 1./(1+np.exp(-z))

def forwardpropogation(thetavector,X):
    t1,t2=deserialize(thetavector)
    a1=X             #(5000,401)
    z2=a1@t1.T       #(5000,25)
    a2=np.insert(g(z2),0,np.ones(len(X)),axis=1) #(5000,26)
    z3=a2@t2.T       #(5000,10)
    a3=g(z3)         #(5000,10)
    return a1,z2,a2,z3,a3


def cost(thetavector,X,Y): 
    a1,z2,a2,z3,h=forwardpropogation(thetavector,X)
    return np.sum(-Y*np.log(h)-(1-Y)*np.log(1-h))/len(X)

def costReg(thetavector,X,Y,l):
    """the first column of t1 and t2 is intercept theta, ignore them when you do regularization"""
    t1,t2=deserialize(thetavector)
    _theta1=t1[:,1:]
    _theta2=t2[:,1:]
    reg=(l/(2*len(X)))*(np.sum(np.power(_theta1,2))+np.sum(np.power(_theta2,2)))
    return cost(thetavector,X,Y)+reg

# print (cost(thetavector,X,Y)) #初始未正则化代价函数成本0.2876291651613189
# print (costReg(thetavector,X,Y,1)) #初始正则化代价函数成0.38376985909092365


# 训练神经网络时，随机初始化参数可以打破数据的对称性。
# 一个有效的策略是在均匀分布(−e，e)中随机选择值，可选择 e = 0.12 确保参数足够小，使得训练更有效率。
def random_init(epsilon=0.12):
    return np.random.uniform(-epsilon,epsilon,10285)
#或者
def initializeparams(epsilon):
    return np.random.random_sample(10285)*2*epsilon-epsilon

def gradient_bp(thetavector,X,Y):
    t1,t2=deserialize(thetavector)
    #(25,401) #(10,26)
    a1,z2,a2,z3,a3=forwardpropogation(thetavector,X)
    #(5000,401) (5000,25) (5000,26) (5000,10) (5000,10)
    d3=a3-Y  #(5000,10)
    d2=(d3@t2[:,1:])*g(z2)*(1-g(z2)) #注意偏置项！(5000,25)
    #返回所有参数的梯度，所以梯度D和参数t的shape相同
    D2=d3.T@a2 #(10,26)
    D1=d2.T@a1 #(25,401)
    D=(1./len(X))*serialize(D1,D2)
    return D


def Reg_gradient_bp(thetavector,X,Y,l):
    D1,D2=deserialize(gradient_bp(thetavector,X,Y)) #(25, 401) (10, 26)
    t1,t2=deserialize(thetavector) #(25,401) #(10,26)

    t1[:,0]=0
    regD1=l/len(X)*t1 #(25,401) 
    t2[:,0]=0
    regD2=l/len(X)*t2 #(10,26)

    return serialize(D1+regD1,D2+regD2) #(10285,)


def gradientcheck(thetavector,X,Y,e):
    gc=np.zeros(len(thetavector))

    for i in range(len(thetavector)):
        plus=thetavector.copy() #！！！！就是这里出错，找了好久，因为我没有copy！！！# deep copy otherwise you will change the raw theta
        minus=thetavector.copy()
        plus[i]+=e
        minus[i]-=e
        gc[i]=(cost(plus,X,Y)-cost(minus,X,Y))/(2*e)
        # gc[i]=(costReg(plus,X,Y,1)-costReg(minus,X,Y,1))/(2*e)

    gbp=gradient_bp(thetavector,X,Y)  
    # gbp=Reg_gradient_bp(thetavector,X,Y,1) 
    print (gc.shape, gbp.shape) #(10285,) (10285,)


    diff=np.linalg.norm(gc-gbp)/np.linalg.norm(gc+gbp)
    print (diff) # 采用原始数据给出的thetavector下，非正则化情况2.1448373423800885e-09 or 正则化情况3.1766894924035752e-09
    
    if diff<10*np.exp(-9):
        print ('给定epsilon为0.0001下，bp算法与梯度检测法差值(%d)不超过10e^-9,故判断bp算法正确执行'%diff)

# thetavector=random_init(epsilon=0.12)
# thetavector=initializeparams(epsilon=0.12)

# gradientcheck(thetavector,X,Y,0.0001)  #检验出bp算法是正确执行的，需要关闭梯度检测！不然程序计算量太大

from scipy.optimize import minimize

res=minimize(fun=costReg,x0=thetavector,args=(X,Y,1),method='TNC',jac=Reg_gradient_bp,options={'maxiter':400})
# print (res)

from sklearn.metrics import classification_report

def accuracy(thetavector,X,y):
    _,_,_,_,h=forwardpropogation(thetavector,X)
    label_pred=np.argmax(h,axis=1)+1
    print (classification_report(y,label_pred))

accuracy(res.x,X,y)


'''
给定一个隐藏层单元，可视化它所计算的内容的方法是找到一个输入x，x可以激活这个单元。
注意到θ1中每一行都是一个401维的向量，代表每个隐藏层单元的参数。
如果我们忽略偏置项，我们就能得到400维的向量，这个向量代表每个样本输入到每个隐层单元的像素的权重。
因此可视化的一个方法是，reshape这个400维的向量为（20，20）的图像然后输出。
'''
def plot_hidden(thetavector):
    theta1,_=deserialize(thetavector)
    theta1=theta1[:,1:]
    _,axs=plt.subplots(5,5,sharex=True,sharey=True,figsize=(5,5))
    for i in range(5):
        for j in range(5):
            axs[i,j].matshow(theta1[i*5+j].reshape(20,20),cmap='gray_r')
    plt.xticks([])
    plt.yticks([])
    plt.show()

plot_hidden(res.x) 