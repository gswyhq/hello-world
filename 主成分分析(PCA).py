#----coding:utf-8----------------------------------------------------------------
# 名称: 主成分分析(PCA)
# 目的:
# 参考:
# 作者:      gswyhq
#
# 日期:      2016-01-23
# 版本:      Python 3.3.5
# 系统:      win32
# Email:     gswyhq@126.com
#-------------------------------------------------------------------------------
# 1、PCA算法介绍
# 主成分分析（Principal Components Analysis），简称PCA，是一种数据降维技术，用于数据预处理。
#一般我们获取的原始数据维度都很高，比如1000个特征，在这1000个特征中可能包含了很多无用的信息或者噪声，
#真正有用的特征才100个，那么我们可以运用PCA算法将1000个特征降到100个特征。这样不仅可以去除无用的噪声，还能减少很大的计算量。
# PCA算法是如何实现的？
# 简单来说，就是将数据从原始的空间中转换到新的特征空间中，例如原始的空间是三维的(x,y,z)，x、y、z分别是原始空间的三个基，
#我们可以通过某种方法，用新的坐标系(a,b,c)来表示原始的数据，那么a、b、c就是新的基，它们组成新的特征空间。
#在新的特征空间中，可能所有的数据在c上的投影都接近于0，即可以忽略，那么我们就可以直接用(a,b)来表示数据，
#这样数据就从三维的(x,y,z)降到了二维的(a,b)。
# 问题是如何求新的基(a,b,c)?
# 一般步骤是这样的：先对原始数据零均值化，然后求协方差矩阵，接着对协方差矩阵求特征向量和特征值，
#这些特征向量组成了新的特征空间。具体的细节，推荐Andrew Ng的网页教程：Ufldl 主成分分析 (http://deeplearning.stanford.edu/wiki/index.php/%E4%B8%BB%E6%88%90%E5%88%86%E5%88%86%E6%9E%90)，写得很详细。

#（1）零均值化
#假如原始数据集为矩阵dataMat，dataMat中每一行代表一个样本，每一列代表同一个特征。
#零均值化就是求每一列的平均值，然后该列上的所有数都减去这个均值。也就是说，这里零均值化是对每一个特征而言的，零均值化都，每个特征的均值变成0。
import numpy as np
def zeroMean(dataMat):
    meanVal=np.mean(dataMat,axis=0)     #按列求均值，即求各个特征的均值
    newData=dataMat-meanVal
    return newData,meanVal
#函数中用numpy中的mean方法来求均值，axis=0表示按列求均值。
#该函数返回两个变量，newData是零均值化后的数据，meanVal是每个特征的均值，是给后面重构数据用的。

#应用PCA的时候，对于一个1000维的数据，我们怎么知道要降到几维的数据才是合理的？即n要取多少，才能保留最多信息同时去除最多的噪声？
#一般，我们是通过方差百分比来确定n的，这一点在Ufldl教程中说得很清楚，并且有一条简单的公式:前n个特征保留方差百分比
#根据这条公式，可以写个函数，函数传入的参数是百分比percentage和特征值向量，然后根据percentage确定n
def percentage2n(eigVals,percentage=0.99):
    sortArray=np.sort(eigVals)   #升序
    sortArray=sortArray[-1::-1]  #逆转，即降序
    arraySum=sum(sortArray)
    tmpSum=0
    num=0
    for i in sortArray:
        tmpSum+=i
        num+=1
        if tmpSum>=arraySum*percentage:
            return num


def pca(dataMat,n=10,percentage=0):
    """PCA算法的步骤
    http://my.oschina.net/dfsj66011/blog/513665
① 样本矩阵X的构成
    假设待观察变量有M个属性，相当于一个数据在M维各维度上的坐标，我们的目标是在
保证比较数据之间相似性不失真的前提下，将描述数据的维度尽量减小至L维（L<M）。
    样本矩阵X在这里用x 1 ,x 2 ,…,x N 共N个数据（这些数据都是以列向量的形式出现）
来表示，那么X=[x 1  x 2  … x N ] MxN 。
② 计算样本X均值
    计算第m维（m=1,2,…,M）的均值如下：
    u[m]=1/N*sum(X[m,i]) 其中i∈[1,N]
③ 计算观察值与均值的偏差
    在每一维上，用当前值X[m,n]减去u[m]，用矩阵运算表示如下
    B=X-u*h
    h[i]=1,其中i∈[1,N]
    明显，h是一行向量，u是一列向量。

④ 计算协方差矩阵
    我们认为b i 代表B的第i行，那么由协方差矩阵
    C=E[B×B]=E[B·B*]=1/N*sum(B·B*)
    推知:c[i,j]=1/N*<bi,bj>

    <>表示向量的内积，也就是每一项的积之和。
⑤ 计算协方差矩阵C的特征值和特征向量
    若XA=aA,其中A为一列向量，a为一实数。那么a就被称为矩阵X的特征值，而A则是特征值a对应的特征向量。
    顺便扯一下，在matlab里面，求解上述A以及a，使用eig函数。如[D,V] = eig(C)，那么D就是n个列特征向量，而V是对角矩阵，对角线上的元素就是特征值。
⑥ 排序
    将特征值按 从大到小 的顺序排列，并根据特征值调整特征向量的排布。
    D’=sort(D);V’=sort(V);
⑦ 计算总能量并选取其中的较大值
    若V’为C的对角阵，那么总能量为对角线所有特征值之和S。
    由于在步骤⑥里面已对V进行了重新排序，所以当v’前几个特征值之和大于等于S的90%时，可以认为这几个特征值可以用来"表征"当前矩阵。
    假设这样的特征值有L个。
⑧ 计算基向量矩阵W
    实际上，W是V矩阵的前L列，所以W的大小就是 MxL 。
⑨ 计算z-分数（这一步可选可不选）
     Z[i,j]=B[i,j]/sqrt(D[i,i])
⑩ 计算降维后的新样本矩阵
     Y=W*·Z=KLT{X}
     W * 表示W的转置的共轭矩阵，大小为 LxM , 而Z的大小为 MxN , 所以Y的大小为 LxN , 即降维为 N 个 L 维向量。
总结一下：
    去除平均值
    计算协方差矩阵
    计算协方差矩阵的特征值和特征向量
    将特征值从大到小排序
    保留最上面的N个特征值

"""
    #numpy中的cov函数用于求协方差矩阵，参数rowvar很重要！若rowvar=0，说明传入的数据一行代表一个样本，
    #若非0，说明传入的数据一列代表一个样本。因为newData每一行代表一个样本，所以将rowvar设置为0。
    #covMat即所求的协方差矩阵。
    newData,meanVal=zeroMean(dataMat)#减去均值后的数值，每列的均值
    covMat=np.cov(newData,rowvar=0)    #求协方差矩阵,return ndarray；若rowvar非0，一列代表一个样本，为0，一行代表一个样本

    #（3）求特征值、特征矩阵
    #调用numpy中的线性代数模块linalg中的eig函数，可以直接由covMat求得特征值和特征向量：
    #eigVals存放特征值，行向量。
    #eigVects存放特征向量，每一列带别一个特征向量。
    #特征值和特征向量是一一对应的
    eigVals,eigVects=np.linalg.eig(np.mat(covMat))#求特征值和特征向量,特征向量是按列放的，即一列代表一个特征向量

    #（4）保留主要的成分[即保留值比较大的前n个特征]
    #第三步得到了特征值向量eigVals，假设里面有m个特征值，我们可以对其排序，排在前面的n个特征值所对应的特征向量就是我们要保留的，
    #它们组成了新的特征空间的一组基n_eigVect。将零均值化后的数据乘以n_eigVect就可以得到降维后的数据。

    if percentage:
        n=percentage2n(eigVals,percentage)                 #要达到percent的方差百分比，需要前n个特征向量

    eigValIndice=np.argsort(eigVals)            #对特征值从小到大排序
    n_eigValIndice=eigValIndice[-1:-(n+1):-1]   #最大的n个特征值的下标
    n_eigVect=eigVects[:,n_eigValIndice]        #最大的n个特征值对应的特征向量
    lowDDataMat=newData*n_eigVect               #将数据转换到新的空间;低维特征空间的数据
    reconMat=(lowDDataMat*n_eigVect.T)+meanVal  #reconMat是重构的数据，乘以n_eigVect的转置矩阵，再加上均值meanVal。
    return lowDDataMat,reconMat

def pca1(dataMat,n=10,percentage=0):
    #numpy中的cov函数用于求协方差矩阵，参数rowvar很重要！若rowvar=0，说明传入的数据一行代表一个样本，
    #若非0，说明传入的数据一列代表一个样本。因为newData每一行代表一个样本，所以将rowvar设置为0。
    #covMat即所求的协方差矩阵。
    import pandas as pd
    df=pd.DataFrame (dataMat)
    newData,meanVal=zeroMean(dataMat)#减去均值后的数值，每列的均值
    print("每列的均值：\n",df.mean(axis=0))
    #covMat=np.cov(newData,rowvar=0)    #求协方差矩阵,return ndarray；若rowvar非0，一列代表一个样本，为0，一行代表一个样本
    covMat=df.cov()#计算协方差
    #（3）求特征值、特征矩阵
    #调用numpy中的线性代数模块linalg中的eig函数，可以直接由covMat求得特征值和特征向量：
    #eigVals存放特征值，行向量。
    #eigVects存放特征向量，每一列带别一个特征向量。
    #特征值和特征向量是一一对应的
    eigVals,eigVects=np.linalg.eig(np.mat(covMat))#求特征值和特征向量,特征向量是按列放的，即一列代表一个特征向量

    #（4）保留主要的成分[即保留值比较大的前n个特征]
    #第三步得到了特征值向量eigVals，假设里面有m个特征值，我们可以对其排序，排在前面的n个特征值所对应的特征向量就是我们要保留的，
    #它们组成了新的特征空间的一组基n_eigVect。将零均值化后的数据乘以n_eigVect就可以得到降维后的数据。

    if percentage:
        n=percentage2n(eigVals,percentage)                 #要达到percent的方差百分比，需要前n个特征向量

    eigValIndice=np.argsort(eigVals)            #对特征值从小到大排序
    n_eigValIndice=eigValIndice[-1:-(n+1):-1]   #最大的n个特征值的下标
    n_eigVect=eigVects[:,n_eigValIndice]        #最大的n个特征值对应的特征向量
    for i in n_eigValIndice:
        print("最大的n个特征值对应的原数据集列标签：\n",df.columns.values.tolist()[int(i)])
    lowDDataMat=newData*n_eigVect               #将数据转换到新的空间;低维特征空间的数据
    reconMat=(lowDDataMat*n_eigVect.T)+meanVal  #reconMat是重构的数据，乘以n_eigVect的转置矩阵，再加上均值meanVal。
    return lowDDataMat,reconMat

def main():
    data=np.array([[-1.48916667, -1.50916667],
       [-1.58916667, -1.55916667],
       [1.47916667, -1.47916667],
       [-0.48916667, -1.50916667],
       [-0.45916667, -0.44916667],
       [0.50916667, -2.61916667],
       [ 0.51083333,  0.49083333],
       [ 0.54083333,  1.54083333],
       [ 2.40083333,  0.59083333],
       [ 1.51083333,  1.49083333],
       [ 1.57083333,  1.51083333],
       [ 1.48083333,  1.50083333]])
    newdata=pca(data,1)
    print(newdata)

if __name__ == '__main__':
    main()
