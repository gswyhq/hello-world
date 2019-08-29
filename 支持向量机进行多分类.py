#!/usr/bin/python3
# coding=utf-8

from __future__ import division
import numpy as np

# https://blog.csdn.net/lx_ros/article/details/81294220

"""
此数据集是MNIST数据集的一部分，但只有3个类，
classes = {0：'0'，1：'1'，2：'2'}，图像被压缩为14 * 14
像素并存储在具有相应标签的矩阵中
结束数据矩阵的形状是
num_of_images x 14*14(pixels)+1(lable)
"""
def load_data(split_ratio):
    # 训练数据获取地址:https://raw.githubusercontent.com/lxrobot/General-source-code/master/SVM/data216x197.npy
    tmp=np.load("/home/gswyhq/Downloads/data216x197.npy")
    data=tmp[:,:-1]
    label=tmp[:,-1]
    mean_data=np.mean(data,axis=0)
    train_data=data[int(split_ratio*data.shape[0]):]-mean_data
    train_label=label[int(split_ratio*data.shape[0]):]
    test_data=data[:int(split_ratio*data.shape[0])]-mean_data
    test_label=label[:int(split_ratio*data.shape[0])]
    return train_data,train_label,test_data,test_label

"""在不使用向量运算的情况下计算Hingle损失，
在处理庞大的数据集时，效率会很低
X's shape [n,14*14+1],Y's shape [n,],W's shape [num_class,14*14+1]"""

def lossAndGradNaive(X,Y,W,reg):
    dW=np.zeros(W.shape)
    loss = 0.0
    num_class=W.shape[0]
    num_X=X.shape[0]
    for i in range(num_X):
        scores=np.dot(W,X[i])
        cur_scores=scores[int(Y[i])]
        for j in range(num_class):
            if j==Y[i]:
                continue
            margin=scores[j]-cur_scores+1
            if margin>0:
                loss+=margin
                dW[j,:]+=X[i]
                dW[int(Y[i]),:]-=X[i]
    loss/=num_X
    dW/=num_X
    loss+=reg*np.sum(W*W)
    dW+=2*reg*W
    return loss,dW

def lossAndGradVector(X,Y,W,reg):
    dW=np.zeros(W.shape)
    N=X.shape[0]
    Y_=X.dot(W.T)
    margin=Y_-Y_[range(N),Y.astype(int)].reshape([-1,1])+1.0
    margin[range(N),Y.astype(int)]=0.0
    margin=(margin>0)*margin
    loss=0.0
    loss+=np.sum(margin)/N
    loss+=reg*np.sum(W*W)
    """For one data,the X[Y[i]] has to be substracted several times"""
    countsX=(margin>0).astype(int)
    countsX[range(N),Y.astype(int)]=-np.sum(countsX,axis=1)
    dW+=np.dot(countsX.T,X)/N+2*reg*W
    return loss,dW

def predict(X,W):
    X=np.hstack([X, np.ones((X.shape[0], 1))])
    Y_=np.dot(X,W.T)
    Y_pre=np.argmax(Y_,axis=1)
    return Y_pre

def accuracy(X,Y,W):
    Y_pre=predict(X,W)
    acc=(Y_pre==Y).mean()
    return acc

def model(X,Y,alpha,steps,reg):
    X=np.hstack([X, np.ones((X.shape[0], 1))])
    W = np.random.randn(3,X.shape[1]) * 0.0001
    for step in range(steps):
        loss,grad=lossAndGradNaive(X,Y,W,reg)
        W-=alpha*grad
        print("The {} step, loss={}, accuracy={}".format(step,loss,accuracy(X[:,:-1],Y,W)))
    return W

train_data,train_label,test_data,test_label=load_data(0.2)
W=model(train_data,train_label,0.0001,25,0.5)
print("Test accuracy of the model is {}".format(accuracy(test_data,test_label,W)))

def main():
    pass


if __name__ == '__main__':
    main()