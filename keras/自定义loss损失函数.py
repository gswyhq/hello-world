#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import numpy as np
import tensorflow as tf
from keras import backend as K
from keras.models import Sequential
from keras.layers import Dense

from sklearn.datasets import load_breast_cancer

from keras.layers import Input,Embedding,LSTM,Dense
from keras.models import Model
from keras import backend as K
from keras.losses import binary_crossentropy

word_size = 128
nb_features = 10000
nb_classes = 10
encode_size = 64

input = Input(shape=(None,))
embedded = Embedding(nb_features,word_size)(input)
encoder = LSTM(encode_size)(embedded)
predict = Dense(nb_classes, activation='softmax')(encoder)

def mycrossentropy(y_true, y_pred, e=0.1):
    # 第一项就是普通的交叉熵，第二项中，先通过K.ones_like(y_pred)/nb_classes构造了一个均匀分布，然后算y_pred与均匀分布的交叉熵。
    loss1 = K.categorical_crossentropy(y_true, y_pred)
    loss2 = K.categorical_crossentropy(K.ones_like(y_pred)/nb_classes, y_pred)
    return (1-e)*loss1 + e*loss2

model = Model(inputs=input, outputs=predict)
model.compile(optimizer='adam', loss=mycrossentropy)


y_true = tf.constant([1., 0., 0., 0., 1., 0., 0., 0., 1.], shape=[3, 3])
y_pred = tf.constant([.9, .05, .05, .05, .89, .06, .05, .01, .94], shape=[3, 3])
mycrossentropy(y_true, y_pred, e=0.1)

# 来源：
# https://analysiscenter.github.io/radio/api/keras_loss.html
# https://www.kaggle.com/code/bigironsphere/loss-function-library-keras-pytorch#Dice-Loss

def dice_loss(y_true, y_pred, smooth=1e-6):
    """ Loss function base on dice coefficient.
    Parameters
    ----------
    y_true : keras tensor
        tensor containing target mask.
    y_pred : keras tensor
        tensor containing predicted mask.
    smooth : float
        small real value used for avoiding division by zero error.
    Returns
    -------
    keras tensor
        tensor containing dice loss.
    """
    y_true=tf.cast(y_true,tf.float32)
    y_pred=tf.cast(y_pred,tf.float32)
    y_true_f = K.flatten(y_true)
    y_pred_f = K.flatten(y_pred)
    intersection = K.sum(y_true_f * y_pred_f)
    dice = (2. * intersection + smooth) / (K.sum(y_true_f) + K.sum(y_pred_f) + smooth)
    return 1 - dice

##########################################################################################################################################################
def tversky_loss(y_true, y_pred, alpha=0.3, beta=0.7, smooth=1e-10):
    """ Tversky loss function.
    Parameters
    ----------
    y_true : keras tensor
        tensor containing target mask.
    y_pred : keras tensor
        tensor containing predicted mask.
    alpha : float
        real value, weight of '0' class.
    beta : float
        real value, weight of '1' class.
    smooth : float
        small real value used for avoiding division by zero error.
    Returns
    -------
    keras tensor
        tensor containing tversky loss.
    """
    y_true=tf.cast(y_true,tf.float32)
    y_pred=tf.cast(y_pred,tf.float32)
    y_true = K.flatten(y_true)
    y_pred = K.flatten(y_pred)
    truepos = K.sum(y_true * y_pred)
    fp_and_fn = alpha * K.sum(y_pred * (1 - y_true)) + beta * K.sum((1 - y_pred) * y_true)
    Tversky = (truepos + smooth) / ((truepos + smooth) + fp_and_fn)
    return 1-Tversky

##########################################################################################################################################################
def jaccard_coef_logloss(y_true, y_pred, smooth=1e-10):
    """ Loss function based on jaccard coefficient.
    Parameters
    ----------
    y_true : keras tensor
        tensor containing target mask.
    y_pred : keras tensor
        tensor containing predicted mask.
    smooth : float
        small real value used for avoiding division by zero error.
    Returns
    -------
    keras tensor
        tensor containing negative logarithm of jaccard coefficient.
    """
    y_true=tf.cast(y_true,tf.float32)
    y_pred=tf.cast(y_pred,tf.float32)
    y_true = K.flatten(y_true)
    y_pred = K.flatten(y_pred)
    truepos = K.sum(y_true * y_pred)
    falsepos = K.sum(y_pred) - truepos
    falseneg = K.sum(y_true) - truepos
    jaccard = (truepos + smooth) / (smooth + truepos + falseneg + falsepos)
    return 1-K.log(jaccard + smooth)

##########################################################################################################################################################

def DiceBCELoss(y_true, y_pred, smooth=1e-6):
    # flatten label and prediction tensors
    targets=tf.cast(y_true,tf.float32)
    inputs=tf.cast(y_pred,tf.float32)
    inputs = K.flatten(inputs)
    targets = K.flatten(targets)
    BCE = binary_crossentropy(targets, inputs)
    intersection = K.sum(targets* inputs)
    dice_loss = 1 - (2 * intersection + smooth) / (K.sum(targets) + K.sum(inputs) + smooth)
    Dice_BCE = BCE + dice_loss
    return Dice_BCE

##########################################################################################################################################################

def IoULoss(y_true, y_pred, smooth=1e-6):
    # flatten label and prediction tensors
    targets=tf.cast(y_true,tf.float32)
    inputs=tf.cast(y_pred,tf.float32)
    inputs = K.flatten(inputs)
    targets = K.flatten(targets)

    intersection = K.sum(targets* inputs)
    total = K.sum(targets) + K.sum(inputs)
    union = total - intersection

    IoU = (intersection + smooth) / (union + smooth)
    return 1 - IoU

##########################################################################################################################################################

def FocalLoss(y_true, y_pred, alpha=0.8, gamma=2.0):
    targets=tf.cast(y_true,tf.float32)
    inputs=tf.cast(y_pred,tf.float32)
    inputs = K.flatten(inputs)
    targets = K.flatten(targets)

    BCE = K.binary_crossentropy(targets, inputs)
    BCE_EXP = K.exp(-BCE)
    focal_loss = K.mean(alpha * K.pow((1 - BCE_EXP), gamma) * BCE)

    return focal_loss

##########################################################################################################################################################

def FocalTverskyLoss(y_true, y_pred, alpha=0.5, beta=0.5, gamma=1.0, smooth=1e-6):
    # flatten label and prediction tensors
    targets=tf.cast(y_true,tf.float32)
    inputs=tf.cast(y_pred,tf.float32)
    inputs = K.flatten(inputs)
    targets = K.flatten(targets)

    # True Positives, False Positives & False Negatives
    TP = K.sum((inputs * targets))
    FP = K.sum(((1 - targets) * inputs))
    FN = K.sum((targets * (1 - inputs)))

    Tversky = (TP + smooth) / (TP + alpha * FP + beta * FN + smooth)
    FocalTversky = K.pow((1 - Tversky), gamma)

    return FocalTversky


##########################################################################################################################################################

def Combo_loss(y_true, y_pred, alpha = 0.5, ce_ratio = 0.5, eps=1e-9, smooth=1e-6):
    # alpha < 0.5 penalises FP more, > 0.5 penalises FN more
    # ce_ratio: weighted contribution of modified CE loss compared to Dice loss
    targets=tf.cast(y_true,tf.float32)
    inputs=tf.cast(y_pred,tf.float32)
    targets = K.flatten(targets)
    inputs = K.flatten(inputs)
    intersection = K.sum(targets * inputs)
    dice = (2. * intersection + smooth) / (K.sum(targets) + K.sum(inputs) + smooth)
    inputs = K.clip(inputs, eps, 1.0 - eps)
    out = - (alpha * ((targets * K.log(inputs)) + ((1 - alpha) * (1.0 - targets) * K.log(1.0 - inputs))))
    weighted_ce = K.mean(out, axis=-1)
    combo = (ce_ratio * weighted_ce) + ((1 - ce_ratio) * (1-dice))
    return combo

##########################################################################################################################################################
def GHMC_Loss(Bins=10, momentum=0, batch_size=100):
    global edges, mmt, bins, shape  ##
    shape = batch_size
    bins = Bins
    edges = [float(x) / bins for x in range(bins + 1)]
    edges[-1] += 1e-6
    mmt = momentum
    if momentum > 0:
        acc_sum = [0.0 for _ in range(bins)]

    def GHMC_Loss_fixed(y_true, y_pred):
        y_true = tf.cast(y_true, tf.float32)
        weights = tf.numpy_function(GHMC_loss_pyfunc, [y_true, y_pred], tf.float32)

        weights = tf.reshape(weights, [len(weights)])
        return tf.nn.weighted_cross_entropy_with_logits(labels=y_true, logits=y_pred, pos_weight=weights)

    return GHMC_Loss_fixed


def GHMC_loss_pyfunc(y_true, y_pred, momentum=0):
    """计算每个样本的权重
    weight = 总样本数/(当前类别数*总区间数)
    如样本x,pred=0.01, label=1, 总样本数为569，0-0.1区间有357个样本，故而其权重为: 569/(357*2)=0.7969187675070029
    同理样本y,pred=0.01, label=1, 总样本数为569，0-0.1区间有212个样本，故而其权重为: 569/(212*2)=1.3419811320754718
    总之，样本区间样本数越少，其权重越大；
    """
    weights = np.zeros(y_true.shape[0])
    g = np.abs(y_true - y_pred)
    tot = len(y_pred)
    n = 0
    for i in range(bins):
        inds = (g >= edges[i]) & (g < edges[i + 1])
        num_in_bin = inds.sum().item()
        if num_in_bin > 0:
            if momentum > 0:
                acc_sum[i] = momentum * acc_sum[i] + (1 - mmt) * num_in_bin
                weights[inds.flatten()] = tot / acc_sum[i]
            else:
                weights[inds.flatten()] = tot / num_in_bin  ##N/GD(gi)
            n += 1
    if n > 0:
        weights = weights / n

    return weights.astype('float32')

##########################################################################################################################################################
def my_loss1(y_true, y_pred):
    '''自定义loss,loss越低越好
    '''
    # 计算tp、tn、fp、fn
    tp = K.sum(K.cast(y_true * y_pred, 'float32'), axis=0)  # 真正
    tn = K.sum(K.cast((1 - y_true) * (1 - y_pred), 'float32'), axis=0) # 真负
    fp = K.sum(K.cast((1 - y_true) * y_pred, 'float32'), axis=0) # 假正，误报
    fn = K.sum(K.cast(y_true * (1 - y_pred), 'float32'), axis=0)  # 假负，漏报
    wubao_rate = (fp) / (fp + fn + tp + K.epsilon())
    loubao_rate = (fn) / (fp + fn + tp + K.epsilon())
    rate = (wubao_rate + loubao_rate) * (1-(2*wubao_rate*loubao_rate)/(wubao_rate+loubao_rate))  # 乘以均衡系数
    return K.mean(rate)


def my_loss2(y_true, y_pred, thr=0.5):
    '''自定义loss, loss越低越好,
    训练存在问题，模型不收敛或者收敛也不是最优；
    '''
    # 计算tp、tn、fp、fn, 计数
    tp = K.cast(tf.math.count_nonzero((y_true * y_pred) >= thr), 'float32')  # 真正
    tn = K.cast(tf.math.count_nonzero((1 - y_true) * (1 - y_pred) > thr), 'float32')  # 真负
    fp = K.cast(tf.math.count_nonzero(((1 - y_true) * y_pred) >= thr), 'float32')  # 误报
    fn = K.cast(tf.math.count_nonzero((y_true * (1 - y_pred)) > thr), 'float32')  # 漏报

    wubao_rate = (fp) / (fp + fn + tp + K.epsilon())
    loubao_rate = (fn) / (fp + fn + tp + K.epsilon())
    rate = (wubao_rate + loubao_rate) * (1 - (2 * wubao_rate * loubao_rate) / (wubao_rate + loubao_rate))  # 乘以均衡系数
    return K.mean(rate)

y_true = [0, 1, 1, 0, 0, 1, 1, 0]
y_score = [0.2, 0.6, 0.3, 0.2, 0.5, 0.6, 0.7, 0.4]
y_true, y_pred = tf.constant(np.array(y_true, dtype='float')), tf.constant(np.array(y_score, dtype='float'))
my_loss1(y_true, y_pred)

##########################################################################################################################################################
def f1_loss(y_true, y_pred, beta=1):
    # 计算tp、tn、fp、fn
    tp = K.sum(K.cast(y_true * y_pred, 'float32'), axis=0)  # 真正
    tn = K.sum(K.cast((1 - y_true) * (1 - y_pred), 'float32'), axis=0) # 真负
    fp = K.sum(K.cast((1 - y_true) * y_pred, 'float32'), axis=0) # 假正，误报
    fn = K.sum(K.cast(y_true * (1 - y_pred), 'float32'), axis=0)  # 假负，漏报
    # percision与recall，这里的K.epsilon代表一个小正数，用来避免分母为零
    p = tp / (tp + fp + K.epsilon())
    r = tp / (tp + fn + K.epsilon())
    # 计算f1
    #     f1 = 2*p*r / (p+r+K.epsilon())
    f1 = (beta * beta + 1) * p * r / (beta * beta * p + r + K.epsilon())
    f1 = tf.where(tf.math.is_nan(f1), tf.zeros_like(f1), f1)  # 其实就是把nan换成0
    return 1 - K.mean(f1)


def main():

    X = load_breast_cancer().data
    y = load_breast_cancer().target
    model = Sequential()
    model.add(Dense(60, input_dim=30, kernel_initializer='normal', activation='relu'))
    model.add(Dense(1, activation='sigmoid'))
    model.compile(loss=GHMC_Loss(Bins=10, momentum=0, batch_size=569), optimizer='adam', metrics=['accuracy'])
    model.fit(X, y, epochs=2, batch_size=569)

    f1_loss(tf.constant(np.array([0, 1, 0, 0, 1, 1, 1, 1, 1, 1, 0, 0, 1, 0, 0]), dtype='float'),
            tf.constant(
                np.array(
                    [0.0, 0.7, 0.0446, 0.0, 0.06668, 0.6279, 0.8957, 0.2, 1.0, 0.9, 0.61, 0.0, 0.0768, 0.109, 0.8]),
                dtype='float'))

    # 自定义F1 score作为loss，训练模型；
    model.compile(loss=f1_loss, optimizer='adam', metrics=['accuracy'])

if __name__ == '__main__':
    main()
