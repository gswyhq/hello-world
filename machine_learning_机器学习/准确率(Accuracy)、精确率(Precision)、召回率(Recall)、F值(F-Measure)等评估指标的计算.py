#!/usr/bin/python3
# coding: utf-8

import numpy as np
from sklearn.metrics import f1_score, accuracy_score, fbeta_score, precision_score, recall_score
from sklearn.metrics import classification_report
from sklearn.metrics import accuracy_score, f1_score, roc_auc_score
from sklearn.metrics import precision_recall_curve
from scipy import stats
from sklearn.metrics import det_curve, DetCurveDisplay

# 真实标签
y_true = [1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 1, 0, 0, 0, 1, 0, 0, 1, 0, 0, 0, 0, 0, 0, 1, 0, 1, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 0, 0, 1, 0, 1, 0, 0, 1, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1]

# 模型预测结果
y_test = [0.8453712241207609, 0.8365137845084419, 0.8396024690959464, 0.8690716625950063, 0.801398983655787, 0.8353417405844167, 0.8887589815396711, 0.8274617726584338, 0.8901324702288052, 0.8515827665762914, 0.8008748432690203, 0.9129143613344268, 0.8213637332093631, 0.7926672650384551, 0.8715962551942291, 0.865989576549353, 0.8487118383625984, 0.893722366823937, 0.8683798090835637, 0.8258107838161615, 0.9067962552630583, 0.8896577622207299, 0.8287242449131549, 0.862162050742874, 0.9145984088092137, 0.8195240228832353, 0.8627208683955114, 0.8667420865435141, 0.833175478131922, 0.8338735760735464, 0.8609573544733866, 0.8270040835455006, 0.8438342928159803, 0.9162216060491829, 0.8681943043237748, 0.825237777063406, 0.9309199493779501, 0.847918698600505, 0.885842165942269, 0.845606331185933, 0.8867428557974891, 0.8569372316111383, 0.8374900840504085, 0.8495098728280119, 0.8475137546498668, 0.8509974354378016, 0.8545542968912262, 0.8369359268265817, 0.8881628216627452, 0.8553054247582024, 0.8715475068300871, 0.8608489638331329, 0.7871896522021451, 0.7986180814516614, 0.8679817198115483, 0.8555312604259576, 0.8737131993516944, 0.8570307159808236, 0.86943760267903, 0.8155454038368009, 0.8284627670247386, 0.7440460226630737, 0.8383901711678877, 0.9176876584197461, 0.8867356968591616, 0.8800298236584221, 0.8534696245512979, 0.9166524864925935, 0.8205450625187547, 0.8235830983361883, 0.8610359125511253, 0.8534495672661243, 0.8343550724006359, 0.826657313239454, 0.8327557274202153, 0.8263809690050867, 0.8449533999089178, 0.7403854533869694, 0.8862881836134406, 0.80930312554624, 0.8390349727384677, 0.7812820207595776, 0.8405256568966404, 0.7208619973606759, 0.8237972236612818, 0.8652031422452744, 0.7788070757633151, 0.8795942431527423, 0.8603826742129177, 0.83330392945359, 0.8487413534443429, 0.8085704307615089, 0.8862416492592033, 0.8154708608934949, 0.8949611666064037, 0.8189329260750865, 0.8328395987596068, 0.9158502403398057, 0.8066900361300818, 0.9277331317048729]

thre = 0.874 # 随机定义一个阈值

tp = 0  # 正真
tn = 0  # 真负
fp = 0  # 假正
fn = 0  # 假负

for t4, t5 in zip(y_true, y_test):
    if t4 == 1 and t5 >= thre:
        tp += 1
    elif t4 == 1:
        fn += 1
    elif t4 == 0 and t5 < thre:
        tn += 1
    else:
        fp += 1

data = {
    "真正": tp,
    "真负": tn,
    "假正": fp,
    "假负": fn
}
print("混淆矩阵数据：", data)

p = tp / (tp + fp )  # 精确率，预测为正的样本中有多少是真正的正样本
r = tp / (tp + fn )  # 召回率，样本中的正例有多少被预测正确了
acc = (tp + tn) / (tp + tn + fp + fn)  # 准确率，被分对的样本数除以所有的样本数
f1 = 2 * p * r / (p + r )

beta = 2
#       (1 + β × β) × P × R
# Fβ = ──────────────────────
#        (β × β) × P + R

f2 = (1+beta*beta) * p * r / (beta*beta*p+r)

data2 = {
    "准确率": acc,
    "精确率": p,
    "召回率": r,
    "f1值": f1,
    "f2值": f2,
}

print('通过精确率，召回率计算的结果：', data2)

# auc
auc = roc_auc_score(y_true, y_test)

# 精确率
p = precision_score(y_true, np.array(y_test)>thre)

# 召回率
r = recall_score(y_true, np.array(y_test) > thre)

# acc
acc = accuracy_score(y_true, np.array(y_test) > thre)

f1 = f1_score(y_true, np.array(y_test) > thre)

f2 = fbeta_score(y_true, np.array(y_test) > thre, beta=2)

data3 = {
    "准确率": acc,
    "ROC曲线下面积": auc,
    "f1值": f1,
    "f2值": f2,
    "精确率": p,
    "召回率": r,
}

print('通过sklearn计算的结果：', data3)

y_true = [0, 1, 2, 2, 2]
y_test = [0, 0, 2, 2, 1]
target_names = ['class 0', 'class 1', 'class 2']
print(classification_report(y_true, y_test, target_names=target_names))

# 多分类的计算
# 'micro':通过先计算总体的TP，FN和FP的数量，再计算F1
# 'macro':分布计算每个类别的F1，然后做平均（各类别F1的权重相同）
# 宏平均（macro-average）和微平均（micro-average）
# 在n个二分类混淆矩阵上要综合考察评价指标的时候就会用到宏平均和微平均。宏平均（macro-average）和微平均（micro-average）是衡量文本分类器的指标。
# 宏平均（Macro-averaging），是先对每一个类统计指标值，然后在对所有类求算术平均值。宏平均指标相对微平均指标而言受小类别的影响更大。
# 微平均（Micro-averaging），是对数据集中的每一个实例不分类别进行统计建立全局混淆矩阵，然后计算相应指标。

# 二分类
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score

y_true = [0, 1, 1, 0, 1, 0]
y_pred = [1, 1, 1, 0, 0, 1]

Accuracy = accuracy_score(y_true, y_pred)  # 注意没有average参数
Precision = precision_score(y_true, y_pred, average='binary')
recall = recall_score(y_true, y_pred, average='binary')
f1score = f1_score(y_true, y_pred, average='binary')

# 微平均（Micro-averaging）
y_true = [1, 1, 1, 1, 1, 2, 2, 2, 2, 3, 3, 3, 4, 4]
y_pred = [1, 1, 1, 0, 0, 2, 2, 3, 3, 3, 4, 3, 4, 3]
# 首先计算总TP值(正真)，TN值(真负,预测为不是此类，实际上也不是此类)，这个很好就算，就是数一数上面有多少个类别被正确分类，比如1这个类别有3个分正确，2有2个，3有2个，4有1个，那TP=3+2+2+1=8
# 其次计算总FP值(假正,预测为此类实际上不是此类)，简单的说就是不属于某一个类别的元数被分到这个类别的数量，比如上面不属于4类的元素被分到4的有1个
# 如果还比较迷糊，我们在计算时候可以把4保留，其他全改成0，就可以更加清楚地看出4类别下面的FP数量了，其实这个原理就是 One-vs-All (OvA)，把4看成正类，其他看出负类

# 同理我们可以再计算FN(假负,预测为不是此类实际上是此类)的数量

#   	0类	1类	2类	3类	4类	总数
# TP	0	3	2	2	1	8
# TN	12	9	10	8	11	50
# FP	2	0	0	3	1	6
# FN	0	2	2	1	1	6

# 所以micro的 精确度P 为 TP/(TP+FP)=8/(8+6)=0.5714    召回率R TP/(TP+FN)=8/(8+6)=0.571   所以F1-micro的值为：0.6153

# 准确率
acc = 8/14
Out[44]: 0.5714285714285714
accuracy_score(y_true, y_pred)
Out[31]: 0.5714285714285714
# 微平均精确率
precision_score(y_true, y_pred, average='micro')
Out[29]: 0.5714285714285714
# 微平均召回率
recall_score(y_true, y_pred, average='micro')
Out[32]: 0.5714285714285714
# 微平均f1
f1_score(y_true, y_pred, average='micro')
Out[33]: 0.5714285714285714
# f1 = 2 * p * r / (p + r )
2 * (8/14) * (8/14) / ((8/14) + (8/14) )
Out[53]: 0.5714285714285714



# 宏平均精确率, “宏查准率”（macro-P）
precision_score(y_true, y_pred, average='macro')
Out[30]: 0.58
# 每一类的精确率再求平均
# p = tp / (tp + fp )  # 精确率，预测为正的样本中有多少是真正的正样本
sum([0/2, 3/3, 2/2, 2/5, 1/2])/5
Out[51]: 0.58

# 宏平均召回率, “宏查全率”（macro-R）
recall_score(y_true, y_pred, average='macro')
Out[34]: 0.4533333333333333
# r = tp / (tp + fn )  # 召回率，样本中的正例有多少被预测正确了
sum([0/0.00001, 3/5, 2/4, 2/3, 1/2])/5
Out[54]: 0.4533333333333333

# 宏平均f1, “宏F1”（macro-F1）
f1_score(y_true, y_pred, average='macro')
Out[35]: 0.4833333333333333
# f1 = 2 * p * r / (p + r )
p_list = [0/2, 3/3, 2/2, 2/5, 1/2]
r_list = [0/0.00001, 3/5, 2/4, 2/3, 1/2]
sum([2 * p * r / (p + r + 0.00001) for p, r in zip(p_list, r_list)])/5
Out[57]: 0.4833295694750185

# 如何找到最适合的阈值及计算f1-score
# 计算方法也非常简单粗暴，直接把可能阈值全部计算一遍，得到一个 F1-score 数组，然后找到最大值以及对一个的阈值即可。
precisions, recalls, thresholds = precision_recall_curve(y_true,y_test)

# 拿到最优结果以及索引
f1_scores = (2 * precisions * recalls) / (precisions + recalls)  # 计算全部f1-score
best_f1_score = np.max(f1_scores[np.isfinite(f1_scores)])
best_f1_score_index = np.argmax(f1_scores[np.isfinite(f1_scores)])

# 阈值
print("最佳阈值及其F1-score: {}, {}".format(thresholds[best_f1_score_index], best_f1_score))

def d_prime(hits, false_alarms, miss, correct_rejection):
    """
    d prime score 指标计算
    Calculate the sensitivity index d'.
    Parameters
    ----------
    hits : float
        命中数量The number of hits when detecting a signal.
    false_alarms : float
        误报数量；The number of false alarms.
    miss : int
        漏报数量，即样本中为正,预测为负的样本数量；
    correct_rejection : int
        正确拒绝数量，即样本为负，预测为负的样本数量；
    Returns
    -------
    d : float
        其中：hit rate(H) = HIT/(HIT+MISS)
        false alarm rate(F) = FALSE ALARM/(FALSE ALARM + CORRECT REJECTION)
        The calculated d' value, z(hit_rate) - z(false_alarm_rate).
    Example
    -------
    >>> d_prime(20, 10, 5, 15)
    1.094968336708714
    """
    if hits == 0 or miss == 0:
        hit_rate = (hits+0.5) / (hits + miss+1)
    else:
        hit_rate = hits /(hits+miss)
    if false_alarms == 0 or correct_rejection == 0:
        fa_rate = (false_alarms+0.5) / (false_alarms+correct_rejection+1)
    else:
        fa_rate = false_alarms / (false_alarms + correct_rejection)
    d_prime_score = stats.norm.ppf(hit_rate) - stats.norm.ppf(fa_rate)
    return d_prime_score

def main():
    pass


if __name__ == '__main__':
    main()