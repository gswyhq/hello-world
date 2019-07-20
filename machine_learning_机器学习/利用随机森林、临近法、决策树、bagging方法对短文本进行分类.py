#!/usr/bin/python3
# coding: utf-8

import os
import pandas as pd
import numpy as np
import  matplotlib.pyplot as  plt
import pickle as pkl
import time
from sklearn.pipeline import make_pipeline
from sklearn.svm import SVC
from sklearn import  metrics
from sklearn.ensemble import RandomForestClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import BaggingClassifier
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.feature_extraction.text import TfidfTransformer

from sklearn.feature_extraction.text import  CountVectorizer
import jieba
from sklearn.model_selection import  train_test_split

tfidf=TfidfTransformer()

def chinese_word_cut(mytext):
    return " ".join(jieba.lcut(mytext))

def read_data(excel_name='/home/gswyhq/Downloads/意图语料V5.0.4（0717）.xlsx'):
    df = pd.read_excel(excel_name)[['意图', '问题']]
    df.head()

    intent_index_dict = {intent: index for index, intent in enumerate(list(set(df.意图)))}

    id_intent_dict = {v: k for k, v in intent_index_dict.items()}

    df['label'] = df['意图'].apply(lambda x:intent_index_dict.get(x))

    df=df.drop_duplicates() ## 去掉重复的评论
    df=df.dropna()


    X=pd.concat([df[['问题']],df[['问题']],df[['问题']]])
    y=pd.concat([df.label,df.label,df.label])

    X['cut_comment']=X["问题"].apply(chinese_word_cut)
    X['cut_comment'].head()


    X_train,X_test,y_train,y_test= train_test_split(X,y,random_state=42,test_size=0.25)

    return id_intent_dict, X_train, X_test, y_train, y_test

def get_custom_stopwords(stop_words_file):
    with open(stop_words_file,encoding="utf-8") as f:
        custom_stopwords_list=[i.strip() for i in f.readlines()]
    return custom_stopwords_list

def train(id_intent_dict, X_train, X_test, y_train, y_test, stopwords, save_model_path='./model', mode='随机森林'):
    vect=CountVectorizer()

    vect.fit_transform(X_train["cut_comment"])

    # vect.fit_transform(X_train["cut_comment"]).toarray().shape
    vect = CountVectorizer(token_pattern=u'(?u)\\b[^\\d\\W]\\w+\\b',stop_words=frozenset(stopwords)) # 去除停用词
    pd.DataFrame(vect.fit_transform(X_train['cut_comment']).toarray(), columns=vect.get_feature_names()).head()
    if mode == '随机森林':
        forest=RandomForestClassifier(criterion='entropy',random_state=1,n_jobs=2)
    elif mode == '支持向量机':
        # 耗时较久
        svc_cl = SVC()  # 实例化
        forest = svc_cl
    elif mode == '临近法':
        knn = KNeighborsClassifier(n_neighbors=5, p=2, metric='minkowski')
        forest = knn
    elif mode == '决策树':
        tree = DecisionTreeClassifier(criterion='entropy', random_state=1)
        forest = tree
    elif mode == 'bagging方法':
        tree = DecisionTreeClassifier(criterion='entropy', random_state=1)
        bag = BaggingClassifier(base_estimator=tree,
                            n_estimators=10,
                            max_samples=1.0,
                            max_features=1.0,
                            bootstrap=True,
                            bootstrap_features=False,
                            n_jobs=1, random_state=1)
        forest = bag

    elif mode == "GradientBoosting方法":
        # 耗时较久
        grd = GradientBoostingClassifier(learning_rate=0.18, max_depth=10, n_estimators=240, random_state=42,
                                         max_features='sqrt', subsample=0.9,
                                         min_impurity_decrease=0.01)
        forest = grd
    else:
        raise ValueError('不支持的训练类型:{}'.format(mode))

    pipe=make_pipeline(vect,forest)
    pipe.fit(X_train.cut_comment, y_train)
    y_pred = pipe.predict(X_test.cut_comment)

    # print(X_test.cut_comment[:10])

    acc = metrics.accuracy_score(y_test,y_pred)
    save_model_file = os.path.join(save_model_path, "{}_model.pkl".format(mode))
    pkl.dump({"id2intent": id_intent_dict, "model": pipe}, open(save_model_file, "wb"), protocol=2)
    print(mode, '准确率：', acc)

def test(question, save_model_path='./model', mode='随机森林'):

    save_model_file = os.path.join(save_model_path, "{}_model.pkl".format(mode))
    model_data = pkl.load(open(save_model_file, "rb"),
                                    encoding="iso-8859-1")
    model = model_data['model']
    id2intent = model_data['id2intent']
    pres = model.predict([chinese_word_cut(question)])
    ret = [id2intent.get(p, '') for p in pres]
    print(mode, '测试结果', ret)
    return ret

def main():
    id_intent_dict, X_train, X_test, y_train, y_test = read_data(excel_name='/home/gswyhq/Downloads/意图语料V5.0.4（0717）.xlsx')

    stopwords = get_custom_stopwords(stop_words_file="stopwords.txt")
    model_names = ['随机森林', '临近法', '决策树', 'bagging方法']
    # 'GradientBoosting方法', '支持向量机',耗时太久，舍弃
    for mode in model_names:
        train(id_intent_dict, X_train, X_test, y_train, y_test, stopwords,
              save_model_path='./model', mode=mode)

        ret = test('给我推荐一款产品可以吗', save_model_path='./model', mode=mode)

if __name__ == '__main__':
    main()

# 更多方法参考： https://github.com/point6013/meituan_reviews_analysis
# xgboost 方法
# from xgboost import XGBClassifier
# # sklearn API 类似于导入的从skearn中导入某个算法，然后再进行实例化即可，初始化算法的时候可以修改默认参数
# from xgboost import plot_importance
# x_train_vect=vect.fit_transform(X_train["cut_comment"])
# x_test_vect= vect.transform(X_test["cut_comment"])
# clf = XGBClassifier(
# silent=1 ,#设置成1则没有运行信息输出，最好是设置为0.是否在运行升级时打印消息。
# # #nthread=4,# cpu 线程数 默认最大
# learning_rate= 0.20, # 学习率
# min_child_weight=0.5,
# # # 这个参数默认是 1，是每个叶子里面 h 的和至少是多少，对正负样本不均衡时的 0-1 分类而言
# # #，假设 h 在 0.01 附近，min_child_weight 为 1 意味着叶子节点中最少需要包含 100 个样本。
# # #这个参数非常影响结果，控制叶子节点中二阶导的和的最小值，该参数值越小，越容易 overfitting。
# gamma=0.1,  # 树的叶子节点上作进一步分区所需的最小损失减少,越大越保守，一般0.1、0.2这样子。
# subsample=0.7, # 随机采样训练样本 训练实例的子采样比
# max_depth=15,
# max_delta_step=0,#最大增量步长，我们允许每个树的权重估计。
# colsample_bylevel=0.7, # Subsample ratio of columns for each split, in each level.
# colsample_bytree=0.6, # 生成树时进行的列采样
# reg_lambda=0.04,  # 控制模型复杂度的权重值的L2正则化项参数，参数越大，模型越保守
# reg_alpha=0.05, # L1 正则项参数，参数越大，模型越保守
# ### 正则化是在梯度提升树种没有的，这是xgboost与GB方法的区别之一。
# scale_pos_weight=1, #如果取值大于0的话，在类别样本不平衡的情况下有助于快速收敛。平衡正负权重=sum(负类样本)/sum(正类样本)
# # objective= 'reg:logistic', #多分类的问题 指定学习任务和相应的学习目标
# objective='binary:logistic' ,
# # #num_class=10, # 类别数，多分类与 multisoftmax 并用
# n_estimators=900, #树的个数
# random_state=42
# # #eval_metric= 'auc'
# )
# # xgb_model=XGBClassifier()
# # clf = GridSearchCV(xgb_model, {'max_depth': [4, 6,8,10],
# #                                'n_estimators': [50, 100, 200,400,600],
# #                                'gamma':[0.1,0.12,0.15,0.18,0.2],
# #                               'subsample':[0.5,0.6,0.7,0.8,0.9,1.0],
# #                               'learning_rate':[0.1,0.15,0.2],
# #                               'reg_lambda':[0.2,0.4,0.6,0.8]}, verbose=1,
# #                               n_jobs=2)
# clf.fit(x_train_vect,y_train,eval_metric=['auc','error'])
# # clf.fit(x_train_vect,y_train,eval_metric=['auc','error'])
# # print(clf.best_score_)
# # print(clf.best_params_)
#
#
# # #获取验证集合结果
# # # evals_result = clf.evals_result()
# # y_true, y_pred = y_test, clf.predict(x_test_vect)
# # print("Accuracy : %d" % metrics.accuracy_score(y_true, y_pred))
# XGBClassifier(base_score=0.5, booster='gbtree', colsample_bylevel=0.7,
#        colsample_bytree=0.6, gamma=0.1, learning_rate=0.2,
#        max_delta_step=0, max_depth=15, min_child_weight=0.5, missing=None,
#        n_estimators=900, n_jobs=1, nthread=None,
#        objective='binary:logistic', random_state=42, reg_alpha=0.05,
#        reg_lambda=0.04, scale_pos_weight=1, seed=None, silent=1,
#        subsample=0.7)
# y_pred=clf.predict(x_test_vect)
# metrics.accuracy_score(y_test,y_pred)
# 0.94777070063694269
# metrics.confusion_matrix(y_test,y_pred)
# array([[260,  29],
#        [ 12, 484]], dtype=int64)