#!/usr/bin/python3
# coding: utf-8

import xgboost as xgb
# 读取数据
# 数据来源：https://github.com/tqchen/xgboost/blob/master/demo/data/agaricus.txt.train
dtrain = xgb.DMatrix('demo/data/agaricus.txt.train')
dtest = xgb.DMatrix('demo/data/agaricus.txt.test')
# 通过 map 指定参数
param = {'max_depth':2, 'eta':1, 'silent':1, 'objective':'binary:logistic' }
num_round = 2
bst = xgb.train(param, dtrain, num_round)
# 预测
preds = bst.predict(dtest)

def main():
    pass


if __name__ == '__main__':
    main()

# 更多使用示例：https://github.com/dmlc/xgboost/tree/master/demo/guide-python

