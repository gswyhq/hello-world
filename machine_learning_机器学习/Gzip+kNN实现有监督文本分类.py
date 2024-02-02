#!/usr/bin/env python
# coding=utf-8

# Gzip+ kNN文本分类竟然击败Transformers：无需预训练、14行代码实现
# 论文地址：https://aclanthology.org/2023.findings-acl.426.pdf
# 代码地址：https://github.com/bazingagin/npc_gzip

import os
import time
import gzip
import numpy as np
import pandas as pd
from tqdm import tqdm
USERNAME = os.getenv("USERNAME")

# 语料来源：https://github.com/zhuanxuhit/nd101/tree/master/1.Intro_to_Deep_Learning/3.How_to_Do_Sentiment_Analysis/data
neg = pd.read_excel(rf'D:\Users\{USERNAME}/data/nd101/neg.xls', header=None, dtype='str')
pos = pd.read_excel(rf'D:\Users\{USERNAME}/data/nd101/pos.xls', header=None, dtype='str')

data = []

for d in neg[0]:
    if not d or pd.isna(d):continue
    data.append((d, 0))

for d in pos[0]:
    if not d or pd.isna(d): continue
    data.append((d, 1))
data = list(set(data))

# 按照9:1的比例划分训练集和验证集
random_order = list(range(len(data)))
np.random.shuffle(random_order)
train_data = [data[j] for i, j in enumerate(random_order) if i % 10 != 0]
valid_data = [data[j] for i, j in enumerate(random_order) if i % 10 == 0]

# len(train_data)
# Out[4]: 18994
# len(valid_data)
# Out[5]: 2111


def text_classification(training_set, test_set, k=101, Cx2_dict = None):
    training_set = np.array(training_set)
    true_count = 0
    for (x1 , label) in tqdm(test_set):
        Cx1 = len( gzip.compress ( x1.encode () ) )
        distance_from_x1 = []
        for (x2 , _ ) in training_set :
            if Cx2_dict is None:
                Cx2 = len(gzip.compress(x2.encode() ))
            else:
                Cx2 = Cx2_dict[x2]
            x1x2 = " ". join ([x1 , x2 ])
            Cx1x2 = len( gzip.compress ( x1x2.encode() ))
            ncd = ( Cx1x2 - min ( Cx1 , Cx2 ) ) / max (Cx1 , Cx2 )
            distance_from_x1.append ( ncd )
        sorted_idx = np.argsort (np.array(distance_from_x1 ))
        top_k_class = training_set[sorted_idx[:k] , 1].tolist()
        predict_class = max(set(top_k_class ) , key = top_k_class.count )
        if int(predict_class) == int(label):
            true_count += 1
    print("整体正确率：{:.4f}({}/{})".format(true_count/len(test_set), true_count, len(test_set)))

start_time = time.time()
Cx2_dict = {x2: len(gzip.compress(x2.encode() )) for x2, _ in train_data}
text_classification(train_data, valid_data, Cx2_dict=Cx2_dict)
print('整体耗时：{}s'.format(time.time() - start_time))

# gzip+knn效果, k=5
#  17%|█▋        | 290/1677 [02:46<14:23,  1.61it/s]
# 100%|██████████| 1677/1677 [28:01<00:00,  1.00s/it]
# 整体正确率：0.8157(1368/1677)
# 整体耗时：1681.8205902576447s

# gzip+knn效果，k=101时
# 100%|██████████| 1677/1677 [27:02<00:00,  1.03it/s]
# 整体正确率：0.8223(1379/1677)
# 整体耗时：1622.767545223236s

# 该数据集用bert进行微调训练效果：
# 792/792 [==============================] - 8767s 11s/step - loss: 0.2194 - acc: 0.9134 - val_loss: 0.1507 - val_acc: 0.9413
# 总耗时：8787.115825176239s

def main():
    pass


if __name__ == "__main__":
    main()
