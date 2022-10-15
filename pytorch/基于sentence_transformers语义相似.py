#!/usr/bin/env python3
# -*- coding: utf-8 -*-

# 来源：https://github.com/UKPLab/sentence-transformers/blob/master/examples/training/sts/training_stsbenchmark.py
# https://www.sbert.net/docs/training/overview.html

import os
import pandas as pd
import numpy as np
import json, pickle
from torch.utils.data import DataLoader
import math
from sentence_transformers import SentenceTransformer,  LoggingHandler, losses, models, util
from sentence_transformers.evaluation import EmbeddingSimilarityEvaluator
from sentence_transformers.readers import InputExample
from datetime import datetime

from sklearn.metrics import accuracy_score, roc_auc_score, roc_curve, confusion_matrix, average_precision_score, precision_recall_curve

task_name = ''
def load_data(filename):
    """加载数据（带标签）
    单条格式：(文本1, 文本2, 标签)
    """
    D = []
    df = pd.read_csv(filename, sep='\t', encoding='utf-8')
    for text_a, text_b, label in df[ ['text_a', 'text_b', 'label']].values:
        D.append((text_a, text_b, float(label)))
    return D

# 数据来源：https://github.com/xiaohai-AI/lcqmc_data
USERNAME = os.getenv("USERNAME")
data_path = rf'D:\Users\{USERNAME}\github_project\lcqmc_data'
datasets = {
    '%s-%s' % (task_name, f):
    load_data('%s/%s.txt' % (data_path, f))
    for f in ['train', 'dev', 'test']
}


train_data = datasets['-train']
dev_data = datasets['-dev']
test_data = datasets['-test']
# train_data = train_data[:100]
# dev_data = dev_data[:10]
# test_data = test_data[:10]

# 模型来源：# https://huggingface.co/uer/chinese_roberta_L-4_H-128/tree/main
word_embedding_model = models.Transformer(rf'D:\Users\{USERNAME}\data\chinese_roberta_L-4_H-128')

# Apply mean pooling to get one fixed sized sentence vector
pooling_model = models.Pooling(word_embedding_model.get_word_embedding_dimension(),
                               pooling_mode_mean_tokens=True,
                               pooling_mode_cls_token=False,
                               pooling_mode_max_tokens=False)

model = SentenceTransformer(modules=[word_embedding_model, pooling_model])

def data_samples(train_data):
    samples = []
    for sentence1, sentence2, score in train_data:
        score = float(score)
        inp_example = InputExample(texts=[sentence1, sentence2], label=score)
        samples.append(inp_example)
    return samples

train_samples = data_samples(train_data)
dev_samples = data_samples(dev_data)
test_samples = data_samples(test_data)

train_batch_size=8
num_epochs = 5
model_save_path = rf"D:\Users\{USERNAME}\Downloads\test/" + datetime.now().strftime("%Y-%m-%d_%H-%M-%S")

train_dataloader = DataLoader(train_samples, shuffle=True, batch_size=train_batch_size)
train_loss = losses.CosineSimilarityLoss(model=model)

evaluator = EmbeddingSimilarityEvaluator.from_input_examples(dev_samples, name='sts-dev')

# Configure the training. We skip evaluation in this example
warmup_steps = math.ceil(len(train_dataloader) * num_epochs  * 0.1) #10% of train data for warm-up

# Train the model
model.fit(train_objectives=[(train_dataloader, train_loss)],
          evaluator=evaluator,
          epochs=num_epochs,
          evaluation_steps=1000,
          warmup_steps=warmup_steps,
          output_path=model_save_path)


model = SentenceTransformer(model_save_path)
test_evaluator = EmbeddingSimilarityEvaluator.from_input_examples(test_samples, name='sts-test')
test_evaluator(model, output_path=model_save_path)

model.evaluate(test_evaluator) # -> 0.7314628278461454
embeddings1 = model.encode([t[0] for t in test_data])
embeddings2 = model.encode([t[1] for t in test_data])
# 计算特征向量间的余弦相似度
# embeddings = model.encode(sentences, convert_to_tensor=True)
# cosine_scores = np.array([util.cos_sim(embeddings1[idx], embeddings2[idx])[0, 0].numpy() for idx in range(len(embeddings1))])
# thr = 0.5
# ret_mat = confusion_matrix([t[-1] for t in test_data], [1 if t>=thr else 0 for t in cosine_scores])
# ret_mat
# Out[29]:
# array([[3388, 2862],
#        [ 226, 6024]], dtype=int64)
# print("准确率：{}".format((ret_mat[0,0]+ret_mat[1,1])/ret_mat.sum()))
# 准确率：0.75296

# 当阈值设置为0.85时，准确率有83%
# thr = 0.85
# ret_mat = confusion_matrix([t[-1] for t in test_data], [1 if t>=thr else 0 for t in cosine_scores])
# ret_mat, (ret_mat[0,0]+ret_mat[1,1])/ret_mat.sum()
# Out[39]:
# (array([[5293,  957],
#         [1047, 5203]], dtype=int64),
#  0.83968)

def main():
    pass

if __name__ == '__main__':
    main()
