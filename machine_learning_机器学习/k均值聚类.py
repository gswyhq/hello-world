#!/usr/bin/python3
# coding: utf-8

import pickle
from sklearn.cluster import KMeans
import numpy as np

PATH_character = "/home/gswyhq/wmd/model/wx_vector_char.pkl" # 字向量文件


def read_huan(file_name):

    questions=[]
    ids=[]
    with open(file_name)as f:
        datas = f.readlines()
    for i, t in enumerate(datas):
        if not t:
            continue
        t = t.strip()
        if not t:
            continue
        questions.append(t)
        ids.append(i)
    return questions, ids

# 字向量文本向量化
def character_vector(texts):
    
    with open(PATH_character, "rb")as f:
        model_character = pickle.load(f, encoding='iso-8859-1')  # 此处耗内存 60.8 MiB

    train_vectors = []
    for text in texts:
        text_vectors = []
        for char in text.strip():
            try:
                vec = model_character[char]
                text_vectors.append(vec)
            except KeyError:
                pass
        data_matrix = np.mat(text_vectors)
        result = np.mean(data_matrix, axis=0)
        train_vectors.append((result.getA()).tolist()[0])
    return train_vectors

def cluster_data_huan(texts, y_pred, ids):
    ds = {}
    for txt, y_p, _id in zip(texts, y_pred, ids):
        ds.setdefault(y_p, [])
        ds[y_p].append(txt)
    return ds

def cluster(texts, ids, n_clusters=100):
    Y = character_vector(texts)
    X = []
    new_texts = []
    new_ids = []
    for num, txt, item in zip(ids, texts, Y):
        if len(item)!=256:
            print ("`{}`行，`{}`的向量长度：{}， 不合要求。".format(num, txt,len(item)))
        else:
            X.append(item)
            new_ids.append(num)
            new_texts.append(txt)
    kmeans = KMeans(n_clusters=n_clusters).fit(X)
    y_pred = kmeans.labels_

    result = cluster_data_huan(new_texts,y_pred,new_ids) # 文本，预测，原始

    return result

def write_result(result_dict, save_file):

    with open(save_file, 'w', encoding='utf-8')as f:
        for label, questions in sorted(result_dict.items(), key=lambda x: x[0]):
            for q in questions:
                f.write("{}\t{}\n".format(label, q))
                
    print('聚类结果保存在：{}'.format(save_file))

def main():
    file_name = '/home/gswyhq/github_projects/text_clustering/data/test_data3.txt' # 每行为一个待聚类的句子；
    save_file = '/home/gswyhq/Downloads/cluster_result.txt'  # 类别id, 该类别的句子
    texts,ids = read_huan(file_name)

    result_dict = cluster(texts, ids, n_clusters=100)
    write_result(result_dict, save_file)

if __name__ == '__main__':
    main()


