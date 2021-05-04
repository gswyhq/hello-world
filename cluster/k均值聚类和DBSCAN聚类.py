#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import sklearn
import time
import json
import numpy as np
# import matplotlib.pyplot as plt
from sklearn.cluster import MiniBatchKMeans, KMeans
from sklearn.cluster import DBSCAN
from sklearn.metrics.pairwise import pairwise_distances_argmin
from sklearn.datasets.samples_generator import make_blobs
from sklearn.externals import joblib
from gensim.models import KeyedVectors
from gensim.models import FastText

def mini_batch_kmeans_cluster(datas, n_clusters):
    mini_km_cluster = MiniBatchKMeans(n_clusters=n_clusters, init='k-means++', max_iter=100,
                 batch_size=n_clusters, verbose=0, compute_labels=True,
                 random_state=None, tol=0.0, max_no_improvement=10,
                 init_size=None, n_init=3, reassignment_ratio=0.01)

    result = mini_km_cluster.fit_predict(datas)
    clusters = result.tolist()
    result_file = '/home/gswyhq/cluster/cluster_result_{}.pkl'.format(n_clusters)

    # 注释语句用来存储你的模型
    joblib.dump(mini_km_cluster,  result_file)
    # km = joblib.load('doc_cluster.pkl')
    # clusters = km.labels_.tolist()
    return clusters

def dbscan(datas, eps=0.3):
    # db = DBSCAN(eps=eps, min_samples=10).fit(datas)
    # core_samples_mask = np.zeros_like(db.labels_, dtype=bool)
    # core_samples_mask[db.core_sample_indices_] = True
    db = DBSCAN(eps=eps, min_samples=10).fit_predict(datas)
    clusters = db.tolist()
    print("eps: {}, len(clusters): {}".format(eps, len(set(clusters))))
    result_file = '/home/gswyhq/cluster/cluster_dbscan_result_{}_{}.pkl'.format(eps, len(set(clusters)))

    # 注释语句用来存储你的模型
    joblib.dump(db,  result_file)
    # km = joblib.load('doc_cluster.pkl')
    # clusters = km.labels_.tolist()
    return clusters

def main():
    # vec_file = r'/home/gswyhq/new_words/test_fasttext.txt'
    # wv_from_text = KeyedVectors.load_word2vec_format(vec_file, binary=False)

    model = FastText.load('/home/gswyhq/new_words/word_vec.model')
    # model.wv.get_vector(model.wv.index2word[0])
    n_clusters = 300
    # for n_clusters in [20, 50, 100, 200, 300, 500, 1000]:
    for eps in [0.3, 0.5, 0.8]: # eps: 0.3, len(clusters): 15
        datas = np.array([model.wv.get_vector(word) for word in model.wv.index2word])
        # clusters = mini_batch_kmeans_cluster(datas, n_clusters)
        clusters = dbscan(datas, eps=eps)
        n_clusters = len(set(clusters))
        data = {}
        for word, label in zip(model.wv.index2word, clusters):
            data.setdefault(label, [])
            data[label].append(word)
        with open('/home/gswyhq/cluster/cluster_dbscan_result_{}.txt'.format(n_clusters), 'w', encoding='utf-8')as f:
            for label, word_list in sorted(data.items(), key=lambda x: x[0]):
                f.write('-'*100+'\n')
                f.write('label: {}\n'.format(label))
                f.write('\t'.join(word_list) + '\n')
                f.write('\n')

        print(n_clusters)
    print('ok')

if __name__ == '__main__':
    main()

# nohup python -u /home/gswyhq/word_clustering.py > log/word_clustering.log &
