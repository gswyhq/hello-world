#!/usr/bin/python3
# coding: utf-8


import pickle
import gensim
from gensim.models import word2vec
from collections import OrderedDict

MODEL_FILE = '/home/gswyhq/data/model/word2vec/news_12g_baidubaike_20g_novel_90g_embedding_64.bin'
# 模型来源：  https://weibo.com/p/23041816d74e01f0102x77v?luicode=20000061&lfid=4098518198740187&featurecode=newtitle

def load_model(model_file=MODEL_FILE):
    """加载向量模型"""
    logger.info("加载向量模型: {}".format(model_file))
    if model_file.endswith('.pkl'):
        with open(model_file, "rb")as f:
            w2v_model = pickle.load(f, encoding='iso-8859-1')  # 此处耗内存 60.8 MiB
    elif model_file.endswith('.bin'):
        # 注意：不可能继续训练从C格式加载的矢量，因为隐藏的权重，词汇频率和二叉树丢失::
        w2v_model = gensim.models.KeyedVectors.load_word2vec_format(model_file, binary=True)
    elif model_file.endswith('.model'):
        # 此种方式加载成功的模型，可用于增量训练
        w2v_model = word2vec.Word2Vec.load(model_file)
    else:
        raise ValueError("模型文件路径有误：{}".format(model_file))
    return w2v_model

def model2pkl(w2v_model, model_file='./model.pkl'):
    """
    将模型保存为pkl格式的模型文件
    :param pickle: 模型的保存路径
    :return: 
    """
    w2v_char_model = OrderedDict()
    for key in w2v_model.index2word:
        w2v_char_model[key] = w2v_model[key]
    pickle.dump(w2v_char_model, open(model_file, "wb"))

w2v_model = load_model(model_file=MODEL_FILE)  # 字向量模型

# w2v_model.save('./model/wiki_ida.pkl') # 保存到文件wiki_ida.pkl中
# model = gensim.models.ldamodel.LdaModel.load('./model/wiki_ida.pkl') # 从文件wiki_ida.pkl中读出结果数据

# 得到与一个词最相关的若干词及相似程度
w2v_model.most_similar(u'重疾险')
# Out[20]:
# [('医疗险', 0.908942699432373),
#  ('定期寿险', 0.8982336521148682),
#  ('意外险', 0.8067277669906616),
#  ('主险', 0.7949002981185913),
#  ('疾病保险', 0.7920745015144348),
#  ('附加险', 0.7812231779098511),
#  ('航空意外险', 0.7791354060173035),
#  ('航意险', 0.7709769010543823),
#  ('分红险', 0.7685513496398926),
#  ('种轻症', 0.766133189201355)]

# 得到两组词的相似度
w2v_model.n_similarity(['重疾险', '如何', '赔付'], ['重疾险', '的', '赔付', '方式'])
# Out[21]: 0.43882406683816255

# 得到一组词中最无关的词
list4 = [u'汽车', u'火车', u'飞机', u'北京']
w2v_model.doesnt_match(list4)
# Out[22]: '北京'

# 计算两个词的相似度
w2v_model.similarity('重疾险', '医疗险')
# Out[23]: 0.9089426518578948

# 计算关联词：
#其意义是计算一个词d（或者词表），使得该词的向量v(d)与v(a="女人")-v(c="男人")+v(b="国王")最近
w2v_model.most_similar(positive=["女人","国王"],negative=["男人"],topn=1)
# Out[31]: [('王后', 0.85798180103302)]

w2v_model.most_similar(["男孩"], topn=3)
# Out[32]:
# [('男孩儿', 0.9407316446304321),
#  ('女孩', 0.9226171970367432),
#  ('小女孩', 0.9133328199386597)]
w2v_model.most_similar("男孩", topn=3)
# Out[33]:
# [('男孩儿', 0.9407316446304321),
#  ('女孩', 0.9226171970367432),
#  ('小女孩', 0.9133328199386597)]

w2v_model['男孩']
# Out[34]:
# array([ -1.63780117,  -0.92047566, -11.39441013,   4.34599972,
#          5.85181952,   3.45414782,  -6.57227373,   1.05580115,
#          1.0951767 ,  -2.70963812,  -0.41607726,  -6.6814599 ,
#         -1.12566876,   1.85306084,   4.45130253,  10.25532436,
#          0.6096614 ,  -2.29226732,   0.6499418 ,   0.98136711,
#          0.83426827,   5.39834213,  -8.71372509,  -0.69955271,
#          2.79552531,  -3.65006995,  -0.47023383,  -0.60166794,
#          3.63869953,   1.47523224,  -1.96910024,   3.97203565,
#         -0.13121423,  -2.11341357,  -4.42379284,   5.00740433,
#          2.1927557 ,  -0.30364683,   2.15308356,  -2.12827563,
#          2.42039704,  -3.7469461 ,   0.56527215,  -0.79935616,
#          3.85497379,  -7.36658144,  -2.91070414,   3.31585169,
#         -1.56717932,   0.90537137,  -1.68752909,   2.90144992,
#          4.26190615,   3.69292283,   6.98834705,  -3.93023849,
#          0.2768223 ,  -4.20171499,  -4.66988516,   3.04182625,
#         -2.94271159,   1.98612726,  -0.59248883,   1.53097141], dtype=float32)
w2v_model['男孩'].shape
# Out[35]: (64,)



def main():
    pass


if __name__ == '__main__':
    main()

