# Faiss的全称是Facebook AI Similarity Search。
# 这是一个开源库，针对高维空间中的海量数据，提供了高效且可靠的检索方法。
# Faiss的主要功能就是相似度搜索！

# 安装：
# pip install faiss-cpu -c pytorch
# CPU-only version
# $ conda install -c pytorch faiss-cpu
# GPU(+CPU) version
# $ conda install -c pytorch faiss-gpu
# or for a specific CUDA version
# $ conda install -c pytorch faiss-gpu cudatoolkit=10.2 # for CUDA 10.2

# 对于一个检索任务，我们的操作流程一定分为三步：训练、构建数据库、查询。
# faiss中由多种类型的索引，我们可以是呀最简单的索引类型：indexFlatL2，这就是暴力检索L2距离（欧式距离）。
# 不管建立什么类型的索引，我们都必须先知道向量的维度。另外，对于大部分索引类型而言，在建立的时候都包含了训练阶段，但是L2这个索引可以跳过。当索引被建立 和训练之后，我能就可以调用add，search着两种方法。
# 精确搜索：faiss.indexFlatL2(欧式距离) faiss.indexFlatIP(内积)
# 在精确搜索的时候，选择上述两种索引类型，遍历计算索引向量，不需要做训练操作。

# 索引Index选择的原则
# 如果要精确的结果：IndexFlatL2
# 如果数据量低于1百万：用k-means聚类向量
# 如果考虑内存：一系列方法

# 准备数据
import faiss
import numpy as np
d = 64                           # dimension
nb = 100000                      # database size
nq = 10000                       # nb of queries
np.random.seed(1234)             # make reproducible
xb = np.random.random((nb, d)).astype('float32') # 训练数据
xb[:, 0] += np.arange(nb) / 1000.
xq = np.random.random((nq, d)).astype('float32') # 查询数据
xq[:, 0] += np.arange(nq) / 1000.

# 创建索引(Index)
# faiss创建索引对向量预处理，提高查询效率。
# faiss提供多种索引方法，这里选择最简单的暴力检索L2距离的索引：IndexFlatL2。
# 创建索引时必须指定向量的维度d。大部分索引需要训练的步骤。IndexFlatL2跳过这一步。
# 当索引创建好并训练(如果需要)之后，我们就可以执行add和search方法了。
# add方法一般添加训练时的样本
# search就是寻找相似相似向量了
# 一些索引可以保存整型的ID，每个向量可以指定一个ID，当查询相似向量时，会返回相似向量的ID及相似度(或距离)。如果不指定，将按照添加的顺序从0开始累加。其中IndexFlatL2不支持指定ID。

index = faiss.IndexFlatL2(d)   # build the index
print(index.is_trained)
index.add(xb)                  # add vectors to the index 训练数据
print(index.ntotal) # 看索引的总数量 按行来

# 查找相似向量
# 我们有了包含向量的索引后，就可以传入搜索向量查找相似向量了。
# D表示与相似向量的距离(distance)，维度，I表示相似用户的ID。
k = 4                          # we want to see 4 nearest neighbors
D, I = index.search(xq, k)     # actual search
print(I[:5])                   # neighbors of the 5 first queries-对应ID
print(D[-5:])                  # neighbors of the 5 last queries-对应距离

# 加速搜索
# 如果需要存储的向量太多，通过暴力搜索索引IndexFlatL2速度很慢
# 加速搜索的方法为IndexIVFFlat，倒排文件。原理是使用K-means建立聚类中心，然后通过查询最近的聚类中心，最后比较聚类中的所有向量得到相似的向量。
# 创建IndexIVFFlat时需要指定一个其他的索引作为量化器(quantizer)来计算距离或相似度。
# 在add方法之前需要先训练
# IndexIVFFlat的参数为：
# faiss.METRIC_L2: faiss定义了两种衡量相似度的方法(metrics)，分别为faiss.METRIC_L2、faiss.METRIC_INNER_PRODUCT。一个是欧式距离，一个是向量内积【等价于cosine】。
# nlist：聚类中心的个数
# k：查找最相似的k个向量
# index.nprobe：查找聚类中心的个数，默认为1个

# 如果存在的向量太多，通过暴力搜索索引indexFlatL2搜索时间会变长，这里介绍一种加速搜索的方法 indexIVFFlat（倒排文件）。
# 起始就是使用k-means建立聚类中心，然后通过查询最近的聚类中心，然后比较聚类中所有向量得到相似的向量。
# 创建IndexIVFFlat的时候需要指定一个其他的索引作为量化器（quantizer）来计算距离或者相似度。
# faiss提供了两种衡量相似度的方法：1）faiss.METRIC_L2、 2）faiss.METRIC_INNER_PRODUCT。一个是欧式距离，一个是向量内积。
# 还有其他几个参数：nlist：聚类中心的个数；k：查找最相似的k个向量；index.nprobe：查找聚类中心的个数，默认为1个。

# 1.index.nprobe 越大，search time 越长，召回效果越好。
# 2.nlist=2500，不见得越大越好，需要与nprobe 配合，这两个参数同时大才有可能做到好效果。
# 3.不管哪种倒排的时间，在search 阶段都是比暴力求解快很多，0.9s与0.1s级别的差距。
# 以上的时间都没有包括train的时间。也暂时没有做内存使用的比较。

import time

nlist = 100                       #聚类中心的个数
k = 4
quantizer = faiss.IndexFlatL2(d)  # the other index
index = faiss.IndexIVFFlat(quantizer, d, nlist, faiss.METRIC_L2)
# here we specify METRIC_L2, by default it performs inner-product search
assert not index.is_trained

t0 = time.time()
index.train(xb) # 训练数据
t1 = time.time()
print('训练数据时间为 %.2f ' % (t1-t0))
assert index.is_trained

t0 = time.time()
index.add(xb)                  # add may be a bit slower as well
t1 = time.time()
print('加索引时间为 %.2f ' % (t1-t0))

t0 = time.time()
D, I = index.search(xq, k)     # actual search
D, I = index.search(xb[:5], k)     # actual search
print('自己搜索自己的结果为: ', D)
print('查看训练集中前五个最接近的的ID为: ',I)
print('查看训练集中和前五个最接近的距离为: ',D)

t1 = time.time()
print('默认1个聚类中心搜索时间为 %.2f ' % (t1-t0))

print(I[-5:])                  # neighbors of the 5 last queries
index.nprobe = 10              # default nprobe is 1, try a few more

t0 = time.time()
D, I = index.search(xq, k)
t1 = time.time()
print('10个聚类中心搜索时间为 %.2f ' % (t1-t0))

print(I[-5:])                  # neighbors of the 5 last queries

# 减少内存
# 索引IndexFlatL2和IndexIVFFlat都会全量存储所有的向量在内存中
# 为满足大的数据量的需求，faiss提供一种基于Product Quantizer(乘积量化)的压缩算法编码向量大小到指定的字节数。此时，存储的向量时压缩过的，查询的距离也是近似的。

# 使用IndexIVFPQ
nlist = 100 # 聚类中心个数
m = 8 # number of bytes per vector 每个向量都被编码为8个字节大小
k = 4 # 查询相似的k个向量
quantizer = faiss.IndexFlatL2(d)  # this remains the same
index = faiss.IndexIVFPQ(quantizer, d, nlist, m, 8)  #注意这个时候 没有相似度 度量参数
# 8 specifies that each sub-vector is encoded as 8 bits
index.train(xb)
index.add(xb)
D, I = index.search(xb[:5], k) # sanity check
print('查看训练集中前五个最接近的的ID为: ',I)
print('查看训练集中和前五个最接近的距离为: ',D)
index.nprobe = 10              # make comparable with experiment above # 搜索的聚类个数
D, I = index.search(xq, k)     # search
print(I[-5:])

# 之前我们定义的维度为d = 64，向量的数据类型为float32。
# 这里压缩成了8个字节。所以压缩比率为 (64*32/8) / 8 = 32
# 返回的结果见上，第一个向量同自己的距离为1.6157446，不是上上个结果0。因为如上所述返回的是近似距离，但是整体上返回的最相似的top k的向量ID没有变化。

# GPU使用
# ngpus = faiss.get_num_gpus()
# print("number of GPUs:", ngpus)
# # number of GPUs: 0
#
# # 使用1块GPU
# # build a flat (CPU) index
# index_flat = faiss.IndexFlatL2(d)
# # make it into a gpu index
# gpu_index_flat = faiss.index_cpu_to_gpu(res, 0, index_flat)
#
# # 使用全部gpu
# cpu_index = faiss.IndexFlatL2(d)
# gpu_index = faiss.index_cpu_to_all_gpus(cpu_index) # build the index
#
# gpu_index.add(xb)              # add vectors to the index
# print(gpu_index.ntotal)
#
# k = 4                          # we want to see 4 nearest neighbors
# D, I = gpu_index.search(xq, k) # actual search
# print(I[:5])                   # neighbors of the 5 first queries
# print(I[-5:])                  # neighbors of the 5 last queries

# 应用举例：同义词查找
import os
import time
import faiss
import numpy as np
import jieba
from gensim.models import KeyedVectors
USERNAME = os.getenv('USERNAME')

# 测试词向量数据来源：https://github.com/qiangsiwei/bert_distill/tree/master/data/cache/word2vec.gz
# gunzip word2vec.gz 解压后，删除空行，并在首行插入一行：22563 300
VECTOR_FILE = rf'D:\Users\{USERNAME}\data\weibo\data_vectors300.txt'  # 300维的词向量；
wv_from_text = KeyedVectors.load_word2vec_format(VECTOR_FILE, binary=False)
index = faiss.IndexFlatL2(300)   # 与词向量的维度一致
print(index.is_trained)
index.add(wv_from_text.vectors) # 添加词向量到索引
print(index.ntotal) # 看索引的总数量 按行来
start_time = time.time()
test_words = [t for t in ['旅游', '偷笑', '怎么', '一些', '同学', '男人', '自由', '是不是', '分钟', '变成', '充满', '欣赏', '接受', '关系', '新年', '另外', '海鲜'] if t in wv_from_text.index2word]
test_vectors = [wv_from_text.get_vector(w) for w in test_words]
D, I = index.search(np.array(test_vectors), 5)  # 查找最相近的5个词；
for word, id_list in zip(test_words, I):
    print(word, [wv_from_text.index2word[i] for i in id_list])
print('{}个词总共耗时：{}'.format(len(test_words), time.time() - start_time))

# 旅游 ['旅游', '精华游', '峨嵋山', '连岛', '山西旅游']
# 怎么 ['怎么', '如何', '？', '呢', '吗']
# 一些 ['一些', '很多', '许多', '不少', '这些']
# 同学 ['同学', '学生', '老师', '完钱', '打赤脚']
# 男人 ['男人', '女人', '单身女人', '活受罪', '鱼水之欢']
# 自由 ['自由', 'v', '.\u3000', 'so', '维柯']
# 是不是 ['是不是', '那么', '当然', '是否', '也许']
# 分钟 ['分钟', '小时', '连换', '钟才', '钟\u3000']
# 变成 ['变成', '变为', '成为', '成', '漂亮起来']
# 充满 ['充满', '国际都市', '透出来', '独出心裁', '意韵']
# 欣赏 ['欣赏', '异国风光', '悠闲自得', '美极了', '独出心裁']
# 接受 ['接受', '曾\u3000', '续住', '数错', '以此为准']
# 关系 ['关系', '处理事件', '有情可原', '维柯', '人多嘴杂']
# 新年 ['新年', '新春', '合家幸福', '守岁', '顺祝']
# 另外 ['另外', '此外', '同时', '西式早餐', '6.18']
# 海鲜 ['海鲜', '河鲜', '大排挡', '生猛海鲜', '酸菜鱼']
# 16个词总共耗时： 0.08199930191040039

# 另外基于1千万条评论语料，jieba分词，训练的50维词向量，
# 词向量训练参数：word2vec -train comments_cut.txt -output comments_cut.bin -cbow 0 -size 50 -windows 10 -negative 5 -hs 0 -binary 1 -sample 1e-4 -threads 20 -iter 15
# 测试结果：
# 336628 50
# 旅游 ['旅游', 'Travelzoo', '自驾游', '欣新', '孙冰']
# 偷笑 ['偷笑', '哈哈', '嘻嘻', '辜贤武', '回约']
# 怎么 ['怎么', '呢', '为什么', '？', '为啥']
# 一些 ['一些', '些', 'Hooke', 'rick42', '小欣儿']
# 同学 ['同学', '何家强', '芦西西', 'hima', '雷猫']
# 男人 ['男人', '女人', '孙艺滋', '真苏', '女孩子']
# 自由 ['自由', '童飞丽', '木桃良野', 'Potti', '真义']
# 是不是 ['是不是', '还是', '不是', '也', '这个']
# 分钟 ['分钟', '重跑', '拆发仅', '五六分钟', '售磬']
# 变成 ['变成', '变', '成', '真仿', '好史']
# 充满 ['充满', '智慧', '延勇', '董小姚', '异国情调']
# 欣赏 ['欣赏', '司铭妍', '亢旭', '郑荻溪', '缺实']
# 接受 ['接受', '对答如流', '期才', '碍于情面', '归岸']
# 关系 ['关系', '内羊', '佛有', '义子', '梗麦']
# 新年 ['新年', '方院', '甲想', '小小鴿', '快樂蛇']
# 另外 ['另外', '穿戴整齐', '成分表', 'JIAC', '李牧腾']
# 海鲜 ['海鲜', '火锅', '龙虾', '三文鱼', '烧烤']
# 17个词总共耗时：0.189988374710083
