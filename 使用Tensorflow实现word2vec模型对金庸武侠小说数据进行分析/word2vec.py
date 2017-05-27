#----coding:utf-8
#  A modified version of Word2Vec TensorFlow implementation
# (github.com/tensorflow/tensorflow/tree/r0.11/tensorflow/examples/tutorials/word2vec)
# 使用Tensorflow实现word2vec模型
# https://github.com/tensorflow/tensorflow/blob/r0.7/tensorflow/examples/tutorials/word2vec/word2vec_basic.py
# According to Stanford 224d Course

import collections
import math
import os
import random
import zipfile

import numpy as np

import tensorflow as tf

# Step 1: Download the data.

url = 'http://mattmahoney.net/dc/'


def maybe_download(filename, expected_bytes):
    """Download a file if not present, and make sure it's the right size."""
    import urllib.request
    if not os.path.exists(filename):
        filename, _ = urllib.request.urlretrieve(url + filename, filename)
    statinfo = os.stat(filename)
    if statinfo.st_size == expected_bytes:
        print('Found and verified', filename)
    else:
        print(statinfo.st_size)
        raise Exception(
                'Failed to verify ' + filename + '. Can you get to it with a browser?')
    return filename

STRING_PUNCTUATION = '—“×}~:%#】－＋!"％]?&|$\'｝：{*》、＠-？/～,>[，…<(.）；【‘=`\\”^／＃｜+。)！’＆＝@_｀;｛￥《（023456789'

def jieba_jinyong(path='/home/gswyhq/jinyong/金庸作品集世纪新修版【TXT】/*.txt'):
    """读取金庸小说集，并用结巴分词，分词后的结果，写入文件"""
    import glob
    import jieba
    out_file = '/home/gswyhq/github_projects/word2vec/jieba_jinyong.txt'
    with open('/home/gswyhq/input/stopwords.dat', encoding='utf8')as f:
        stopwords = [t.strip() for t in f.readlines() if t]

    stopwords += [t for t in STRING_PUNCTUATION]
    files = glob.glob(path)
    with open(out_file, 'a', encoding='utf8')as out_f:
        for fi in files:
            with open(fi, encoding='gbk') as f:
                data = [line.strip() for line in f.readlines() if line.strip()]
            for line in data:
                words = [t for t in jieba.cut(line) if t not in stopwords]
                print(' '.join(words), file=out_f)

# http://mattmahoney.net/dc/text8.zip
# filename = maybe_download('text8.zip', 31344016)

# Read the data into a list of strings.
def read_data(filename='text8.zip'):
    """Extract the first file enclosed in a zip file as a list of words"""
    if filename.endswith('.zip'):
        with zipfile.ZipFile(filename) as f:
            data = tf.compat.as_str(f.read(f.namelist()[0])).split()
    else:
        with open(filename, encoding='utf8')as f:
            data = tf.compat.as_str(f.read()).split()
    return data

# filename = 'text8'
filename = 'jieba_jinyong.txt'

words = read_data(filename)
print('Data size', len(words))
# Step 2: 构建词典，并用 UNK 替换罕见字词.
vocabulary_size = 50000 # 取前50000个词


def build_dataset(words):
    """
    取top vocabulary_size 频数的单词放入dictionary中; 出现在top vocabulary_size 之外的单词，统一令其为“UNK”（未知），编号为0，并统计这些单词的数量。
    输入一个词列表，统计其词频，top词及id字典，id及词字典，词列表对应的id列表
    :param words:
    :return:
    """
    count = [['UNK', -1]]  # 词及其词数
    # In[21]: ts = collections.Counter(['今天', '天气', '今天', '今天', '天气', '今天', '不错', '今天', '天气', '今天', '今天', '不错', '呢我'])
    # In[22]: ts
    # Out[22]: Counter({'不错': 2, '今天': 7, '呢我': 1, '天气': 3})
    # In[23]: ts.most_common(3)
    # Out[23]: [('今天', 7), ('天气', 3), ('不错', 2)]
    # 获取词频，取词频前vocabulary_size-1个词
    count.extend(collections.Counter(words).most_common(vocabulary_size - 1))
    dictionary = dict()  # 词及其id字典
    for word, _ in count:
        dictionary[word] = len(dictionary)  # 词频表中的排序位置即为其id
    data = list()  # 词表对应的词id表；
    unk_count = 0
    for word in words:
        if word in dictionary:
            index = dictionary[word]
        else:
            index = 0  # dictionary['UNK']
            unk_count += 1
        data.append(index)
    count[0][1] = unk_count
    reverse_dictionary = dict(zip(dictionary.values(), dictionary.keys()))  # 词id及其对应的词

    # 词id列表，词及词频表，词及id字典，id及对应词字典
    return data, count, dictionary, reverse_dictionary


data, count, dictionary, reverse_dictionary = build_dataset(words)
del words  # Hint to reduce memory.
print('最常见的词 (+UNK)', count[:5])
print('样例数据', data[:10], [reverse_dictionary[i] for i in data[:10]])

data_index = 0


# Step 3: Function to generate a training batch for the skip-gram model.
def generate_batch(batch_size, num_skips, skip_window):
    """

    :param batch_size: 每个批次训练的样本数
    :param num_skips: 为每个单词生成多少样本，batch_size必须是num_skips的整数倍,这样可以确保由一个目标词汇生成的样本在同一个批次中。
                      num_skips表示在两侧窗口内实际取多少个词，数量可以小于2*skip_window
    :param skip_window: 单词最远可以联系的距离（即目标单词只能和前后相邻的2*skip_window个单词生成样本），2*skip_window>=num_skips
    :return:
    """
    global data_index
    assert batch_size % num_skips == 0  # batch_size // num_skips即为中心词数
    assert num_skips <= 2 * skip_window
    batch = np.ndarray(shape=(batch_size), dtype=np.int32)
    labels = np.ndarray(shape=(batch_size, 1), dtype=np.int32)
    span = 2 * skip_window + 1  # [ skip_window target skip_window ]
    #buffer这个队列，不断地保存span个单词在里面，然后不断往后滑动，而且buffer[skip_window]就是中心词
    buffer = collections.deque(maxlen=span)
    for _ in range(span):
        buffer.append(data[data_index])
        data_index = (data_index + 1) % len(data)
    for i in range(batch_size // num_skips):
        target = skip_window  # skip_window在buffer里正好是中心词所在位置
        targets_to_avoid = [skip_window]
        for j in range(num_skips):
            while target in targets_to_avoid:
                target = random.randint(0, span - 1)  # 当span=3时，target可以取0，1，2
            targets_to_avoid.append(target)
            batch[i * num_skips + j] = buffer[skip_window]
            labels[i * num_skips + j, 0] = buffer[target]
        buffer.append(data[data_index])
        data_index = (data_index + 1) % len(data)
    return batch, labels


batch, labels = generate_batch(batch_size=12, num_skips=6, skip_window=3)
for i in range(12):
    print(batch[i], reverse_dictionary[batch[i]],
          '->', labels[i, 0], reverse_dictionary[labels[i, 0]])

# Step 4: 构建训练一个skip-gram model.

unigrams = [c / vocabulary_size for token, c in count]
batch_size = 128
embedding_size = 128  # Dimension of the embedding vector.
skip_window = 2  # How many words to consider left and right.
num_skips = 1  # How many times to reuse an input to generate a label.

# We pick a random validation set to sample nearest neighbors. Here we limit the
# validation samples to the words that have a low numeric ID, which by
# construction are also the most frequent.
valid_size = 16  # 抽取的验证单词数
valid_window = 100  # 验证单词只从频数最高的100个单词中抽取
valid_examples = np.random.choice(valid_window, valid_size, replace=False)  # 在不重复的0～valid_window,取valid_size个值
num_sampled = 64  # 训练时用来做负样本的噪声单词的数量

graph = tf.Graph()

with graph.as_default():
    # 训练集和标签，以及验证集
    train_inputs = tf.placeholder(tf.int32, shape=[batch_size])
    train_labels = tf.placeholder(tf.int32, shape=[batch_size, 1])
    valid_dataset = tf.constant(valid_examples, dtype=tf.int32)

    # Ops and variables pinned to the CPU because of missing GPU implementation
    with tf.device('/cpu:0'):
        # Look up embeddings for inputs.
        # embeddings = tf.Variable(
        #     tf.random_uniform([vocabulary_size, embedding_size], -1.0, 1.0))
        # embed = tf.nn.embedding_lookup(embeddings, train_inputs)
        #
        # # Construct the variables for the NCE loss
        # nce_weights = tf.Variable(
        #     tf.truncated_normal([vocabulary_size, embedding_size],
        #                         stddev=1.0 / math.sqrt(embedding_size)))
        # nce_biases = tf.Variable(tf.zeros([vocabulary_size]))


        input_ids = train_inputs
        labels = tf.reshape(train_labels, [batch_size])
        # [vocabulary_size, emb_dim] - input vectors

        # 初始化词向量
        input_vectors = tf.Variable(
                tf.random_uniform([vocabulary_size, embedding_size], -1.0, 1.0),
                name="input_vectors")

        # [vocabulary_size, emb_dim] - output vectors
        output_vectors = tf.Variable(
                tf.random_uniform([vocabulary_size, embedding_size], -1.0, 1.0),
                name="output_vectors")

        # [batch_size, 1] - labels
        labels_matrix = tf.reshape(
                tf.cast(labels,
                        dtype=tf.int64),
                [batch_size, 1])

        # Negative sampling.
        sampled_ids, _, _ = (tf.nn.fixed_unigram_candidate_sampler(
                true_classes=labels_matrix,
                num_true=1,
                num_sampled=200,
                unique=True,
                range_max=vocabulary_size,
                distortion=0.75,
                unigrams=unigrams))

        # [batch_size, emb_dim] - Input vectors for center words
        center_vects = tf.nn.embedding_lookup(input_vectors, input_ids)
        # [batch_size, emb_dim] - Output vectors for context words that
        # (center_word, context_word) is in corpus
        context_vects = tf.nn.embedding_lookup(output_vectors, labels)
        # [num_sampled, emb_dim] - vector for sampled words that
        # (center_word, sampled_word) probably isn't in corpus
        sampled_vects = tf.nn.embedding_lookup(output_vectors, sampled_ids)
        # compute logits for pairs of words that are in corpus
        # [batch_size, 1]
        incorpus_logits = tf.reduce_sum(tf.mul(center_vects, context_vects), 1)
        incorpus_probabilities = tf.nn.sigmoid(incorpus_logits)

        # Sampled logits: [batch_size, num_sampled]
        # We replicate sampled noise labels for all examples in the batch
        # using the matmul.
        sampled_logits = tf.matmul(center_vects,
                                   sampled_vects,
                                   transpose_b=True)
        outcorpus_probabilities = tf.nn.sigmoid(-sampled_logits)

    # Compute the average NCE loss for the batch.
    # tf.nce_loss automatically draws a new sample of the negative labels each
    # time we evaluate the loss.
    # [batch_size, 1]
    outcorpus_loss_perexample = tf.reduce_sum(tf.log(outcorpus_probabilities), 1)
    loss_perexample = - tf.log(incorpus_probabilities) - outcorpus_loss_perexample

    # 计算损失
    loss = tf.reduce_sum(loss_perexample) / batch_size

    # 使用0.4的优化率构建SGD优化器
    optimizer = tf.train.GradientDescentOptimizer(.4).minimize(loss)

    # 验证集单词与其他所有单词的相似度计算
    norm = tf.sqrt(tf.reduce_sum(tf.square(input_vectors + output_vectors), 1, keep_dims=True))
    normalized_embeddings = (input_vectors + output_vectors) / norm  # 除以其L2范数后得到标准化后的normalized_embeddings
    valid_embeddings = tf.nn.embedding_lookup(
            normalized_embeddings, valid_dataset)  # #如果输入的是64，那么对应的embedding是normalized_embeddings第64行的vector
    similarity = tf.matmul(
            valid_embeddings, normalized_embeddings, transpose_b=True)  # 计算验证单词的嵌入向量与词汇表中所有单词的相似性

    # Add variable initializer.
    init = tf.global_variables_initializer()

# Step 5: 开始训练.
num_steps = 200000
# num_steps = 100001

with tf.Session(graph=graph) as session:
    # We must initialize all variables before we use them.
    init.run()
    print("Initialized")
    average_loss = 0
    for step in range(num_steps):
        batch_inputs, batch_labels = generate_batch(
                batch_size, num_skips, skip_window)  # 产生批次训练样本
        feed_dict = {train_inputs: batch_inputs, train_labels: batch_labels}  # 赋值

        # We perform one update step by evaluating the optimizer op (including it
        # in the list of returned values for session.run()
        _, loss_val = session.run([optimizer, loss], feed_dict=feed_dict)
        average_loss += loss_val

        if step % 1000 == 0:
            if step > 0:
                average_loss /= 1000
            # The average loss is an estimate of the loss over the last 2000 batches.
            print("平均损失 at step ", step, ": ", average_loss)
            average_loss = 0

        # Note that this is expensive (~20% slowdown if computed every 500 steps)
        if step % 10000 == 0:
            sim = similarity.eval()
            for i in range(valid_size):
                valid_word = reverse_dictionary[valid_examples[i]]  # 得到验证单词
                top_k = 8  # number of nearest neighbors
                nearest = (-sim[i, :]).argsort()[1:top_k + 1]  # 每一个valid_example相似度最高的top-k个单词
                log_str = "临近的 %s:" % valid_word
                for k in range(top_k):
                    close_word = reverse_dictionary[nearest[k]]
                    log_str = "%s %s," % (log_str, close_word)
                print(log_str)
    # 一直到训练结束，再对所有embeddings做一次正则化，得到最后的embedding
    final_embeddings = normalized_embeddings.eval()


# Step 6: Visualize the embeddings.

def plot_with_labels(low_dim_embs, labels, filename='tsne.png'):
    """可视化Word2Vec散点图并保存
    """
    assert low_dim_embs.shape[0] >= len(labels), "More labels than embeddings"
    import matplotlib.font_manager as fm
    myfont = fm.FontProperties(fname='/usr/share/fonts/local/simsun.ttc')
    plt.figure(figsize=(18, 18))  # in inches
    for i, label in enumerate(labels):
        x, y = low_dim_embs[i, :]
        plt.scatter(x, y)
        plt.annotate(label,
                     xy=(x, y),
                     xytext=(5, 2),
                     textcoords='offset points',
                     ha='right',
                     va='bottom',
                     fontproperties=myfont)

    plt.savefig(filename)


try:
    from sklearn.manifold import TSNE
    import matplotlib.pyplot as plt

    # PCA降维
    tsne = TSNE(perplexity=30, n_components=2, init='pca', n_iter=5000)
    plot_only = 500
    low_dim_embs = tsne.fit_transform(final_embeddings[:plot_only, :])
    labels = [reverse_dictionary[i] for i in range(plot_only)]
    plot_with_labels(low_dim_embs, labels)

except ImportError:
    print("Please install sklearn, matplotlib, and scipy to visualize embeddings.")

