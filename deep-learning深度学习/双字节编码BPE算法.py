
# BPE是一种压缩算法，是一种自下而上的算法。将单词作为单词片段处理（word pieces），以便于处理未出现单词。在NMT任务中，先将训练集单词划分成片段（利用BPE），然后将片段随机赋值后放到RNNs或CNNs中训练出片段的embedding，再将片段组合得出word的embedding后，进行NMT工作。这样如果在训练集或者其他情况中，遇到生僻词或者未登录词时，直接利用片段进行组合来进行NMT任务。

# BPE算法基本过程如下：

#（1）首先将统计text中单词，做成词汇表（单词-频率），然后按照unigram进行分解。
#（2）寻找频率最大的片段（字符），进行组合，将组合片段加入词汇表。
#（3）继续重复上述操作，直到达到设定的阈值（词汇数+操作数）->操作数是唯一的超参数

import re, collections
def get_stats(vocab):
    pairs = collections.defaultdict(int)
    for word, freq in vocab.items():
        symbols = word.split()
        for i in range(len(symbols)-1):
            pairs[symbols[i],symbols[i+1]] += freq
    return pairs
def merge_vocab(pair, v_in):
    v_out = {}
    bigram = re.escape(' '.join(pair))
    p = re.compile(r'(?<!\S)' + bigram + r'(?!\S)')
    for word in v_in:
        w_out = p.sub(''.join(pair), word)
        v_out[w_out] = v_in[word]
    return v_out
vocab = {'l o w </w>' : 5, 'l o w e r </w>' : 2,
'n e w e s t </w>':6, 'w i d e s t </w>':3}
num_merges = 10
for i in range(num_merges):
    pairs = get_stats(vocab)
    best = max(pairs, key=pairs.get)
    vocab = merge_vocab(best, vocab)
    print(best)

# 其中</w>是作为结束符来使用的。

# BPE仅使用一个字符频率来训练合并操作。频繁的子字符串将在早期连接，从而使常用单词作为唯一的符号保留下来（如the and 等）。
# 由罕见字符组合组成的单词将被分割成更小的单元，例如，子字符串或字符。
# 因此，只有在固定的词汇量很小的情况下(通常是16k到32k)，对一个句子进行编码所需要的符号数量不会显著增加，这是高效解码的一个重要特征。

