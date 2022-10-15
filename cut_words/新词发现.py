#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import re
import json
from tqdm import tqdm
import hashlib
from collections import defaultdict
import numpy as np
import pandas as pd
from w3lib.html import remove_tags

CSV_FILE_PATH = r'D:\Users\data\result.csv'


WORD_MAX_LEN = 6
WORD_MIN_COUNT = 128
WORD_MIN_PROBA = {2:5, 3:25, 4:125, 5: 625, 6:3125, 7:15625, 8:125000}

md5 = lambda s: hashlib.md5(s.encode('utf-8')).hexdigest()

# https://spaces.ac.cn/archives/4256

# 引入hashlib是为了对文章去重；引入正则表达式re是为了预先去掉无意义字符（非中文、非英文、非数字）；引入tqdm是为了显示进度。

def texts():
    # 逐篇输出文章
    texts_set = set()
    df = pd.read_csv(CSV_FILE_PATH, encoding='utf-8', error_bad_lines=False, engine='python')
    for message, summary in tqdm(df[[ 'message', 'summary']].values):
        message, summary = remove_tags(str(message)), remove_tags(str(summary))
        for text in [message, summary]:
            if md5(text) in texts_set:
                continue
            else:
                texts_set.add(md5(text))
                for t in re.split(u'[^\u4e00-\u9fa50-9a-zA-Z]+', text):
                    if t:
                        yield t
    # print(u'最终计算了%s篇文章' % len(texts_set))


# 直接计数
# 这里的n就是需要考虑的最长片段的字数（前面说的ngrams），建议至少设为3，min_count看需求而设。

ngrams = defaultdict(int)

for t in texts():
    for i in range(len(t)):
        for j in range(1, WORD_MAX_LEN + 1):
            if i+j <= len(t):
                ngrams[t[i:i+j]] += 1

ngrams = {i:j for i,j in ngrams.items() if j >= WORD_MIN_COUNT}
total = 1.*sum([j for i,j in ngrams.items() if len(i) == 1])

# 接着就是凝固度的筛选了：
# 可以为不同长度的gram设置不同的阈值，因此用了个字典来制定阈值。个人感觉，阈值成5倍的等比数列比较好，当然，这个还有看数据大小。

def is_keep(s, WORD_MIN_PROBA):
    if len(s) >= 2:
        score = min([total*ngrams[s]/(ngrams[s[:i+1]]*ngrams[s[i+1:]]) for i in range(len(s)-1)])
        if score > WORD_MIN_PROBA[len(s)]:
            return True
    else:
        return False

ngrams_ = set(i for i,j in ngrams.items() if is_keep(i, WORD_MIN_PROBA))

# 接着，定义切分函数，并进行切分统计：
def cut(s):
    r = np.array([0]*(len(s)-1))
    for i in range(len(s)-1):
        for j in range(2, WORD_MAX_LEN+1):
            if s[i:i+j] in ngrams_:
                r[i:i+j-1] += 1
    w = [s[0]]
    for i in range(1, len(s)):
        if r[i-1] > 0:
            w[-1] += s[i]
        else:
            w.append(s[i])
    return w

words = defaultdict(int)
for t in texts():
    for i in cut(t):
        words[i] += 1

words = {i:j for i,j in words.items() if j >= WORD_MIN_COUNT}


# 回溯：
def is_real(s):
    if len(s) >= 3:
        for i in range(3, WORD_MAX_LEN+1):
            for j in range(len(s)-i+1):
                if s[j:j+i] not in ngrams_:
                    return False
        return True
    else:
        return True

w = {i:j for i,j in words.items() if is_real(i)}

# 算法的主要时间花在ngrams的统计，以及最后文本的切分上。

with open(CSV_FILE_PATH + '_new_word_result.txt', 'w', encoding='utf-8')as fw:
    for word, weight in sorted(w.items(), key=lambda x: x[1], reverse=True):
        fw.write('{}\t{}\n'.format(word, weight))

def main():
    pass


if __name__ == '__main__':
    main()

