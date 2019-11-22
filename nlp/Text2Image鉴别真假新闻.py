#!/usr/bin/env python
# coding: utf-8

# In[27]:


import pandas as pd
import re
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfTransformer
from scipy.ndimage.filters import gaussian_filter
import time
import numpy as np
import seaborn as sn
from matplotlib.colors import LogNorm
import matplotlib.pyplot as plt
from operator import itemgetter
from fastai.vision import Path, verify_images, ImageDataBunch, cnn_learner, models, accuracy


# Text2Image。其基本思想是将文本转换成我们可以绘制的热图。热图标识着每个单词的 TF-IDF 值。
# 词频 - 逆文档频率 (TF-IDF) 是一种统计方法，用于确定一个单词相对于文档中其他单词的重要性。
# 在基本的预处理和计算 TF-IDF 值之后，我们使用一些平滑的高斯滤波将它们绘制成对数尺度的热图。
# 一旦热图绘制完成，我们使用 fast.ai 实现了一个 CNN，并尝试区分真实和虚假的热图。


#Loading the Dataset
# 真假新闻数据： George Mclntire 的假新闻数据集的子集。它包含大约 1000 篇假新闻和真实新闻的文章
# https://raw.githubusercontent.com/cabhijith/Fake-News/master/fake_or_real_news.csv.zip

df_tes = pd.read_csv("/home/gswyhq/data/fake_or_real_news.csv")
df_test = df_tes[:1000]
df_idf = df_test[df_test['label'] == 'FAKE']


# In[ ]:


# 注意：
# 1）IDF分数分别针对假新闻和真实新闻计算。 因此，上述所有单元必须运行两次。

# 预处理-小写字母，删除标签和特殊字符
def pre_process(text):
    
    
    text=text.lower()

    text=re.sub("</?.*?>"," <> ",text)
 
    text=re.sub("(\\d|\\W)+"," ",text)
    
    return text

df_idf['content'] = df_idf['title'] + df_idf['text']
df_idf['content'] = df_idf['content'].apply(lambda x:pre_process(x))


# In[5]:


from sklearn.feature_extraction.text import CountVectorizer
import re

def get_stop_words(stop_file_path):
    """load stop words """
    
    with open(stop_file_path, 'r', encoding="utf-8") as f:
        stopwords = f.readlines()
        stop_set = set(m.strip() for m in stopwords)
        return frozenset(stop_set)

    #Standard Sklearn stopwords used
# gswyhq@gswyhq-PC:~/data/stanfordnlp$ wget -c -t 0 https://raw.githubusercontent.com/stanfordnlp/CoreNLP/master/data/edu/stanford/nlp/patterns/surface/stopwords.txt
stopwords=get_stop_words("/home/gswyhq/data/stanfordnlp/stopwords.txt")

docs=df_idf['content'].tolist()

# 数据采用小写形式，删除所有特殊字符，并将文本和标题连接起来。
# 文件中 85% 以上的文字也被删除。此外，要明确避免使用单词列表 (stopwords)。
# 使用的是一份标准的停顿词列表，大部分是没有信息的重复词。
# 特别是要对假新闻的断句进行修改，这是未来值得探索的一个领域，特别是可以为假新闻带来独特的写作风格。

cv=CountVectorizer(max_df=0.85,stop_words=stopwords)
word_count_vector=cv.fit_transform(docs)


# In[6]:


# 计算 IDF
# 对于假新闻语料库和真实新闻语料库，IDF 分别计算。与整个语料库的单个 IDF 分数相比，计算单独的 IDF 分数会导致准确性大幅提高。
# 然后迭代计算每个文档的 tf-idf 分数。在这里，标题和文本不是分开评分的，而是一起评分的。

tfidf_transformer=TfidfTransformer(smooth_idf=True,use_idf=True)
tfidf_transformer.fit(word_count_vector)


# In[7]:


# 连接文章标题和内容；
df_tes=pd.read_csv("/home/gswyhq/data/fake_or_real_news.csv")
df_test = df_tes[:1000]

df_test['content'] = df_test['title'] + df_test['text']
df_test['content'] =df_test['content'].apply(lambda x:pre_process(x))

fake_or_real = df_test[df_test['label'] == 'FAKE']
# putting them into a list
docs_test=fake_or_real['content'].tolist()
docs_title=fake_or_real['title'].tolist()
docs_body=fake_or_real['text'].tolist()


# In[8]:


def sort_coo(coo_matrix):
    tuples = zip(coo_matrix.col, coo_matrix.data)
    return sorted(tuples, key=lambda x: (x[1], x[0]), reverse=True)

def extract_topn_from_vector(feature_names, sorted_items, topn=10):
    """获取前n项的特征和tf-idf得分"""
    
    # 获取前n项
    sorted_items = sorted_items[:topn]

    score_vals = []
    feature_vals = []

    for idx, score in sorted_items:
        fname = feature_names[idx]
        
        score_vals.append(round(score, 3))
        feature_vals.append(feature_names[idx])

    #创建一个（特征，得分）的元组
    results= {}
    for idx in range(len(feature_vals)):
        results[feature_vals[idx]]=score_vals[idx]
    
    return results


# 对于每个文档，将提取具有最高 TF-IDF 值的 121 个单词。这些单词然后用于创建一个 11x11 数组。
# 在这里，选择的单词数量就像一个超参数。对于更短、更简单的文本，可以使用更少的单词，而使用更多的单词来表示更长的、更复杂的文本。
# 根据经验，11x11 是这个数据集的理想大小。将 TF-IDF 值按大小降序排列，而不是按其在文本中的位置映射。
# TF-IDF 值以这种方式映射，因为它看起来更能代表文本，并且为模型提供了更丰富的特性来进行训练。
# 因为一个单词可以在一篇文章中出现多次，所以要考虑第一次出现的单词。
#
# 不按原样绘制 TF-IDF 值，而是按对数刻度绘制所有值。这样做是为了减少顶部和底部值之间的巨大差异。
#
#
#
# 在绘制时，由于这种差异，大多数热图不会显示任何颜色的变化。因此，它们被绘制在一个对数刻度上，以便更好地找出差异。


itr = 0 
for i in docs_test:
    feature_names=cv.get_feature_names()

    doc=docs_test[itr]
    itr += 1
    tf_idf_vector=tfidf_transformer.transform(cv.transform([doc]))

    sorted_items=sort_coo(tf_idf_vector.tocoo())
    
    # 获取前81个关键词及 TF-IDF分数
    keywords=extract_topn_from_vector(feature_names,sorted_items,81)
 
   #Getting all of the words according to their positions in the article
    words = []

    for k in keywords:

        words.append(k)

    positions = []
    mapped_values={}
    for i in words:
        for m in re.finditer(i,doc):
            positions.append(m.start())
            mapped_values.update({i:positions})

        positions = []

    sort_vals = {x:sorted(mapped_values[x]) for x in mapped_values.keys()}

    l = [(k,i) for k,v in sort_vals.items() for i in v]
    final_mapping = list(zip(*sorted(l, key=itemgetter(1))[:121]))[0]#Extracting the first 121 words

    value = []
    for f in final_mapping:
        value.append(keywords[f])

        #dividing them into 11 lists each with 11 elements
    def divide(l, n): 


        for i in range(0, len(l), n):  
            yield l[i:i + n] 


    n = 11

    x = list(divide(value, n)) 
    try:
        #smooth =gaussian_filter(x, sigma=1) || Uncomment for gaussian filtering 

        data = np.asarray(x) 
        
        #Plotting the heatmaps in logarithmic scale. 
        log_norm = LogNorm(vmin=data.min().min(), vmax=data.max().max())
        
        pol = sn.heatmap(
            data,
            cmap = 'plasma',
            norm = log_norm,
            xticklabels = False, #Disabling x-axis
            yticklabels = False, #Disabling y-axis
            cbar=False #Disabling colour bar
        ) 
        plt.pause(0.1) #pasuing as to avoid plots getting clogged up
        fig = pol.get_figure()
        a = 'Fake'+ str(itr) 
        fig.savefig('Aretrace10/Fake/' + a) 

        print('These many have been plotted so far --', itr)

    except:
        print('Skipped!')
        continue

    #绘制热图完成


# 最终的热图尺寸为 11x11，用 seaborn 绘制。因为 x 轴和 y 轴以及颜色条在训练时都没有传达任何信息，所以我们删除了它们。使用的热图类型是“等离子体”，因为它显示了理想的颜色变化。
# 该模型使用 fast.ai 在 resnet34 上进行训练。识别出假新闻 489 篇，真新闻 511 篇。在不增加数据的情况下，在训练集和测试集之间采用标准的 80:20 分割。


# 从相对路径加载图像
path = Path("Aretrace10")
path.ls()


# In[14]:


classes = ['Fake', 'Real']


# In[17]:


#Removing any corrupt plots
for c in classes:
    print(c)
    verify_images(path/c, delete=True, max_size=1200)


# In[21]:


#注意：
# 1）数据扩充已被明确关闭，因为对图的任何更改都会改变其基本含义
# 2）使用20％的随机验证集
np.random.seed(42)

data = ImageDataBunch.from_folder(path, train=".", valid_pct=0.20, size = 224 , num_workers=10)


# In[22]:


data.show_batch(rows = 3)


# In[29]:


#Using a resnet34 model. 
learn = cnn_learner(data, models.resnet34 , metrics= accuracy)


# In[30]:


#Running 3 epochs
learn.fit_one_cycle(4)


# In[31]:


learn.save('1Cycle')


# In[32]:


learn.load('1Cycle')
learn.unfreeze()


# In[ ]:


learn.lr_find()


# In[ ]:


learn.recorder.plot(suggestion = True)


# In[ ]:


learn.fit_one_cycle(5, 1e-04) #Training for 4 more epochs

# 来源： https://mp.weixin.qq.com/s/R044ZigOsr8Bvnez7D_JDA
# https://github.com/cabhijith/Text2Image/blob/master/Code.html
