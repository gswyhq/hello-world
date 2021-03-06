#/usr/lib/python3.5
# -*- coding: utf-8 -*-

#http://www.52nlp.cn/%E4%B8%AD%E8%8B%B1%E6%96%87%E7%BB%B4%E5%9F%BA%E7%99%BE%E7%A7%91%E8%AF%AD%E6%96%99%E4%B8%8A%E7%9A%84word2vec%E5%AE%9E%E9%AA%8C

#python train_word2vec_model.py wiki.zh.text.jian.seg wiki.zh.text.model wiki.zh.text.vector


import logging
import os.path,json
import sys, time
import multiprocessing

#from gensim.corpora import WikiCorpus
from gensim.models import Word2Vec
import gensim
#from gensim.models.word2vec import LineSentence
class MySentences(object):
    def __init__(self, dirname,key=''):
        self.dirname = dirname
        self.key=key

    def __iter__(self):
        for fname in os.listdir(self.dirname):
            if self.key and(self.key not in fname):
                continue

            with open(os.path.join(self.dirname, fname),encoding='utf-8')as f:
                for line in json.load(f):
                    yield line[1].split()

def main():

    lei='互动百科'
    path='/home/gswewf/百科/{}/分割文件/三段'.format(lei)
    key='abstract'
    line = MySentences(path,key)

    #加载现有的模型
    model = gensim.models.Word2Vec.load('/home/gswewf/百科/百度百科_维基百科.model')
    #进一步训练模型
    model.train(line)

    #model = Word2Vec(sentences, size=200, window=5, min_count=5,workers=multiprocessing.cpu_count())
    #Word2Vec()
    #Word2vec有很多可以影响训练速度和质量的参数.
    #第一个参数可以对字典做截断. 少于min_count次数的单词会被丢弃掉, 默认值为5
    #另外一个是神经网络的隐藏层的单元数:大的size需要更多的训练数据, 但是效果会更好.
    #最后一个主要的参数控制训练的并行:worker参数只有在安装了Cython后才有效. 没有Cython的话, 只能使用单核.

    # trim unneeded model memory = use(much) less RAM
    #model.init_sims(replace=True)
    #默认格式的word2vec model
    model.save('/home/gswewf/百科/百度百科_维基百科_{}.model'.format(lei))

    #一个原始c版本word2vec的vector格式的模型
    #model.save_word2vec_format(outp2, binary=False)

if __name__ == '__main__':
    start=time.time()
    main()
    print('训练完成总共耗时{}'.format(time.time()-start))
#python train_word2vec_model.py wiki.zh.text.jian.seg wiki.zh.text.model wiki.zh.text.vector
''''

word2vec的参数被存储为矩阵(Numpy array). array的大小为#vocabulary  乘以 #size大小的浮点数(4 byte)矩阵.

内存中有三个这样的矩阵, 如果你的输入包含100,000个单词, 隐层单元数为200, 则需要的内存大小为100,000 * 200 * 4 * 3 bytes, 约为229MB.

另外还需要一些内存来存储字典树, 但是除非你的单词是特别长的字符串, 大部分内存占用都来自前面说的三个矩阵.
评测

Word2vec的训练是无监督的, 没有可以客观的评测结果的好方法. Google提供的一种评测方式为诸如"A之于B相当于C至于D"之类的任务: 参见http://word2vec.googlecode.com/svn/trunk/questions-words.txt

存储、加载模型的方法如下:

>>> model.save('/tmp/mymodel')
>>> new_model = gensim.models.Word2Vec.load('/tmp/mymodel')


另外, 可以直接加载由C生成的模型:

>>> model = Word2Vec.load_word2vec_format('/tmp/vectors.txt', binary=False)
>>> # using gzipped/bz2 input works too, no need to unzip:
>>> model = Word2Vec.load_word2vec_format('/tmp/vectors.bin.gz', binary=True)

可以在加载模型之后使用另外的句子来进一步训练模型

>>> model = gensim.models.Word2Vec.load('/tmp/mymodel')
>>> model.train(more_sentences)

Word2vec支持数种单词相似度任务:

>>> model.most_similar(positive=['woman', 'king'], negative=['man'], topn=1)
[('queen', 0.50882536)]
>>> model.doesnt_match("breakfast cereal dinner lunch".split())
'cereal'
>>> model.similarity('woman', 'man')
.73723527


可以通过以下方式来得到单词的向量:

>>> model['computer']  # raw NumPy vector of a word
array([-0.00449447, -0.00310097,  0.02421786, ...], dtype=float32)






#将输入视为Python的内置列表很简单, 但是在输入很大时会占用大量的内存
sentences=[['毕业', '上海', '上海戏剧学院', '戏剧', '戏剧学院', '戏剧学院表演系', '学院', '表演'],
 [ '表演系', '2011', '中央', '八套', '热播', '皇粮', '胡同', '19', '饰演', '机智', '机智勇敢'],
 [ '智勇', '勇敢', '警察', '扮演', '上司', '刘金山', '金山', '大量', '对手'],
 ['对手戏', '迷雾', '重重', '悬念', '迭起', '剧情', '扣人心弦', '人心', '心弦', '剧中', '侦破'],
 [ '每个', '个案', '案件', '一种', '如释重负', '释重', '重负', '感觉', '刘金山', '金山', '老师', '演对', '对手', '对手戏', '学习'],
 ['乐山', '信息', '科技', '有限', '有限责任', '责任', '公司', '维科', '科技', 'TopWay', '专注'],
 [ 'WEB', '应用', '系统', '统和', '企事业', '事业', '管理', '管理软件', '软件', '软件开发', '开发', '服务']]

model = gensim.models.Word2Vec(sentences, min_count=1)


class MySentences(object):
    def __init__(self, dirname):
        self.dirname = dirname

    def __iter__(self):
        for fname in os.listdir(self.dirname):
            for line in open(os.path.join(self.dirname, fname)):
                yield line.split()

class MySentences(object):
    def __init__(self, dirname):
        self.dirname = dirname

    def __iter__(self):
        for fname in os.listdir(self.dirname):
            yield os.path.join(self.dirname, fname)


import  json
n=len(abstract_datas)
for j in range(10):
    print(int(n/10*j),int(n/10*(j+1)))
    datasn=abstract_datas[int(n/10*j):int(n/10*(j+1))]
    text=''
    for line in datasn:
        text+=line[1]+'\n'
    with open('/home/gswewf/服务器上的文件/百科/三段/百度百科摘要被分词后并分割的文件/百度百科摘要{}'.format(j),'w+',encoding='utf-8')as f:
        print(text,file=f)

>>> import gensim
>>> new_model = gensim.models.Word2Vec.load('/home/gswewf/gow69/baidutext.model')
>>> new_model['家禽']
array([ 0.88960493,  0.14945506, -0.73302388, -2.30815172,  2.05703282,
        0.43654519, -0.9240911 , -1.02483976, -0.79898733, -0.33145291,
        0.06338704,  0.86507887,  0.74484074, -0.8327651 ,  0.92220509,
        0.59975022, -2.61130977,  0.45243055, -0.3451319 , -0.57754385,
       -0.47111478, -0.2026955 , -1.33754623, -0.43259716,  0.45098481,
        0.27361047,  0.42417929, -1.03137755,  0.53435564, -1.02566278,
        0.35103804, -0.41825393,  0.09344709, -2.38145876,  1.511832  ,
       -0.43094352, -0.36848494,  0.0032521 , -1.04929471,  0.83142853,
       -0.22678992, -1.01349318, -0.72621685,  0.66283345,  1.59112716,
        0.65603364,  1.10681164, -0.93984574, -1.11810565, -0.82530183,
       -1.69858253,  1.68676519, -0.38287669,  1.53910363, -1.10241938,
        0.38460553, -0.5443722 ,  0.11216721,  0.83170229, -0.08584389,
        0.66150069,  0.30934882, -1.9350611 ,  0.07849685,  0.92376667,
       -0.58844149,  1.22679055,  0.09482469, -1.12628269, -1.53353679,
        2.19165897,  0.35655046, -0.60840648,  1.12337995,  0.60258001,
       -0.51570576,  0.23710127, -0.26535642, -0.05647113,  0.03858553,
       -0.51930529, -0.13769798,  0.05636241, -0.23334864, -0.87928981,
       -0.12836459, -0.01110756,  1.30872619, -0.49678075,  0.58129603,
        1.4179585 , -0.10165775, -1.00132954, -1.24558294,  1.44584739,
        0.09134513,  0.10097589, -2.16686559, -0.3688232 ,  0.39541999], dtype=float32)
>>> new_model.similarity('中国','中华人民共和国')
0.1646134851480858
>>> new_model.similarity('西红柿','番茄')
0.87069574016264606
>>> new_model.similarity('男人','女人')
0.92044851779235393
>>> new_model.most_similar("西红柿")
[('番茄', 0.8706957697868347), ('豆角', 0.8482689261436462), ('豆芽', 0.8462470173835754), ('南瓜', 0.8442108035087585), ('葱头', 0.8412971496582031), ('黄瓜', 0.8305303454399109), ('青椒', 0.8298680782318115), ('蕃茄', 0.8288747072219849), ('炒蛋', 0.8274224400520325), ('洋葱', 0.8247127532958984)]
>>>
>>> import gensim
>>> new_model = gensim.models.Word2Vec.load('/home/gswewf/gow69/baidutext2.model')   # 此步需要依赖目标文件目录中的`baidutext2.model.syn0.npy`文件
>>> new_model['家禽']
array([  5.51216081e-02,  -4.58016127e-01,  -5.42117774e-01,
         8.97823423e-02,  -1.25151420e+00,   2.29572564e-01,
        -2.42438346e-01,   2.49321669e-01,  -6.49384797e-01,
        -4.91128445e-01,  -1.05605209e+00,  -6.84980094e-01,
        -2.18940437e-01,  -4.95350182e-01,  -2.04297379e-01,
         3.16835910e-01,   1.23431373e+00,  -2.97765851e-01,
         4.78421301e-01,   1.33435857e+00,  -2.07932568e+00,
         2.57597685e-01,   7.60719776e-01,   9.23733473e-01,
        -8.00882131e-02,  -4.79164779e-01,   7.69252360e-01,
        -4.98569399e-01,  -8.94436359e-01,   3.95942293e-02,
         1.26016450e+00,  -9.47634757e-01,   1.03206408e+00,
        -5.12417972e-01,  -1.58549339e-01,  -6.61102653e-01,
         2.35602453e-01,  -6.49534047e-01,   1.94344610e-01,
        -6.13840818e-01,   1.42781174e+00,   1.05032492e+00,
         6.16367459e-01,   3.17391247e-01,  -1.66231140e-01,
        -8.32291543e-01,   9.05503154e-01,   2.30287239e-01,
        -1.92475155e-01,   9.55671668e-02,  -7.91112661e-01,
        -3.24589908e-02,  -4.04689997e-01,   1.38755525e-02,
        -1.57807842e-01,   6.96640491e-01,   7.59823993e-03,
         4.54270452e-01,   1.07830606e-01,   3.74366313e-01,
        -3.72888118e-01,   4.94113080e-02,  -6.04117751e-01,
         7.32849315e-02,   6.39928758e-01,   1.44034326e+00,
        -1.67427182e-01,  -6.14677191e-01,   5.18910706e-01,
         3.43527287e-01,   7.78412044e-01,  -5.65282643e-01,
        -4.39611942e-01,  -3.13354790e-01,  -8.30328986e-02,
        -1.18446842e-01,   1.05900419e+00,  -1.09619880e+00,
        -1.17406189e-01,  -1.50894374e-01,  -6.03995845e-02,
        -9.65311304e-02,  -8.51271331e-01,   1.54758945e-01,
         3.78298938e-01,   2.36431047e-01,  -1.21286082e+00,
        -5.01140356e-01,   2.65203357e-01,   9.08267558e-01,
         3.86332989e-01,   2.51181692e-01,  -1.75864562e-01,
         9.89557862e-01,   9.32287812e-01,  -1.42814815e+00,
        -2.78761983e-01,  -5.86377271e-02,   2.69017190e-01,
         7.82931626e-01,   1.82427838e-01,   2.20635816e-01,
        -6.90142095e-01,   7.98060775e-01,  -2.95841433e-02,
         4.11865801e-01,  -5.04646339e-02,   6.50955200e-01,
         5.67140162e-01,   2.35911503e-01,  -9.81936455e-02,
         1.54145861e+00,  -1.69105932e-01,  -2.25725070e-01,
         5.04198551e-01,   1.16628110e-01,   6.10207140e-01,
         5.91402888e-01,   8.96708220e-02,  -8.33684504e-01,
         3.46887469e-01,   1.48740685e+00,   9.14191961e-01,
         9.57152426e-01,   2.49110445e-01,   2.60175258e-01,
        -3.08678269e-01,  -1.76089406e-01,   7.44278908e-01,
         8.29504430e-01,   7.74918318e-01,  -5.57394743e-01,
        -1.08179130e-01,   8.86960745e-01,  -2.78985143e-01,
        -4.66506213e-01,   1.25811553e+00,   5.75722873e-01,
         5.41218162e-01,  -4.39393938e-01,  -4.20431111118856e-01,
         6.99457109e-01,  -1.91870902e-03,  -1.69534028e-01,
         5.20021737e-01,   4.36077267e-01,   3.75892490e-01,
        -9.81403232e-01,   1.72307655e-01,  -7.85605788e-01,
         1.12113237e+00,  -3.25808376e-01,   2.04899698e-01,
        -6.34368300e-01,  -1.11341929e+00,   1.23272264e+00,
         2.75102645e-01,   1.75640154e+00,   7.86599517e-01,
         4.08096045e-01,   1.35563648e+00,  -2.27261156e-01,
        -1.08042395e+00,   1.44572389e+00,  -9.02759194e-01,
         7.19902158e-01,  -3.77062887e-01,  -4.58282441e-01,
         6.43581510e-01,  -5.38595803e-02,   3.76059681e-01,
        -9.02233899e-01,  -2.41590083e-01,   7.06481874e-01,
        -3.59261394e-01,  -2.33982384e-01,   6.51867986e-01,
        -3.17801833e-01,  -7.59138107e-01,   6.63368821e-01,
        -8.20871666e-02,  -8.14739168e-01,   8.82799923e-01,
         7.99710572e-01,   3.17172915e-01,  -1.22211802e+00,
         1.68628931e+00,  -8.93050134e-01,  -2.89342552e-01,
         4.96631920e-01,  -7.65614331e-01,   4.45084214e-01,
        -1.16906679e+00,   6.02872849e-01,   1.03040433e+00,
        -3.76153678e-01,   2.56610602e-01,   2.99683988e-01,
        -5.86793303e-01,   2.97688872e-01], dtype=float32)
>>> new_model.similarity('中国','中华人民共和国')
0.1253233272626102
>>> new_model.similarity('西红柿','番茄')
0.81940705488697541
>>> new_model.similarity('男人','女人')
0.86733459068152452
>>> new_model.most_similar("西红柿")
[('番茄', 0.8194071650505066), ('青椒', 0.8092970252037048), ('酸豆', 0.7964215278625488), ('豆角', 0.7939008474349976), ('洋葱', 0.7938410639762878), ('青豆', 0.7906121015548706), ('豆丝', 0.7888227105140686), ('红椒', 0.7866231799125671), ('黄瓜', 0.7829074859619141), ('土豆丝', 0.7817968130111694)]
>>>

http://www.52nlp.cn/%E4%B8%AD%E8%8B%B1%E6%96%87%E7%BB%B4%E5%9F%BA%E7%99%BE%E7%A7%91%E8%AF%AD%E6%96%99%E4%B8%8A%E7%9A%84word2vec%E5%AE%9E%E9%AA%8C

http://ju.outofmemory.cn/entry/80023
'''
