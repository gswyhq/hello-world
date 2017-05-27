#----coding:utf-8----------------------------------------------------------------
# 名称:        模块1
# 目的:
# 参考:
# 作者:      gswyhq
#
# 日期:      2016-05-21
# 版本:      Python 3.3.5
# 系统:      win32
# Email:     gswyhq@126.com
#-------------------------------------------------------------------------------
'''
 任务 4: 单词联想（Word Association）

自由单词联想是神经语言学（Psycholinguistics）常见的任务，尤其是在词汇检索（lexical retrieval）的语境下——对于人类受试者（human subjects）而言，在单词联想上，更倾向于选择有高度联想性词，而非完全无关的词。这说明单词联想的处理是相当直接的——受试者在听到一个特殊的词时要马上从心里泛起另一个词。

任务：利用大规模词性标注过的语料库来实现自由单词联想。忽略功能词（function words），假设联想词都是名词。

对于这个任务而言，需要用到“词共现”（word co-occurrences）这一概念，例如：统计彼此间最接近的单词出现次数，然后藉此估算出联想度。对于句子中的每个词例，我们将其观察规定范围内接下来所有的词并且利用条件频率分布统计它们在该语境的出现率。清单 4 演示了我们怎么用 Python 和 NLTK 对规定在 5 个单词的范围内的词性标注过的布朗语料库进行处理。
'''
from nltk.corpus import brown, stopwords
from nltk import ConditionalFreqDist
cfd = ConditionalFreqDist()
# 得到英文停用词表
stopwords_list = stopwords.words('english')
# 定义一个函数，如果属于名词类则返回true
def is_noun(tag):
    return tag.lower() in ['nn','nns','nn$','nn-tl','nn+bez','nn+hvz', 'nns$','np','np$','np+bez','nps','nps$','nr','np-tl','nrs','nr$']
...
# 统计前 5 个单词的出现次数
for sentence in brown.tagged_sents():
    for (index, tagtuple) in enumerate(sentence):
        (token, tag) = tagtuple
        token = token.lower()
        if token not in stopwords_list and is_noun(tag):
            window = sentence[index+1:index+5]
            for (window_token, window_tag) in window:
                window_token = window_token.lower()
                if window_token not in stopwords_list and is_noun(window_tag):
                    cfd[token].inc(window_token)
# 好了。我们完成了！让我们开始进行联想！
print( cfd['left'].max())
print (cfd['life'].max())
print (cfd['man'].max())
print (cfd['woman'].max())
print (cfd['boy'].max())
print (cfd['girl'].max())
print (cfd['male'].max())
print (cfd['ball'].max())
print (cfd['doctor'].max())
print (cfd['road'].max())

def main():
    pass

if __name__ == '__main__':
    main()
