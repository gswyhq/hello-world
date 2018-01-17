#!/usr/bin/python3
# coding: utf-8

from jieba.analyse import TFIDF

class ExtractKeywords(TFIDF):
    """
    TF-IDF的主要思想就是：如果某个词在一篇文档中出现的频率高，也即TF高；并且在语料库中其他文档中很少出现，即DF的低，也即IDF高，则认为这个词具有很好的类别区分能力。
    TF-IDF在实际中主要是将二者相乘，也即TF * IDF，TF为词频（Term Frequency），表示词t在文档d中出现的频率；IDF为反文档频率（Inverse Document Frequency），表示语料库中包含词t的文档的数目的倒数。
    应用到关键词抽取：
    1. 预处理，首先进行分词和词性标注，将满足指定词性的词作为候选词；
    2. 分别计算每个词的TF-IDF值；
    3. 根据每个词的TF-IDF值降序排列，并输出指定个数的词汇作为可能的关键词；
    https://www.cnblogs.com/Jace06/p/7106641.html
    """
    def __init__(self, tokenizer=None, postokenizer=None, idf_path=None):
        super(ExtractKeywords, self).__init__(idf_path=idf_path)
        # 使用自定义的分词器覆盖原有分词器, 提取关键词
        if tokenizer:
            self.tokenizer = tokenizer
        if postokenizer:
            self.postokenizer = postokenizer

def main():
    kg_file = '/home/gswyhq/yhb/input/jykl/kg_zhongjixian.txt'
    synonyms_file = '/home/gswyhq/yhb/input/jykl/synonyms_zhongjixian.txt'
    from text_pretreatment import TextPretreatment
    text_pre = TextPretreatment(synonyms_file=synonyms_file, kg_file=kg_file, stop_word_file='')

    ext_key = ExtractKeywords(tokenizer=text_pre.cut_word.jieba_cw, postokenizer=text_pre.cut_word.jieba_cw_posseg)
    # keywords = ext_key.extract_tags('我理解不了安康重疾险是什么意思', withWeight=True, allowPOS=['ns', 'n', 'vn', 'v','nr'], withFlag=True)
    keywords = ext_key.extract_tags('我理解不了安康重疾险是什么意思', withWeight=True, allowPOS=(), withFlag=True)

    print(keywords)

if __name__ == '__main__':
    main()