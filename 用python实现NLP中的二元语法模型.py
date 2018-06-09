#----coding:utf-8----------------------------------------------------------------
# 名称:  用python实现NLP中的二元语法模型
# 目的: 基于以下语料建立语言模型
# 研究生物很有意思。
# 他大学时代是研究生物的。
# 生物专业是他的首选目标。
# 他是研究生。
# 尝试以“词”作为基元计算出现句子“他是研究生物的”的概率
# 参考:
# 作者:      gswewf
#
# 日期:      2017-04-08
# 版本:      Python 3.3.5
# 系统:      win32
# Email:     gswewf@126.com
#-------------------------------------------------------------------------------

import jieba


def reform(sentence):
    """
    将句子变为"BOSxxxxxEOS"这种形式
    #如果是以“。”结束的则将“。”删掉
    """
    if sentence.endswith("。"):
        sentence=sentence[:-1]
    #添加起始符BOS和终止符EOS
    sentence_modify1=sentence.replace("。", "EOSBOS")
    sentence_modify2="BOS"+sentence_modify1+"EOS"
    return sentence_modify2


#分词并统计词频
def segmentation(sentence,dicts=None):
    """
    接受一个字符串，转换成对应的词频字典及分词列表
    """
    jieba.suggest_freq("BOS", True)
    jieba.suggest_freq("EOS", True)
    sentence = jieba.cut(sentence,HMM=False)
    # format_sentence=",".join(sentence)
    #将词按","分割后依次填入数组word_list[]
    # lists=format_sentence.split(",")

    #统计词频，如果词在字典word_dir{}中出现过则+1，未出现则=1
    lists = [t for t in sentence]
    if isinstance(dicts, dict):
        for index, word in enumerate(lists):
            if index != 0:
                # 二元语法, 记录与前一个词一起出现的词频
                word_2 = '{}_{}'.format(lists[index-1], word)
                dicts.setdefault(word_2, 0)
                dicts[word_2] += 1
            dicts.setdefault(word, 0)
            dicts[word] += 1

    return lists

#计算概率
def probability(test_list,ori_dict):
    """本词跟后一个词一起出现的词频，除以本词单独出现的词频；再累积
    """
    p = 1
    for index, key in enumerate(test_list[:-1]):
        #数据平滑处理：加1法
        # 要计算P(wi|wi−1)条件概率，就计算wi−1wi在文本中出现的次数，再除以wi−1在整个文本中出现的次数
        p *= (ori_dict.get('{}_{}'.format(key, test_list[index+1]), 0)+1)/(ori_dict.get(key, 0)+1)
    return p


def main():
    #语料句子
    sentence_ori="研究生物很有意思。他大学时代是研究生物的。生物专业是他的首选目标。他是研究生。这个杯子是他们国家的圣物，人民都很尊敬圣物，不能亵渎圣物"

    ori_dict={}  # 词频字典
    #分词并将结果存入一个list，词频统计结果存入字典
    sentence_ori_temp=reform(sentence_ori)
    ori_list=segmentation(sentence_ori_temp,ori_dict)


    #测试句子
    sentence_test="他是研究生物的"
    sentence_test2 = "他是研究圣物的"

    sentence_test_temp=reform(sentence_test)
    test_list=segmentation(sentence_test_temp)

    p=probability(test_list,ori_dict)
    print(p)

    sentence_test_temp2=reform(sentence_test2)
    test_list2=segmentation(sentence_test_temp2)

    p2=probability(test_list2,ori_dict)
    print(p2)

if __name__ == "__main__":
    main()
