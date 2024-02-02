# !/usr/bin/env python
# -*- coding:utf-8 -*-

# 安装：
# root@a0e9fe0b494e:~# pip3 install fastHan==1.2 -i http://pypi.douban.com/simple --trusted-host=pypi.douban.com

# 执行以下代码可以加载模型：

from fastHan import FastHan
model=FastHan()

# 模型对句子进行依存分析、命名实体识别的简单例子如下：

sentence="郭靖是金庸笔下的一名男主。"
answer=model(sentence,target="Parsing")
print(answer)
answer=model(sentence,target="NER")
print(answer)


# 词性标注与jieba分词对比：
# >>> from jieba import posseg
# >>> posseg.lcut("设立科创板并试点注册制是提升服务科技创新企业能力、增强市场包容性、强化市场功能的一项资本市场重大改革举措。")
# [pair('设立', 'v'), pair('科', 'n'), pair('创板', 'n'), pair('并', 'c'), pair('试点', 'n'), pair('注册', 'v'), pair('制', 'v'), pair('是', 'v'), pair('提升', 'v'), pair('服务', 'vn'), pair('科技', 'n'), pair('创新', 'v'), pair('企业', 'n'), pair('能力', 'n'), pair('、', 'x'), pair('增强', 'v'), pair('市场', 'n'), pair('包容性', 'n'), pair('、', 'x'), pair('强化', 'v'), pair('市场', 'n'), pair('功能', 'n'), pair('的', 'uj'), pair('一项', 'm'), pair('资本', 'n'), pair('市场', 'n'), pair('重大', 'a'), pair('改革', 'vn'), pair('举措', 'v'), pair('。', 'x')]
# >>> model("设立科创板并试点注册制是提升服务科技创新企业能力、增强市场包容性、强化市场功能的一项资本市场重大改革举措。",target="POS")
# [[['设立', 'VV'], ['科创板', 'NN'], ('并', 'CC'), ['试点', 'VV'], ['注册制', 'NN'], ('是', 'VC'), ['提升', 'VV'], ['服务', 'NN'], ['科技', 'NN'], ['创新', 'NN'], ['企业', 'NN'], ['能力', 'NN'], ('、', 'PU'), ['增强', 'VV'], ['市场', 'NN'], ['包容性', 'NN'], ('、', 'PU'), ['强化', 'VV'], ['市场', 'NN'], ['功能', 'NN'], ('的', 'DEC'), ('一', 'CD'), ('项', 'M'), ['资本', 'NN'], ['市场', 'NN'], ['重大', 'JJ'], ['改革', 'NN'], ['举措', 'NN'], ('。', 'PU')]]
# >>> posseg.lcut("地摊经济是城市的一种边缘经济，一直是影响市容环境的关键因素，但地摊经济有其独特优势，在金融危机背景下能一定程度上缓解就业压力。")
# [pair('地摊', 'n'), pair('经济', 'n'), pair('是', 'v'), pair('城市', 'ns'), pair('的', 'uj'), pair('一种', 'm'), pair('边缘', 'n'), pair('经济', 'n'), pair('，', 'x'), pair('一直', 'd'), pair('是', 'v'), pair('影响', 'vn'), pair('市容', 'n'), pair('环境', 'n'), pair('的', 'uj'), pair('关键因素', 'nr'), pair('，', 'x'), pair('但', 'c'), pair('地摊', 'n'), pair('经济', 'n'), pair('有', 'v'), pair('其', 'r'), pair('独特', 'a'), pair('优势', 'n'), pair('，', 'x'), pair('在', 'p'), pair('金融危机', 'n'), pair('背景', 'n'), pair('下能', 'v'), pair('一定', 'd'), pair('程度', 'n'), pair('上', 'f'), pair('缓解', 'v'), pair('就业', 'v'), pair('压力', 'n'), pair('。', 'x')]
# >>> model("地摊经济是城市的一种边缘经济，一直是影响市容环境的关键因素，但地摊经济有其独特优势，在金融危机背景下能一定程度上缓解就业压力。",target="POS")
# [[['地摊', 'NN'], ['经济', 'NN'], ('是', 'VC'), ['城市', 'NN'], ('的', 'DEG'), ('一', 'CD'), ('种', 'M'), ['边缘', 'NN'], ['经济', 'NN'], ('，', 'PU'), ['一直', 'AD'], ('是', 'VC'), ['影响', 'VV'], ['市容', 'NN'], ['环境', 'NN'], ('的', 'DEC'), ['关键', 'JJ'], ['因素', 'NN'], ('，', 'PU'), ('但', 'AD'), ['地摊', 'NN'], ['经济', 'NN'], ('有', 'VE'), ('其', 'PN'), ['独特', 'JJ'], ['优势', 'NN'], ('，', 'PU'), ('在', 'P'), ['金融', 'NN'], ['危机', 'NN'], ['背景', 'NN'], ('下', 'LC'), ('能', 'VV'), ['一定', 'JJ'], ['程度', 'NN'], ('上', 'LC'), ['缓解', 'VV'], ['就业', 'NN'], ['压力', 'NN'], ('。', 'PU')]]


# 任务选择
#
# target参数可在'Parsing'、'CWS'、'POS'、'NER'四个选项中取值，模型将分别进行依存分析、分词、词性标注、命名实体识别任务,模型默认进行CWS任务。其中词性标注任务包含了分词的信息，而依存分析任务又包含了词性标注任务的信息。命名实体识别任务相较其他任务独立。
#
# 如果分别运行CWS、POS、Parsing任务，模型输出的分词结果等可能存在冲突。如果想获得不冲突的各类信息，请直接运行包含全部所需信息的那项任务。
#
# 模型的POS、Parsing任务均使用CTB标签集。NER使用msra标签集。
#
# 分词风格
#
# 分词风格，指的是训练模型中文分词模块的10个语料库，模型可以区分这10个语料库，设置分词style为S即令模型认为现在正在处理S语料库的分词。所以分词style实际上是与语料库的覆盖面、分词粒度相关的。如本模型默认的CTB语料库分词粒度较细。如果想切换不同的粒度，可以使用模型的set_cws_style函数，例子如下：

sentence="一个苹果。"
print(model(sentence,'CWS'))
model.set_cws_style('cnc')
print(model(sentence,'CWS'))

# 对语料库的选取参考了下方CWS SOTA模型的论文，共包括：SIGHAN 2005的 MSR、PKU、AS、CITYU 语料库，由山西大学发布的 SXU 语料库，由斯坦福的CoreNLP 发布的 CTB6 语料库，由国家语委公布的 CNC 语料库，由王威廉先生公开的微博树库 WTB，由张梅山先生公开的诛仙语料库 ZX，Universal Dependencies 项目的 UD 语料库。
#
# 输入与输出
#
# 输入模型的可以是单独的字符串，也可是由字符串组成的列表。如果输入的是列表，模型将一次性处理所有输入的字符串，所以请自行控制 batch size。
#
# 模型的输出是在fastHan模块中定义的sentence与token类。模型将输出一个由sentence组成的列表，而每个sentence又由token组成。每个token本身代表一个被分好的词，有pos、head、head_label、ner四项属性，代表了该词的词性、依存关系、命名实体识别信息。
#
# 一则输入输出的例子如下所示：

sentence=["我爱踢足球。","林丹是冠军"]
answer=model(sentence,'Parsing')
for i,sentence in enumerate(answer):
    print(i)
    for token in sentence:
        print(token,token.pos,token.head,token.head_label)

# 可在分词风格中选择'as'、'cityu'进行繁体字分词，这两项为繁体语料库。
#
# 此外，由于各项任务共享词表、词嵌入，即使不切换模型的分词风格，模型对繁体字、英文字母、数字均具有一定识别能力。

# 利用下载好了模型的docker镜像，直接使用示例：
# docker run --rm -it -w /root gswyhq/fasthan:1.2 python3 使用fastHan理中文分词、词性标注、依存分析、命名实体识别四项任务.py
