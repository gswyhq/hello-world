# NLP常见任务

序列标注：分词、实体识别、语义标注
分类任务：文本分类、情感计算
句子关系判断：entailment(推理、蕴含)、QA、自然语言处理
生成式任务：机器翻译、文本摘要



| 简写     | 全称                                                         | 数据集描述                                                   | 备注                   |
| -------- | ------------------------------------------------------------ | ------------------------------------------------------------ | ---------------------- |
| MultiNLI | multi-genre natural language inference(多类型文本蕴含关系识别) | 文本间的推理关系，又称为文本蕴含关系。样本都是文本对，第一个文本M作为前提，如果能够从文本M推理出第二个文本N,即可说M蕴含N，M->N。两个文本关系一共有三种entailment（蕴含）、contradiction（矛盾）、neutral（中立） | 基于句子对的分类任务   |
| QQP      | quora question pairs(文本匹配)                               | 判断两个问题的语义是否等价的。                               |                        |
| QNLI     | question natural language inference(自然语言问题推理)        | 二分类任务。正样本为（question，sentence），包含正确的answer；负样本为（question，sentence），不包含正确的answer。用于判断文本是否包含问题的答案，类似于我们做阅读理解定位问题所在的段落。 |                        |
| STS-B    | the semantic textual similarity benchmark(语义文本相似度数据集) | 样本为文本对，评判两个文本语义信息的相似度，分数为1-5。      |                        |
| MRPC     | microsoft research paraphrase corpus(微软研究释义语料库)     | 样本为文本对，判断两个文本的信息是否是等价的                 |                        |
| RTE      | recognizing textual entailment(文本蕴含关系识别)             | 类似于MNLI，但是只是对蕴含关系的二分类判断，而且数据集更小。 |                        |
| SWAG     | the situations with adversarial generations dataset          | 从四个句子中选择为可能为前句下文的那个                       |                        |
| SST-2    | the stanford sentiment treebank(斯坦福情感分类树)            | 分类任务。电影评价的情感分析。                               | 基于单个句子的分类任务 |
| CoLA     | the corpus of linguistic acceptability(语言可接受性语料库)   | 分类任务，预测一个句子是否是acceptable。                     |                        |
| SQuAD    | the standFord question answering dataset(斯坦福问答数据集)   | question，从phrase中选取answer。                             | 问答任务               |
| NER      | named entity recognition(命名实体识别)                       |                                                              | 命名实体识别           |



### GLUE 是一个自然语言任务集合，包括以下这些数据集

| name  | 全称(full name)                      | task                                                  | chinese                                |
| ----- | ------------------------------------ | ----------------------------------------------------- | -------------------------------------- |
| MNLI  | Multi-Genre NLI                      | Natural language inference                            | 自然语言推断                           |
| QQP   | Quora Quora Question Pairs           | Semantic textual similarity/Paraphrase identification | 语义文本相似性/问题对是否等价/释义识别 |
| QNLI  | Question NLI                         |                                                       | 句子是否回答了问题                     |
| SST-2 | Stanford Sentiment Treebank          | Sentiment analysis                                    | 情感分析                               |
| CoLA  | Corpus of Linguistic Acceptability   |                                                       | 句子语言性判断                         |
| STS-B | Semantic Textual Similarity          | Semantic textual similarity                           | 语义相似                               |
| MRPC  | Microsoft Research Paraphrase Corpus |                                                       | 句子对是否语义等价                     |
| RTE   | Recognizing Texual Entailment        | Natural language inference                            | 自然语言推断　识别蕴涵                 |
| WNLI  | Winograd NLI                         | Natural language inference                            | 自然语言推断　识别蕴涵                 |



### 根据判断主题的级别, 将所有的NLP任务分为两种类型:

- **token-level task**: token级别的任务. 如**完形填空**(Cloze), 预测句子中某个位置的单词; 或者**实体识别**; 或是**词性标注**; **SQuAD**等.
- **sequence-level task**: 序列级别的任务, 也可以理解为句子级别的任务. 如**情感分类**等各种句子分类问题; 推断两个句子的是否是同义等.

## token级别的任务(token-level task)

#### 完形填空(Cloze task)

即`BERT`模型预训练的两个任务之一, 等价于**完形填空任务**, 即给出句子中其他的上下文`token`, 推测出当前位置应当是什么`token`.

解决这个问题就可以直接参考`BERT`在预训练时使用到的模型: **masked language model(掩码语言模型)**. 即在与训练时, 将句子中的部分`token`用`[masked]`这个特殊的`token`进行替换, 就是将部分单词遮掩住, 然后目标就是预测`[masked]`对应位置的单词.

这种训练的好处是不需要人工标注的数据. 只需要通过合适的方法, 对现有语料中的句子进行随机的遮掩即可得到可以用来训练的语料. 训练好的模型, 就可以直接使用了.

#### SQuAD(Standford Question Answering Dataset,斯坦福问答数据集) task

这是一个**生成式**的任务. 样本为语句对. 给出一个问题, 和一段来自于*Wikipedia*的文本, 其中这段文本之中, 包含这个问题的答案, 返回一短语句作为答案.

因为给出答案, 这是一个生成式的问题, 这个问题的特殊性在于最终的答案包含在语句对的文本内容之中, 是有范围的, 而且是连续分布在内容之中的.

因此, 我们找出答案在文本语句的开始和结尾处, 就能找到最后的答案. 通过对文本语句序列中每个token对应的所有**hidden vector**做**softmax**判断是开始的概率和是结束的概率, 最大化这个概率就能进行训练, 并得到输出的结果.

#### 命名实体识别(*NamedEntityRecognition*,NER)

本质是对句子中的每个token打标签, 判断每个token的类别.

常用的数据集有:

- **NER**(Named Entity Recognition) **dataset**: 对应于`Person`, `Organization`, `Location`, `Miscellaneous`, or `Other (non-named entity)`.

## 句子级别的任务(sequence-level task)

#### 自然语言推理(Natural Language Inference,NLI) task

**自然语言推断任务**, 即给出**一对**(a pair of)句子, 判断两个句子是*entailment*(相近), *contradiction*(矛盾)还是*neutral*(中立)的. 由于也是分类问题, 也被称为**sentence pair classification tasks**.

在智能问答, 智能客服, 多轮对话中有应用.

常用的数据集有:

- **MNLI**(Multi-Genre Natural Language Inference): 是[**GLUE Datasets**](https://gluebenchmark.com/leaderboard)(通用语言理解评估, General Language Understanding Evaluation)中的一个数据集. 是一个大规模的来源众多的数据集, 目的就是推断两个句子是意思相近, 矛盾, 还是无关的.
- **WNLI**(Winograd NLI, Winograd 自然语言推理)

#### 句对分类任务(即语义匹配任务)Sentence Pair Classification tasks

两个句子相关性的分类问题, `NLI task`是其中的特殊情况. 经典的此类问题和对应的数据集有:

- **QQP**(Quora Question Pairs): 这是一个**二分类**数据集. 目的是判断两个来自于`Quora`的问题句子在语义上是否是等价的.
- **QNLI**(Question Natural Language Inference): 也是一个**二分类**问题, 两个句子是一个`(question, answer)`对. 正样本为`answer`是对应`question`的答案, 负样本则相反.
- **STS-B**(Semantic Textual Similarity Benchmark): 这是一个类似**回归**的问题. 给出一对句子, 使用`1~5`的评分评价两者在语义上的相似程度.
- **MRPC**(Microsoft Research Paraphrase Corpus): 句子对来源于对同一条新闻的评论. 判断这一对句子在语义上是否相同.
- **RTE**(文本蕴含识别，Recognizing Textual Entailment): 是一个**二分类**问题, 类似于**MNLI**, 但是数据量少很多.

#### 单句分类任务(Single Sentence Classification tasks)

- **SST-2**(Stanford Sentiment Treebank): 单句的**二分类**问题, 句子的来源于人们对一部电影的评价, 判断这个句子的情感.
- **CoLA**(Corpus of Linguistic Acceptability): 单句的**二分类**问题, 判断一个英文句子在语法上是不是可接受的.

#### SWAG(Situations With Adversarial Generations)

给出一个陈述句子和4个备选句子, 判断前者与后者中的哪一个最有**逻辑的连续性**, 相当于**阅读理解**问题.



[来源](https://www.cnblogs.com/databingo/p/10182663.html)