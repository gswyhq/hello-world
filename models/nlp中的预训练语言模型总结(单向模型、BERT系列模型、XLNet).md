
nlp中的预训练语言模型总结(单向模型、BERT系列模型、XLNet)

单向特征表示的自回归预训练语言模型，统称为单向模型：
ELMO/ULMFiT/SiATL/GPT1.0/GPT2.0；

双向特征表示的自编码预训练语言模型，统称为BERT系列模型：
(BERT/MASS/UNILM/ERNIE1.0/ERNIE(THU)/MTDNN/ERNIE2.0/SpanBERT/RoBERTa)

双向特征表示的自回归预训练语言模型：XLNet；


[nlp中的预训练语言模型总结](https://mp.weixin.qq.com/s/TLIV0AXgdYupIHpyDlFplw)
[Transformer 模型是怎样实现的](https://mp.weixin.qq.com/s/RwbiEfYUBJKkwwG1YtfvXw)
[XLNet预训练模型](https://mp.weixin.qq.com/s/35JrYKJGtoS1c93rKJGHEA)


模型 | 语言模型 | 特征抽取 | 上下文表征 | 最大亮点
--- | --- | --- | --- | ---
ELMO | BiLM | BiLSTM | 单向 | 2个单向语言模型拼接
ULMFiT | LM | AWD-LSTM | 单向 | 引入逐层解冻解决finetune中的灾难性问题；
SiATL | LM | LSTM | 单向 | 引入逐层解冻+辅助LM解决finetune中的灾难性问题；
GPT1.0 | LM | Transformer | 单向 | 统一下游任务框架，验证Transformer在LM中的强大
GPT2.0 | LM | Transformer | 单向 | 没有特定模型的精调流程，生成任务取得很好效果
BERT | MLM | Transformer | 双向 | MLM获取上下文相关的双向特征表示；
MASS | LM+MLM | Transformer | 单向/双向 | 改进BERT生成任务：统一为类似Seq2Seq的预训练框架；
UNILM | LM+MLM+S2SLM | Transformer | 单向/双向 | 改进BERT生成任务: 直接从mask矩阵的角度出发
ENRIE1.0 | MLM(BPE) | Transformer | 双向 | 引入知识：3种[MASK]策略(BPE)预测短语和实体；
ENRIE | MLM+DEA | Transformer | 双向 | 引入知识: 将实体向量与文本表示融合；
MTDNN | MLM | Transformer | 双向 | 引入多任务学习：在下游阶段；
ENRIE2.0 | MLM+Multi-Task | Transformer | 双向 | 引入多任务学习：在预训练阶段，连续增量学习；
SpanBERT | MLM+SPO | Transformer | 双向 | 不需要按照边界信息进行mask;
RoBERTa | MLM | Transformer | 双向 | 精细调参，舍弃NSP；
XLNet | PLM | Transformer-XL | 双向 | 排列语言模型+双注意力流+Transformer

不同的预训练语言模型目标
自编码（AutoEncode）：BERT系列模型；改进生成任务(MASS/unilm)；引入知识（ENRIE1.0/ENRIE(THU)）；引入多任务(MTDNN/ENRIE2.0);改进MASK（SpanBERT）；精细调参(RoBERTa)
自回归（AutoRegression）：单向模型（ELMO/ULMFiT/SiATL/GPT1.0/GPT2.0）和XLNet(广义自回归)；

特征表示（是否能表示上下文）
单向特征表示：单向模型（ELMO/ULMFiT/SiATL/GPT1.0/GPT2.0）；
双向特征表示：BERT系列模型+XLNet；

长距离依赖建模能力：Transformer-XL > Transformer > RNNs > CNNs

多层感知机(MLP)：不考虑序列（位置）信息，不能处理变长序列，如NNLM和word2vec；
CNNs：考虑序列（位置）信息，不能处理长距离依赖，聚焦于n-gram提取，池化(pooling) 操作会导致序列（位置）信息丢失；
RNNs：天然适合处理序列（位置）信息，但仍不能处理长距离依赖（由于BPTT导致的梯度消失等问题），故又称之为“较长的短期记忆单元(LSTM)”；
Transformer/Transformer-XL：self-attention解决长距离依赖，无位置偏差；

MLP/CNNs/Transformer：前馈/并行
RNNs/ Transformer-XL：循环/串行

自回归语言模型
优点：文本序列联合概率的密度估计，即为传统的语言模型，天然适合处理自然生成任务；
缺点：联合概率按照文本序列从左至右分解（顺序拆解），无法通过上下文信息进行双向特征表征；
改进：XLNet将传统的自回归语言模型进行推广，将顺序拆解变为随机拆解（排列语言模型），产生上下文相关的双向特征表示；

自编码语言模型
优点：本质为降噪自编码特征表示，通过引入噪声[MASK] (掩码)构建MLM(掩码语言模型)，获取上下文相关的双向特征表示；
缺点：引入独立性假设，为联合概率的有偏估计，没有考虑预测[MASK]之间的相关性
    不适合直接处理生成任务，MLM预训练目标的设置造成预训练过程和生成过程不一致；
    预训练时的[MASK]噪声在finetune（微调）阶段不会出现，造成两阶段不匹配问题；



