
BERT将输入文本中的每一个词（token)送入token embedding层从而将每一个词转换成向量形式。
但不同于其他模型的是，BERT又多了两个嵌入层，即segment embeddings和 position embeddings。
token embedding 层是要将各个词转换成固定维度的向量。在BERT中，每个词会被转换成768维的向量表示。

输入文本在送入token embeddings 层之前要先进行tokenization处理。此外，两个特殊的token会被插入到tokenization的结果的开头 ([CLS])和结尾 ([SEP]) 。
Token Embeddings 层会将每一个wordpiece token转换成768维的向量。

segment embeddings 是用来区分一个句子对中的两个句子，句子对中的两个句子被简单的拼接在一起后送入到模型中。
Segment Embeddings 层只有两种向量表示。前一个向量是把0赋给第一个句子中的各个token, 后一个向量是把1赋给第二个句子中的各个token。
如果输入仅仅只有一个句子，那么它的segment embedding就是全0。

position embeddings 会让BERT理解相同的词在句子中不同位置，相同的词在不同位置应该有着不同的向量表示。

BERT能够处理最长512个token的输入序列。Position Embeddings layer 实际上就是一个大小为 (512, 768) 的lookup表，
表的第一行是代表第一个序列的第一个位置，第二行代表序列的第二个位置，以此类推。
因此，如果有这样两个句子“Hello world” 和“Hi there”, “Hello” 和“Hi”会由完全相同的position embeddings，因为他们都是句子的第一个词。
同理，“world” 和“there”也会有相同的position embedding。

合成表示
长度为n的输入序列将获得的三种不同的向量表示，分别是：
Token Embeddings， (1, n, 768) ，词的向量表示
Segment Embeddings， (1, n, 768)，辅助BERT区别句子对中的两个句子的向量表示
Position Embeddings ，(1, n, 768) ，让BERT学习到输入的顺序属性

这些表示会被按元素相加，得到一个大小为(1, n, 768)的合成表示。这一表示就是BERT编码层的输入了。

