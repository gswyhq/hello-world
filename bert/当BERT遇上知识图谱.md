  
1、KG-BERT: BERT for Knowledge Graph Completion(2019)  
具体的做法，就是修改了BERT模型的输入使其适用于知识库三元组的形式。  
  
首先是KG-BERT(a)，输入为三元组的形式，当然还有BERT自带的special tokens。举个例子，对于三元组   
subject, predicate, object  
头尾实体的输入可以是实体描述句子或者实体名本身, 编码如下：  
[CLS] Token s1, Token s2, ..., Token sn, [SEP] Token r1, Token r2, ..., Token rn, [SEP]  Token o1, Token o2, ..., Token on, [SEP]  
编码说明：  
Token s1, Token s2, ..., Token sn，即为头实体（Head Entity），subject的编码；  
Token r1, Token r2, ..., Token rn，即为关系（Relation）, predicate的编码；  
Token o1, Token o2, ..., Token on，即为尾实体（Tail Entity）, object的编码  
模型训练是首先分别构建positive triple set和negative triple set，然后用BERT的[CLS]标签做一个sigmoid打分以及最后交叉熵损失  
  
在三元组分类、链接预测可以用上面的上述的KG-BERT(a)，但在关系预测任务中需要对上面的输入编码进行改动，改动后的编码变成了：  
[CLS] Token s1, Token s2, ..., Token sn, [SEP]  Token o1, Token o2, ..., Token on, [SEP]  
编码说明：  
Token s1, Token s2, ..., Token sn，即为头实体（Head Entity），subject的编码；  
Token o1, Token o2, ..., Token on，即为尾实体（Tail Entity）, object的编码  
与KG-BERT(a)相比，在KG-BERT(b)中关系（Relation）, predicate的编码并不是作为输出编码；而是将Relation Label作为多分类输出。  
即把sigmoid的二分类改成了softmax的关系多分类。  
  
2、K-BERT: Enabling Language Representation with Knowledge Graph(2019)  
K-BERT通过将知识库中的结构化信息（三元组，也即领域知识图谱）融入到预训练模型中，  
可以更好地解决领域相关任务。如何将外部知识整合到模型中成了一个关键点，这一步通常存在两个难点：  
Heterogeneous Embedding Space： 即文本的单词embedding和知识库的实体实体embedding通常是通过不同方式获取的，使得他俩的向量空间不一致；  
Knowledge Noise： 即过多的知识融合可能会使原始句子偏离正确的本意，过犹不及。  
  
K-BERT 模型的整体框架主要包括了四个子模块： knowledge layer, embedding layer, seeing layer 和 mask-transformer。  
对于一个给定的输入 s = {w0, w1, w2, ..., wn}，首先 knowledge layer会从一个KG中注入相关的三元组，将原来的句子转换成一个knowledge-rich的句子树；  
接着句子树被同时送入embedding layer和seeing layer生成一个token级别的embedding表示和一个可见矩阵（visible matrix）；  
最后通过mask-transformer层编码后用于下游任务的输出。  
Knowledge Layer: 这一层的输入是原始句子 s = {w0, w1, w2, ..., wn}，输出是融入KG信息后的句子树:  
t = {w0, w1, ..., wi{(rio, wio),...,(rik, wik)}, ..., wn}  
 通过两步完成：  
K-Query 输入句子中涉及的所有实体都被选中，并查询它们在KG中对应的三元组 E；  
K-Inject 将查询到的三元组注入到句子中，将 E 中的三元组插入到它们相应的位置，并生成一个句子树 t 。  
  
Embedding Layer  
K-BERT的输入和原始BERT的输入形式是一样的，都需要token embedding, position embedding和segment embedding，  
不同的是，K-BERT的输入是一个句子树，因此问题就变成了句子树到序列化句子的转化，并同时保留结构化信息。  
原始句子： Tim Cook is visiting Beijing now  
关联三元组： [Apple, CEO, Tim Cook], [China, capital, Beijing], [Beijing, is_a, City]  
Token embedding  
句子树的序列化，作者提出一种简单的重排策略：分支中的token被插入到相应节点之后，而后续的token被向后移动。  
举个例子，对于上图中的句子树，则重排后变成了Tim Cook CEO Apple is visiting Beijing capital China is a City now。  
没错，这的确看上去毫无逻辑，但是还好后面可以通过下面Soft-position embedding来解决。  
Soft-position embedding  
通过重排后的句子显然是毫无意义的，这里利用了position embedding来还原回结构信息。  
还是以上图为例，重排后，CEO和Apple被插入在了Cook和is之间，但是is应该是接在Cook之后一个位置的，那么我们直接把is的position number 设置为3即可。  
Segment embedding 部分同BERT一样。  
  
Seeing Layer  
作者认为Seeing layer的mask matrix是K-BERT有效的关键，主要解决了前面提到的Knowledge Noise问题。  
例子中China仅仅修饰的是Beijing，和Apple半毛钱关系没有，因此像这种token之间就不应该有相互影响。  
为此定义一个可见矩阵，判断句子中的单词之间是否彼此影响  
  
Mask-Transformer  
BERT中的Transformer Encoder不能接受上述可见矩阵作为输入，因此需要稍作改进。  
Mask-Transformer是一层层mask-self-attention的堆叠  

[资料来源](https://cloud.tencent.com/developer/article/1543804)
