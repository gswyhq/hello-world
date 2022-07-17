# encode:utf-8

# Bert手写版本+MLM+NSP
import re
import math
import torch
import numpy as np
from random import *
import torch.nn as nn
import torch.nn.functional as F 
import torch.optim as optim
import torch.utils.data as Data


# 数据预处理
# 构造单词表和映射

text = (
    'Hello, how are you? I am Romeo.\n'                   # R
    'Hello, Romeo My name is Juliet. Nice to meet you.\n' # J
    'Nice to meet you too. How are you today?\n'          # R
    'Great. My baseball team won the competition.\n'      # J
    'Oh Congratulations, Juliet\n'                        # R
    'Thank you Romeo\n'                                   # J
    'Where are you going today?\n'                        # R
    'I am going shopping. What about you?\n'              # J
    'I am going to visit my grandmother. she is not very well' # R
)
sentences = re.sub("[.,!?\\-]", '', text.lower()).split('\n')    # filter '.', ',', '?', '!'

# 所有句子的单词list
word_list = list(set(" ".join(sentences).split()))               # ['hello', 'how', 'are', 'you',...]

# 给单词表中所有单词设置序号
word2idx = {'[PAD]' : 0, '[CLS]' : 1, '[SEP]' : 2, '[MASK]' : 3}
for i, w in enumerate(word_list):
    word2idx[w] = i + 4

# 用于 idx 映射回 word
idx2word = {i: w for i, w in enumerate(word2idx)}
vocab_size = len(word2idx)         # 40

# token: 就是每个单词在词表中的index
token_list = list()                # token_list存储了每一句的token
for sentence in sentences:
    arr = [word2idx[s] for s in sentence.split()]
    token_list.append(arr)

print(sentences[1])   # hello romeo my name is juliet nice to meet you
print(token_list[1])  # [14, 31, 35, 33, 27, 11, 8, 16, 5, 34]
# hello romeo my name is juliet nice to meet you
# [38, 14, 23, 15, 24, 30, 5, 13, 39, 19]


# 设置超参数

maxlen = 30      # 句子pad到的最大长度，即下面句子中的seq_len
batch_size = 6 

max_pred = 5     # max tokens of prediction
n_layers = 6     # Bert中Transformer的层数
n_heads = 12     # Multi-head的数量
d_model = 768    # 即embedding_dim
d_ff = 768*4     # 4*d_model, FeedForward dimension
d_k = d_v = 64   # dimension of K(=Q), V，是d_model分割成n_heads之后的长度, 768 // 12 = 64

n_segments = 2   # 分隔句子数


# 实现Dataloader
# 生成data
# 选中语料中所有词的15%进行随机mask
# 在确定要Mask掉的单词之后：
# 选中的单词，在80%的概率下被用 [MASK] 来代替
# 选中的单词，在10%的概率下不做mask，用任意非标记词代替
# 选中的单词，在10%的概率下不做mask，仍然保留原来真实的词


# sample IsNext and NotNext to be same in small batch size
def make_data():
    batch = []
    positive = negative = 0
    while (positive != batch_size / 2) or (negative != batch_size / 2):
        # ==========================BERT 的 input 表示================================
        # 随机取两个句子的index
        tokens_a_index, tokens_b_index = randrange(len(sentences)), randrange(len(sentences)) # sample random index in sentences
        # 随机取两个句子
        tokens_a, tokens_b = token_list[tokens_a_index], token_list[tokens_b_index]
        # Token (没有使用word piece): 单词在词典中的编码 
        input_ids = [word2idx['[CLS]']] + tokens_a + [word2idx['[SEP]']] + tokens_b + [word2idx['[SEP]']]
        # Segment: 区分两个句子的编码（上句全为0 (CLS~SEP)，下句全为1）
        segment_ids = [0] * (1 + len(tokens_a) + 1) + [1] * (len(tokens_b) + 1)
        
        # ========================== MASK LM ==========================================
        n_pred = min(max_pred, max(1, int(len(input_ids) * 0.15)))                        # 15 % of tokens in one sentence
        # token在 input_ids 中的下标(不包括[CLS], [SEP])
        cand_maked_pos = [i for i, token in enumerate(input_ids) 
                          if token != word2idx['[CLS]'] and token != word2idx['[SEP]']]  # candidate masked position
        shuffle(cand_maked_pos)
        
        masked_tokens, masked_pos = [], []     # 被mask的tokens，被mask的tokens的索引号
        for pos in cand_maked_pos[:n_pred]:   #  随机mask 15% 的tokens
            masked_pos.append(pos)
            masked_tokens.append(input_ids[pos])
            # 选定要mask的词
            if random() < 0.8:                           # 80%：被真实mask
                input_ids[pos] = word2idx['[MASK]']
            elif random() > 0.9:                        # 10%
                index = randint(0, vocab_size - 1)      # random index in vocabulary
                while index < 4:                       # 不能是 [PAD], [CLS], [SEP], [MASK]
                    index = randint(0, vocab_size - 1)
                input_ids[pos] = index                 # 10%：不做mask，用任意非标记词代替
            # 还有10%：不做mask，什么也不做
            
        # =========================== Paddings ========================================
        # input_ids全部padding到相同的长度
        n_pad = maxlen - len(input_ids)
        input_ids.extend([word2idx['[PAD]']] * n_pad)
        segment_ids.extend([word2idx['[PAD]']] * n_pad)
            
        # zero padding (100% - 15%) tokens
        if max_pred > n_pred:
            n_pad = max_pred - n_pred
            masked_tokens.extend([0] * n_pad)
            masked_pos.extend([0] * n_pad)
        
        # =====================batch添加数据, 让正例 和 负例 数量相同=======================
        if tokens_a_index + 1 == tokens_b_index and positive < batch_size / 2:
            batch.append([input_ids, segment_ids, masked_tokens, masked_pos, True])  # IsNext
            positive += 1
        elif tokens_a_index + 1 != tokens_b_index and negative < batch_size / 2:
            batch.append([input_ids, segment_ids, masked_tokens, masked_pos, False]) # NotNext
            negative += 1
        
    return batch


# 调用上面函数：


batch = make_data()

input_ids, segment_ids, masked_tokens, masked_pos, isNext = zip(*batch)  
print(len(isNext))
# # 全部要转成LongTensor类型
# input_ids, segment_ids, masked_tokens, masked_pos, isNext = \
#     torch.LongTensor(input_ids), torch.LongTensor(segment_ids), torch.LongTensor(masked_tokens), \
#     torch.LongTensor(masked_pos), torch.LongTensor(isNext)
# 6

# 生成DataLoader
# 为了使用dataloader，我们需要定义以下两个function:
# __len__ function：需要返回整个数据集中有多少个item
# __get__：根据给定的index返回一个item
# 有了dataloader之后，我们可以轻松随机打乱整个数据集，拿到一个batch的数据等等。


class MyDataSet(Data.Dataset):
    def __init__(self, input_ids, segment_ids, masked_tokens, masked_pos, isNext):
        # 全部要转成LongTensor类型
        self.input_ids = torch.LongTensor(input_ids)
        self.segment_ids = torch.LongTensor(segment_ids)
        self.masked_tokens = torch.LongTensor(masked_tokens) 
        self.masked_pos = torch.LongTensor(masked_pos) 
        self.isNext = torch.LongTensor(isNext)
        
    def __len__(self):
        return len(self.input_ids)
    
    def __getitem__(self, idx):
        return self.input_ids[idx], self.segment_ids[idx], self.masked_tokens[idx], self.masked_pos[idx], self.isNext[idx]
    
dataset = MyDataSet(input_ids, segment_ids, masked_tokens, masked_pos, isNext)
dataloader = Data.DataLoader(dataset, batch_size=batch_size, shuffle=True)

print(next(iter(dataloader)))
print(len(dataloader))           # 就一个batch
# [tensor([[ 1, 36, 23,  9, 16, 33,  3, 18,  2, 31, 21, 30,  2,  0,  0,  0,  0,  0,
#           0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0],
#         [ 1, 36, 23,  9, 16, 33,  3, 18,  2, 22,  8,  6, 13, 28, 23, 34,  3, 24,
#          11, 27, 37,  2,  0,  0,  0,  0,  0,  0,  0,  0],
#         [ 1, 22,  8,  6,  3, 35, 12, 19,  2,  5, 13, 39, 19, 10, 25, 26, 19, 17,
#           2,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0],
#         [ 1, 38, 14, 23, 15, 24, 30,  5, 13, 39, 19,  2, 38, 14, 23, 15, 24, 30,
#           5, 13,  3, 19,  2,  0,  0,  0,  0,  0,  0,  0],
#         [ 1, 29, 26, 19,  6,  3,  2, 22,  8,  6, 32, 35,  3, 19,  2,  0,  0,  0,
#           0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0],
#         [ 1, 38, 14, 23, 15, 24, 30,  5, 13, 39, 19,  2,  5, 13, 39, 19, 10, 25,
#           3, 19, 17,  2,  0,  0,  0,  0,  0,  0,  0,  0]]), tensor([[0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
#          0, 0, 0, 0, 0, 0],
#         [0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0, 0,
#          0, 0, 0, 0, 0, 0],
#         [0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0, 0, 0, 0, 0,
#          0, 0, 0, 0, 0, 0],
#         [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0,
#          0, 0, 0, 0, 0, 0],
#         [0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0,
#          0, 0, 0, 0, 0, 0],
#         [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0, 0,
#          0, 0, 0, 0, 0, 0]]), tensor([[20,  0,  0,  0,  0],
#         [ 4, 20, 23,  0,  0],
#         [12, 32,  0,  0,  0],
#         [14, 14, 39,  0,  0],
#         [17, 12,  0,  0,  0],
#         [17, 39, 26,  0,  0]]), tensor([[ 6,  0,  0,  0,  0],
#         [16,  6, 14,  0,  0],
#         [ 6,  4,  0,  0,  0],
#         [ 2, 13, 20,  0,  0],
#         [ 5, 12,  0,  0,  0],
#         [20,  9, 18,  0,  0]]), tensor([1, 0, 0, 0, 1, 1])]
# 1


# Bert模型
# Embedding

class BertEmbedding(nn.Module):
    def __init__(self):
        super(BertEmbedding, self).__init__()
        # d_model:即embedding_dim
        # token embedding
        self.tok_embed = nn.Embedding(vocab_size, d_model)  

        # position embedding: 这里简写了,源码中位置编码使用了sin，cos
#         self.pos_embed = nn.Embedding(maxlen, d_model)      
        self.pos_embed = torch.tensor(
            [[pos / (10000.0 ** (i // 2 * 2.0 / d_model)) for i in range(d_model)] for pos in range(maxlen)]
        )
        self.pos_embed[:, 0::2] = torch.sin(self.pos_embed[:, 0::2])
        self.pos_embed[:, 1::2] = torch.cos(self.pos_embed[:, 1::2])
        
        # segment embedding
        self.seg_embed = nn.Embedding(n_segments, d_model)  # segment(token type) embedding

        # LayerNorm
        self.norm = nn.LayerNorm(d_model)
        
    def forward(self, x, seq):                  # x 和 pos的shape 都是[batch_size, seq_len]

#         seq_len = x.size(1)        
#         pos = torch.arange(seq_len, dtype=torch.long)
        # unsqueeze(0): 在索引0处，增加维度--> [1, seq_len]
        # expand: 某个 size=1 的维度上扩展到size
        # expand_as: 把一个tensor变成和函数括号内一样形状的tensor
#         pos = pos.unsqueeze(0).expand_as(x)     # [seq_len] -> [batch_size, seq_len]
    
        # 三个embedding相加
        input_embedding = self.tok_embed(x) + nn.Parameter(self.pos_embed, requires_grad=False) + self.seg_embed(seq)
        
        return self.norm(input_embedding)


# 生成mask

# Padding的部分不应该计算概率，所以需要在相应位置设置mask
# mask==0的内容填充1e-9，使得计算softmax时概率接近0
# 在计算attention时，使用
def get_attn_pad_mask(seq_q, seq_k):    # seq_q 和 seq_k 的 shape 都是 [batch_size, seq_len]
    batch_size, seq_len = seq_q.size()
    # eq(zero) is PAD token
    pad_attn_mask = seq_q.data.eq(0).unsqueeze(1)              # [batcb_size, 1, seq_len]
    return pad_attn_mask.expand(batch_size, seq_len, seq_len) # [batch_size, seq_len, seq_len]


# 构建激活函数

def gelu(x):
    return x * 0.5 * (1.0 + torch.erf(x / math.sqrt(2.0)))


# 缩放点乘注意力计算
# $self-att(Q,K,V) = V \cdot softmax(\frac{K^T \cdot Q}{\sqrt{D_k}}$)

class ScaledDotProductAttention(nn.Module): 
        """
        Scaled Dot-Product Attention
        """
        def __init__(self):
            super(ScaledDotProductAttention, self).__init__()
            
        def forward(self, Q, K, V, attn_mask):
            """
            Args:
                Q: [batch_size, n_heads, seq_len, d_k]
                K: [batch_size, n_heads, seq_len, d_k]
                V: [batch_size, n_heads, seq_len, d_k]
            Return:
                self-attention后的张量，以及attention张量
            """
            # [batch_size, n_heads, seq_len, d_k] * [batch_size, n_heads, d_k, seq_len] = [batch_size, n_heads, seq_len, seq_len]
            score = torch.matmul(Q, K.transpose(-2, -1)) / np.sqrt(d_k)
            
            # mask==0 is PAD token
            # 我们需要防止解码器中的向左信息流来保持自回归属性。 通过屏蔽softmax的输入中所有不合法连接的值（设置为-∞）
            score = score.masked_fill_(attn_mask, -1e9) # mask==0的内容填充-1e9，使得计算softmax时概率接近0
                
            attention = F.softmax(score, dim = -1)          # [bz, n_hs, seq_len, seq_len]
            context = torch.matmul(attention, V)            # [batch_size, n_heads, seq_len, d_k]
            
            return context


# Multi-Head Attention

class MultiHeadAttention(nn.Module):
    def __init__(self):
        super(MultiHeadAttention, self).__init__()
        self.W_Q = nn.Linear(d_model, d_k * n_heads)   # 其实就是[d_model, d_model]
        self.W_K = nn.Linear(d_model, d_k * n_heads)
        self.W_V = nn.Linear(d_model, d_v * n_heads)

    def forward(self, Q, K, V, attn_mask):             # Q和K: [batch_size, seq_len, d_model], V: [batch_size, seq_len, d_model], attn_mask: [batch_size, seq_len, seq_len]
        residual, batch_size = Q, Q.size(0)
        # (B, S, D) -proj-> (B, S, D) -split-> (B, S, H, W) -trans-> (B, H, S, W)
        q_s = self.W_Q(Q).view(batch_size, -1, n_heads, d_k).transpose(1,2)  # q_s: [batch_size, n_heads, seq_len, d_k]
        k_s = self.W_K(K).view(batch_size, -1, n_heads, d_k).transpose(1,2)  # k_s: [batch_size, n_heads, seq_len, d_k]
        v_s = self.W_V(V).view(batch_size, -1, n_heads, d_v).transpose(1,2)  # v_s: [batch_size, n_heads, seq_len, d_v]

        attn_mask = attn_mask.unsqueeze(1).repeat(1, n_heads, 1, 1)          # attn_mask : [batch_size, n_heads, seq_len, seq_len]

        # context: [batch_size, n_heads, seq_len, d_v], attn_mask: [batch_size, n_heads, seq_len, seq_len]
        context = ScaledDotProductAttention()(q_s, k_s, v_s, attn_mask)
        context = context.transpose(1, 2).contiguous().view(batch_size, -1, n_heads * d_v) # context: [batch_size, seq_len, n_heads, d_v]
        
        output = nn.Linear(n_heads * d_v, d_model)(context)
       
        return nn.LayerNorm(d_model)(output + residual)                      # output: [batch_size, seq_len, d_model]


# 前向传播
# Position_wise_Feed_Forward

class PoswiseFeedForwardNet(nn.Module):        # 前向传播，线性激活再线性
    def __init__(self):
        super(PoswiseFeedForwardNet, self).__init__()
        self.fc1 = nn.Linear(d_model, d_ff)
        self.fc2 = nn.Linear(d_ff, d_model)

    def forward(self, x):
        # [batch_size, seq_len, d_model] -> [batch_size, seq_len, d_ff] -> [batch_size, seq_len, d_model]
        return self.fc2(gelu(self.fc1(x)))


# 编码层EncoderLayer
# 源码中 Bidirectional Encoder = Transformer (self-attention)
#
# Transformer = MultiHead_Attention + Feed_Forward with sublayer connection，下面代码省去了sublayer。


class EncoderLayer(nn.Module):    #多头注意力和前向传播的组合
    def __init__(self):
        super(EncoderLayer, self).__init__()
        self.enc_self_attn = MultiHeadAttention()
        self.pos_ffn = PoswiseFeedForwardNet()

    def forward(self, enc_inputs, enc_self_attn_mask):
        enc_outputs = self.enc_self_attn(enc_inputs, enc_inputs, enc_inputs, enc_self_attn_mask) # enc_inputs to same Q,K,V
        enc_outputs = self.pos_ffn(enc_outputs)             # enc_outputs: [batch_size, seq_len, d_model]
        return enc_outputs


# BERT模型

class BERT(nn.Module):
    def __init__(self):
        super(BERT, self).__init__()
        self.embedding = BertEmbedding()
        self.layers = nn.ModuleList([EncoderLayer() for _ in range(n_layers)])
        self.fc = nn.Sequential(
            nn.Linear(d_model, d_model),
            nn.Dropout(0.5),
            nn.Tanh(),
        )
        self.classifier = nn.Linear(d_model, 2)
        self.linear = nn.Linear(d_model, d_model)
        self.activ2 = gelu
        # fc2 is shared with embedding layer
        embed_weight = self.embedding.tok_embed.weight         
        self.fc2 = nn.Linear(d_model, vocab_size, bias=False)
        self.fc2.weight = embed_weight

    # input_ids和segment_ids的shape[batch_size, seq_len]，masked_pos的shape是[batch_size, max_pred]
    def forward(self, input_ids, segment_ids, masked_pos):          
        output = self.embedding(input_ids, segment_ids)             # [bach_size, seq_len, d_model]

        enc_self_attn_mask = get_attn_pad_mask(input_ids, input_ids)# [batch_size, seq_len, seq_len]
        for layer in self.layers:                                  # 这里对layers遍历，相当于源码中多个transformer_blocks
            output = layer(output, enc_self_attn_mask)              # output: [batch_size, seq_len, d_model]

        # it will be decided by first token(CLS)
        h_pooled = self.fc(output[:, 0])                   # [batch_size, d_model]
        logits_clsf = self.classifier(h_pooled)            # [batch_size, 2] predict isNext

        masked_pos = masked_pos[:, :, None].expand(-1, -1, d_model) # [batch_size, max_pred, d_model]
        h_masked = torch.gather(output, 1, masked_pos)              # masking position [batch_size, max_pred, d_model]
        h_masked = self.activ2(self.linear(h_masked))               # [batch_size, max_pred, d_model]
        logits_lm = self.fc2(h_masked)                              # [batch_size, max_pred, vocab_size]
        
        # logits_lm: [batch_size, max_pred, vocab_size], logits_clsf: [batch_size, 2]
        return logits_lm, logits_clsf


# 定义模型

model = BERT()
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adadelta(model.parameters(), lr=0.001)


# 训练模型

for epoch in range(50):
    for input_ids, segment_ids, masked_tokens, masked_pos, isNext in dataloader:
        
        # logits_lm: [batch_size, max_pred, vocab_size]
        # logits_clsf: [batch_size, 2]
        logits_lm, logits_clsf = model(input_ids, segment_ids, masked_pos)          
        
        loss_lm = criterion(logits_lm.view(-1, vocab_size), masked_tokens.view(-1)) # for masked LM
        loss_lm = (loss_lm.float()).mean()
        
        loss_clsf = criterion(logits_clsf, isNext) # for sentence classification
        loss = loss_lm + loss_clsf
        
        if (epoch + 1) % 10 == 0:
            print('Epoch:', '%04d' % (epoch + 1), 'loss =', '{:.6f}'.format(loss))
        
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()


# Epoch: 0010 loss = 1.908749
# Epoch: 0020 loss = 1.354349
# Epoch: 0030 loss = 1.131212
# Epoch: 0040 loss = 1.091269
# Epoch: 0050 loss = 0.891469


# 预测

input_ids, segment_ids, masked_tokens, masked_pos, isNext = batch[1]
print(text)
print('================================')
print([idx2word[w] for w in input_ids if idx2word[w] != '[PAD]'])
# Hello, how are you? I am Romeo.
# Hello, Romeo My name is Juliet. Nice to meet you.
# Nice to meet you too. How are you today?
# Great. My baseball team won the competition.
# Oh Congratulations, Juliet
# Thank you Romeo
# Where are you going today?
# I am going shopping. What about you?
# I am going to visit my grandmother. she is not very well
# ================================
# ['[CLS]', 'great', 'my', 'baseball', 'team', 'won', '[MASK]', 'competition', '[SEP]', 'i', 'am', 'going', 'to', 'visit', 'my', 'grandmother', '[MASK]', 'is', 'not', 'very', 'well', '[SEP]']

logits_lm, logits_clsf = model(torch.LongTensor([input_ids]), torch.LongTensor([segment_ids]), 
                               torch.LongTensor([masked_pos])) # batch=1
# vocab_size维上求max, 输出最大值的索引，第一个batch的max_pred个输出
logits_lm = logits_lm.data.max(2)[1][0].data.numpy()
print('masked tokens list: ', [pos for pos in masked_tokens if pos != 0])
print('predict masked tokens list: ', [pos for pos in logits_lm if pos != 0])
# masked tokens list:  [4, 20, 23]
# predict masked tokens list:  [26, 20, 23]

pred = logits_clsf.data.max(1)[1].data.numpy()[0]
print('isNext : ', True if isNext else False)
print('predict isNext ：', True if pred else False)
# isNext :  False
# predict isNext ： False

# 资料来源： https://github.com/douzujun/NLP-Project/blob/master/Bert%E6%89%8B%E5%86%99%E7%89%88%E6%9C%AC%2BMLM%2BNSP.ipynb

