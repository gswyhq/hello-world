#!/usr/bin/env python3
# -*- coding: utf-8 -*-


# 链接：https://zhuanlan.zhihu.com/p/359519862

# 一、数据集构造
# 等我们构建完网络之后，要先把参与训练的样本准备好，才更方便设置对应参数。
# 这里最重要的是保证网络的输入输出和数据集是对应的。
# 所以首先先构造一个常用的风险分析数据集。其中包含开发样本（dev）和测试样本（off），并保存相关变量名称。
from sklearn.datasets import make_classification
import pandas as pd
import warnings
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import train_test_split
import toad
import torch
import torch.nn as nn
import torch.nn.functional as F
import math
import numpy as np
torch.set_default_tensor_type(torch.DoubleTensor)

warnings.filterwarnings('ignore')


X, y = make_classification(
    n_samples=2000, n_features=30, n_classes=2, random_state=328)
data = pd.DataFrame(X)
data['bad_ind'] = y
data['imei'] = [i for i in range(len(data))]
data.columns = [
    'f0_radius',
    'f0_texture',
    'f0_perimeter',
    'f0_area',
    'f0_smoothness',
    'f0_compactness',
    'f0_concavity',
    'f0_concave_points',
    'f0_symmetry',
    'f0_fractal_dimension',
    'f1_radius_error',
    'f1_texture_error',
    'f1_perimeter_error',
    'f2_area_error',
    'f2_smoothness_error',
    'f2_compactness_error',
    'f2_concavity_error',
    'f2_concave_points_error',
    'f2_symmetry_error',
    'f2_fractal_dimension_error',
    'f3_radius',
    'f3_texture',
    'f3_perimeter',
    'f3_area',
    'f3_smoothness',
    'f3_compactness',
    'f3_concavity',
    'f3_concave_points',
    'f3_symmetry',
    'f3_fractal_dimension',
    'bad_ind',
    'imei']

df = data.copy()
uid, dep = 'imei', 'bad_ind'
dev, off = train_test_split(
    df, test_size=0.3, stratify=df[dep], random_state=328)
var_names = [i for i in list(df.columns) if i not in [uid, dep]]
trans_var_names = var_names.copy()
# 构造训练与测试数据
x = torch.tensor(dev[trans_var_names].to_numpy(), dtype=torch.double)
y = torch.tensor(dev[dep].to_numpy(), dtype=torch.double)

val_x = torch.tensor(off[trans_var_names].to_numpy(), dtype=torch.double)
val_y = torch.tensor(off[dep].to_numpy(), dtype=torch.double)

del dev, off

# 二、实体嵌入
# 在开始构建网络前，我们首先对实体嵌入过程进行一些介绍。
# 对于取值数量为k的变量x，取值映射为到该变量n个固定分位点的距离，从而将1*k维度的变量，转变为n*k维度的变量。
# 从而更利于后续喂进注意力机制中进行特征调整。
# 而这里经过了一系列实验，在网络中直接使用embedding层进行变换之前，
# 首先使用了toad包中的卡方分箱方法，寻找对应的n个质心（Centroid），然后在网络中计算每个变量实际取值到每个质心的距离。
# 注意，质心的个数是一个超参数，这里的示例中设置为20。

n_bins = 20


combiner = toad.transform.Combiner()
combiner.fit(df[trans_var_names + [dep]], df[dep],
             method='quantile', n_bins=n_bins, exclude=[])
bins = combiner.export()
df_bin = combiner.transform(df[trans_var_names + [dep]])

bin_base = None
for round_num, ft in enumerate(trans_var_names):
    bin_ary = np.array([])
    for i in range(n_bins):
        if np.isnan(df[df_bin[ft] == i][ft].mean()) == False:
            avg = df[df_bin[ft] == i][ft].mean()
        else:
            avg = -1
        bin_ary = np.append(bin_ary, avg)
    if round_num == 0:
        bin_base = bin_ary.copy()
    else:
        bin_base = np.vstack((bin_base, bin_ary))
centroid = torch.tensor(bin_base, dtype=torch.double)
print(centroid.shape)
# 打印结果为：torch.Size([30, 20])。即30个变量，每个变量有一个1*20维度的质心。
# 然后我们只需要结合质心，通过一个embedding层将变量进行映射，然后再喂入Multi-Head Attention结构中，
# 再通过一些神经网络中常见的优质结构，就可以完成这个针对结构化数据进行训练的深度网络。
# 接下来首先搭建实体嵌入使用的embedding层。
# 代码逻辑很清晰，就是计算变量到质心的距离。需要注意的是，网络搭建的过程中，要对变量的维度进行对齐，否则会报错。


class EntityEmbeddingLayer(nn.Module):
    def __init__(self, config):
        super(EntityEmbeddingLayer, self).__init__()
        self.embedding = nn.Embedding(config.num_level, config.embedding_dim)
        self.centroid = config.centroid
        self.EPS = config.EPS

    def forward(self, x):
        """
        x must be batch_size times 1
        """
        cent_hat = torch.tensor(self.centroid[0, :]).detach_().unsqueeze(1)
        x_hat = x[:, 0].unsqueeze(1).unsqueeze(1)
        d = 1.0 / ((x_hat - cent_hat).abs() + self.EPS)
        w = F.softmax(d.squeeze(2), 1)
        v = torch.mm(
            w.type(
                torch.DoubleTensor), self.embedding.weight.type(
                torch.DoubleTensor))
        result = v.unsqueeze(1).type(torch.DoubleTensor)

        if x.size()[1] > 1:
            for i in range(1, x.size()[1]):
                cent_hat = torch.tensor(
                    self.centroid[i, :]).detach_().unsqueeze(1)
                x_hat = x[:, i].unsqueeze(1).unsqueeze(1)
                d = 1.0 / ((x_hat - cent_hat).abs() + self.EPS)
                w = F.softmax(d.squeeze(2), 1)
                v = torch.mm(
                    w.type(
                        torch.DoubleTensor), self.embedding.weight.type(
                        torch.DoubleTensor)).type(
                    torch.DoubleTensor)
                result = torch.cat((result, v.unsqueeze(1)), 1)
        return result

# 三、注意力机制
# 接下来是注意力机制。它和门控机制的功能相同，是从各种可能的角度对我们网络中引入的变量权重进行调整。从而实现特征筛选。这里我们选择Hugging-Face版本中Encoding的部分，抽取并调整后的代码如下：


class SelfAttention(nn.Module):
    def __init__(self, config):
        super().__init__()
        if config.hidden_size % config.num_attention_heads != 0 and not hasattr(
                config, "embedding_size"):
            raise ValueError(
                "The hidden size (%d) is not a multiple of the number of attention "
                "heads (%d)" %
                (config.hidden_size, config.num_attention_heads))
        self.output_attentions = config.output_attentions

        self.num_attention_heads = config.num_attention_heads
        self.attention_head_size = int(
            config.hidden_size / config.num_attention_heads)
        self.all_head_size = self.num_attention_heads * self.attention_head_size

        self.query = nn.Linear(config.hidden_size, self.all_head_size)
        self.key = nn.Linear(config.hidden_size, self.all_head_size)
        self.value = nn.Linear(config.hidden_size, self.all_head_size)

        self.dropout = nn.Dropout(config.attention_probs_dropout_prob)

    def transpose_for_scores(self, x):
        new_x_shape = x.size()[
            :-1] + (self.num_attention_heads, self.attention_head_size)
        x = x.view(*new_x_shape)
        return x.permute(0, 2, 1, 3)

    def forward(self, hidden_states):
        mixed_query_layer = self.query(hidden_states)
        mixed_key_layer = self.key(hidden_states)
        mixed_value_layer = self.value(hidden_states)

        # 计算attention
        query_layer = self.transpose_for_scores(mixed_query_layer)
        key_layer = self.transpose_for_scores(mixed_key_layer)
        value_layer = self.transpose_for_scores(mixed_value_layer)
        attention_scores = torch.matmul(
            query_layer, key_layer.transpose(-1, -2))
        attention_scores = attention_scores / \
            math.sqrt(self.attention_head_size)
        attention_probs = nn.Softmax(dim=-1)(attention_scores)
        attention_probs = self.dropout(attention_probs)
        context_layer = torch.matmul(attention_probs, value_layer)

        # 维度还原
        context_layer = context_layer.permute(0, 2, 1, 3).contiguous()
        new_context_layer_shape = context_layer.size()[
            :-2] + (self.all_head_size,)
        context_layer = context_layer.view(*new_context_layer_shape)

        outputs = (
            context_layer,
            attention_probs) if self.output_attentions else (
            context_layer,
        )
        return outputs

# 四、网络搭建
# 然后对于一个完整的网络我们还需要补齐一些其余的部分，比方说前面提到的稠密连接网络结构。
# 这里也有很多经验性的判断，比方说取巧地使用了注意力机制输出的第一维度向量作为输出结果（即 self_attention_outputs[0]
# ），用于后续的梯度传递，具体代码内容如下：


class TransformerConfig:
    def __init__(self,
                 output_attentions=True,
                 n_var=64,
                 num_attention_heads=2,
                 hidden_size=24,
                 attention_probs_dropout_prob=0.1,
                 hidden_dropout_prob=0.1,
                 intermediate_size=24,
                 num_level=10,
                 embedding_dim=64,
                 EPS=1e-7,
                 centroid=None):
        if centroid is None:
            centroid = {}
        self.output_attentions = output_attentions
        self.num_attention_heads = num_attention_heads
        self.hidden_size = hidden_size
        self.attention_probs_dropout_prob = attention_probs_dropout_prob
        self.hidden_dropout_prob = hidden_dropout_prob
        self.intermediate_size = intermediate_size
        self.n_var = n_var
        self.num_level = num_level
        self.embedding_dim = embedding_dim
        self.EPS = EPS
        self.centroid = centroid


class TransformerOutput(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.dense = nn.Linear(
            config.hidden_size,
            config.hidden_size)  # 稠密链接（网络要想深且有效，dense和res二选一）
        self.LayerNorm = nn.LayerNorm(config.hidden_size)  # 层归一化
        self.dropout = nn.Dropout(config.hidden_dropout_prob)  # dropout

    def forward(self, hidden_states, input_tensor):
        hidden_states = self.dense(hidden_states)
        hidden_states = self.dropout(hidden_states)
        hidden_states = self.LayerNorm(hidden_states + input_tensor)
        return hidden_states


class TransformerIntermediate(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.dense = nn.Linear(config.hidden_size, config.intermediate_size)
        self.intermediate_act_fn = F.relu

    def forward(self, hidden_states):
        hidden_states = self.dense(hidden_states)
        hidden_states = self.intermediate_act_fn(hidden_states)
        return hidden_states


class TransformerLayer(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.EntityEmbeddingLayer = EntityEmbeddingLayer(config)
        self.attention = SelfAttention(config)
        self.intermediate = TransformerIntermediate(config)
        self.output = TransformerOutput(config)
        self.linear = nn.Linear(config.hidden_size * config.n_var, 1)

    def forward(self, hidden):
        result = self.EntityEmbeddingLayer(hidden.type(torch.DoubleTensor))
        self_attention_outputs = self.attention(
            result.type(torch.DoubleTensor))
        attention_output = self_attention_outputs[0]
        # add self attentions if we output attention weights
        outputs = self_attention_outputs[1:]

        intermediate_output = self.intermediate(
            attention_output.type(torch.DoubleTensor))  # 一部分走dense
        layer_output = self.output(
            intermediate_output.type(
                torch.DoubleTensor), attention_output.type(
                torch.DoubleTensor))
        outputs = (layer_output,) + outputs

        result = torch.sigmoid(self.linear(torch.flatten(
            layer_output.type(torch.DoubleTensor), start_dim=1)).squeeze(-1))

        return result


# 接下来，我们设置网络参数，然后打印网络结构来看一下，
# 前面究竟搞了一个什么样子的网络结构。
# 对于隐层维度（hidden_size）、嵌入层的维度（embedding_dim）、随机失活（dropout）、注意力机制的个数（num_attention_heads）等，
# 可以做适当的参数调整。
transformer_config = TransformerConfig(output_attentions=True,
                                       n_var=len(trans_var_names),
                                       num_attention_heads=4,
                                       hidden_size=32,
                                       attention_probs_dropout_prob=0.1,
                                       hidden_dropout_prob=0.1,
                                       intermediate_size=32,
                                       num_level=n_bins,
                                       embedding_dim=32,
                                       centroid=centroid)
#self_attention = SelfAttention(transformer_config)
transformer_layer = TransformerLayer(transformer_config)
transformer_layer
# Out[32]:
# TransformerLayer(
#   (EntityEmbeddingLayer): EntityEmbeddingLayer(
#     (embedding): Embedding(20, 32)
#   )
#   (attention): SelfAttention(
#     (query): Linear(in_features=32, out_features=32, bias=True)
#     (key): Linear(in_features=32, out_features=32, bias=True)
#     (value): Linear(in_features=32, out_features=32, bias=True)
#     (dropout): Dropout(p=0.1, inplace=False)
#   )
#   (intermediate): TransformerIntermediate(
#     (dense): Linear(in_features=32, out_features=32, bias=True)
#   )
#   (output): TransformerOutput(
#     (dense): Linear(in_features=32, out_features=32, bias=True)
#     (LayerNorm): LayerNorm((32,), eps=1e-05, elementwise_affine=True)
#     (dropout): Dropout(p=0.1, inplace=False)
#   )
#   (linear): Linear(in_features=960, out_features=1, bias=True)
# )

# 五、网络训练
# 运行上面所有代码后，接下来就到了训练的过程，这里使用梯度之和进行优化，由于仅做示意使用，也不对学习率这些采取一些特别的调整了。

loss_fn = nn.BCELoss(reduction='sum')  # 返回梯度之和

learning_rate = 1e-4
optimizer = torch.optim.Adam(transformer_layer.parameters(), lr=learning_rate)

NUM_EPOCHS = 10
BATCH_SIZE = 32

loss = None
for epoch in range(NUM_EPOCHS):

    for start in range(0, len(x), BATCH_SIZE):
        end = start + BATCH_SIZE
        batchX = x[start:end, :]
        batchY = y[start:end]

        # 前向传播
        y_pred = transformer_layer(batchX)

        # 计算损失
        loss = loss_fn(y_pred, batchY)

        # 梯度清零
        optimizer.zero_grad()

        # 反向传播
        loss.backward()

        # 参数自动一步更新
        optimizer.step()

    val_pred = transformer_layer(val_x).detach().numpy()
    if epoch % 10:
        print(
            '第{}个epoch'.format(epoch),
            loss.item(),
            "test AUC",
            round(
                roc_auc_score(
                    val_y.detach().numpy(),
                    val_pred),
                4))

# 第1个epoch 16.078802497596456 test AUC 0.6517
# 第2个epoch 15.506801172390134 test AUC 0.6888
# 第3个epoch 14.075249959661278 test AUC 0.7415
# 第4个epoch 12.827904301849285 test AUC 0.7963
# 第5个epoch 12.353158011980069 test AUC 0.8419
# 第6个epoch 11.562640626498762 test AUC 0.8778
# 第7个epoch 10.927640594880671 test AUC 0.8952
# 第8个epoch 10.13935675137856 test AUC 0.909
# 第9个epoch 9.433002768193042 test AUC 0.9186


# 最后对模型进行保存。
PATH = 'transformer_layer.pth'
torch.save(transformer_layer, PATH)
# torch.save(transformer_layer.state_dict(), PATH)  # 仅仅保存参数


def main():
    pass


if __name__ == '__main__':
    main()
