#!/usr/bin/env python3
# -*- coding: utf-8 -*-

from torchstat import stat
from transformers import BertForSequenceClassification
from transformers import BertTokenizer
import torch
from torch import nn, optim
import os

USERNAME = os.getenv("USERNAME")

# 模型来源：https://huggingface.co/IDEA-CCNL/Erlangshen-Roberta-110M-Sentiment
model_dir = rf"D:/Users/{USERNAME}/data/Erlangshen-Roberta-110M-Sentiment"

tokenizer=BertTokenizer.from_pretrained(model_dir)
model=BertForSequenceClassification.from_pretrained(model_dir)
model.eval()
text='今天心情不好'

output=model(torch.tensor([tokenizer.encode(text)]))
print(torch.nn.functional.softmax(output.logits,dim=-1))
# tensor([[0.9551, 0.0449]], grad_fn=<SoftmaxBackward0>)

################################################### 查看模型结构及参数,方法1 #########################################################################
from torchsummary import summary
input_x = tokenizer(text)
input_x = {k: torch.as_tensor([v]) for k, v in input_x.items()}
summary(model, **input_x)
# =================================================================
# Total params: 102,269,186
# Trainable params: 102,269,186
# Non-trainable params: 0
################################################### 查看模型结构及参数,方法2 #########################################################################
from torchkeras import summary
input_x = tokenizer(text)
input_x = {k: torch.as_tensor([v]) for k, v in input_x.items()}
summary(model, input_data_args=[input_x['input_ids'], input_x['attention_mask'], input_x['token_type_ids']])

################################################### 网络结构，可视化 #########################################################################
from torchviz import make_dot
from IPython.display import Image

vise=make_dot(output['logits'], params=dict([t for t in model.named_parameters()] + [(k, v) for k, v in input_x.items()]))
vise.render(filename=rf"D:\Users\{USERNAME}\Downloads\test\123", view=False, format='png')  # 保存 pdf或png
Image(rf"D:\Users\{USERNAME}\Downloads\test\123.png")

################################################### 利用 hiddenlayer 可视化网络结构###################################################
import torch
import hiddenlayer as h
input_x = tokenizer(text)
input_x = {k: torch.as_tensor([v]) for k, v in input_x.items()}
myNetGraph = h.build_graph(model, (input_x['input_ids'], input_x['attention_mask'], input_x['token_type_ids']))  # 建立网络模型图
# myNetGraph.theme = h.graph.THEMES['blue']  # blue 和 basic 两种颜色，可以不要
myNetGraph.save(path=rf"D:\Users\{USERNAME}\Downloads\test\123.png", format='png')  # 保存网络模型图，可以设置 png 和 pdf 等
Image(rf"D:\Users\{USERNAME}\Downloads\test\123.png")

# 问题：运行HiddenLayer报错module ‘torch.onnx’ has no attribute ‘_optimize_trace’
# 原因分析：
# 由于pytorch版本较新，hiddenLayer内部的API没有相应地更新，HiddenLayer还是调用的_optimize_trace，而新版pytorch已经改成了_optimize_graph。
# 解决方案：
# 只需要找到，hiddenlayer包里面的pytorch_builder.py程序：我的环境绝对路径是：D:\Anaconda3\setup\envs\AI\Lib\sitepackages\hiddenlayer\pytorch_builder.py，然后在71行改一下就行了，
# torch_graph = torch.onnx._optimize_trace(trace, torch.onnx.OperatorExportTypes.ONNX)  # 将该行注释掉
# torch_graph = torch.onnx._optimize_graph(trace, torch.onnx.OperatorExportTypes.ONNX)  # _optimize_trace -> _optimize_graph

#############################################################################################################################
# 在网络中添加一层：
# BertForSequenceClassification网络下面有三个结点，分别是(bert, dropout, classifier), 我们先在bert结点添加一层'lastlayer'层

model.bert.add_module('lastlayer', nn.Conv2d(512,512, kernel_size=3, stride=1, padding=1))
print(model)

# 在classifier结点添加一个线性层：
model.classifier.add_module('Linear', nn.Linear(2, 10))
print(model)

############################################################################################################################
# 修改网络中的某一层( classifier 结点举例）：
model.classifier = nn.Linear(768, 64, bias=True)
model.dropout = nn.Dropout(p=0.2, inplace=False)

# 或者一次将一个节点改为多层：
model.classifier = nn.Sequential(nn.Linear(768, 64, bias=True), nn.Linear(64, 2, bias=True))

# 也可以对某一节点，根据下标位置修改某一层，如
net.features[8] = nn.Conv2d(512, 512, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
print(net)

############################################################################################################################
# 网络层的删除（features举例） classifier结点的操作相同。
# 直接使用nn.Sequential()对改层设置为空即可
model.bert.pooler = nn.Sequential()
print(model)
net.features[13] = nn.Sequential()
print(net)

############################################################################################################################
# 使用列表切片的方法来删除网络层
model.classifier = nn.Sequential(*list(model.classifier.children())[:-1])  # 删除 classifier 节点下的最后一层
net.features = nn.Sequential(*list(net.features.children())[:-4]) # 删除features节点下的后四层

############################################################################################################################
# 冻结网络中某些层 (直接使该层的requires_grad = False）即可, 这样在反向传播的时候,不会更新该层的参数
#冻结指定层的预训练参数：
net.feature[26].weight.requires_grad = False

# 将不更新的参数的requires_grad设置为False, 节省了计算这部分参数梯度的时间, 提升速度
for name, param in model.named_parameters():
    if "fc1" in name:
        param.requires_grad = False

# 定义一个fliter，只传入requires_grad=True的模型参数, 节省显存
optimizer = optim.Adam(filter(lambda p: p.requires_grad, model.parameters()), lr=1e-5)

################################################## 模型结构修改后，但不生效问题 ##########################################################################
# 注意事项
# 1、为何修改了模型结构却不生效的问题，因为如果是替换，只要保证前后的forward输入输出维度一致，就可以不用改写A.forward()。如果是增减，则需要考虑重写A.forward()。
# 2、如果使用了cuda，并且多卡，需要将model放回cpu后进行结构修改。

# 重写forward示例1：
ALBERT_CHINESE_TINY_PATH = "clue/albert_chinese_tiny"
class MyNet(nn.Module):
    def __init__(self):
        super(MyNet, self).__init__()
        self.albert = AlbertModel.from_pretrained(ALBERT_CHINESE_TINY_PATH)
        for name, param in self.albert.named_parameters():
            param.requires_grad = False
        self.albert.add_module("classifier", torch.nn.Linear(312, 2))

    def forward(self, *args, **kwargs):
        bert_out = self.albert(*args, **kwargs)
        out = self.albert.classifier(bert_out.pooler_output)
        return out
model = MyNet()

# 重写forward示例2：
class MyNet2(nn.Module):
    def __init__(self):
        super(MyNet2, self).__init__()
        self.albert = AlbertModel.from_pretrained(ALBERT_CHINESE_TINY_PATH)
        for name, param in self.albert.named_parameters():
            param.requires_grad = False
        self.dense_out = torch.nn.Sequential()
        self.dense_out.add_module("classifier", torch.nn.Linear(312, 2))

    def forward(self, *args, **kwargs):
        bert_out = self.albert(*args, **kwargs)
        out = self.dense_out(bert_out.pooler_output)
        return out

model2 = MyNet2()

# 需要将model放回cpu后进行结构修改, 示例
model: torch.nn.Module
para_model = torch.nn.DataParallel(model).cuda()
train_or_validate_or_something_else(para_model)
model.cpu()
model.add_module('conv1', NewOne())
para_model = torch.nn.DataParallel(model).cuda()

############################################################################################################################
import torch
from transformers import BertTokenizer, AlbertModel
# https://huggingface.co/clue/albert_chinese_tiny
ALBERT_CHINESE_TINY_PATH = rf'D:\Users\{USERNAME}\data\albert_chinese_tiny'
tokenizer = BertTokenizer.from_pretrained(ALBERT_CHINESE_TINY_PATH)
albert = AlbertModel.from_pretrained(ALBERT_CHINESE_TINY_PATH)

albert.forward(input_x['input_ids'], input_x['attention_mask'], input_x['token_type_ids'])

# 资料来源：https://blog.csdn.net/qq_53345829/article/details/124641236

def main():
    pass


if __name__ == '__main__':
    main()
