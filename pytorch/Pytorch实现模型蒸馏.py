# decode=utf-8

# 训练模型的优劣性取决于模型的泛化能力，在对预测数据进行预测时，会出现较好的预测结果；

# 通常情况下，复杂度高的网络结构会具有较好的泛化能力，但是资源消耗较大，且存在信息冗余。
# 而所谓的Distilling就是将复杂网络中的有用信息提取出来迁移到一个更小的网络上，这样学习来的小网络可以具备和大的复杂网络想接近的性能效果，并且也大大的节省了计算资源。
# 这个复杂的网络可以看成一个教师，而小的网络则可以看成是一个学生；蒸馏最终的目的是使得学生网络可以具备老师网络的性能，且降低模型复杂度，减少资源消耗。
import os
import copy
import torch
import torch.nn as nn
import numpy as np

from torch.nn import CrossEntropyLoss
from torch.utils.data import TensorDataset,DataLoader,SequentialSampler

# from transformers import BertModel
# from transformers import BertConfig
# USERNAME = os.getenv('USERNAME')
# BERT_BASE_CHINESE_PATH = rf'D:\Users\{USERNAME}\data\bert_base_pytorch\bert-base-chinese'
# model_config = BertConfig.from_pretrained(BERT_BASE_CHINESE_PATH)
# # 通过配置和路径导入模型
# bert_model = BertModel.from_pretrained(BERT_BASE_CHINESE_PATH, config = model_config)

# tiny_model_config = copy.deepcopy(model_config)
# # 修改配置
# tiny_model_config.num_hidden_layers = 4
# tiny_model = BertModel(tiny_model_config)
# model_student = tiny_model
# model_teacher = bert_model

# 基于bert的模型蒸馏详细例子可以参考：https://github.com/qiangsiwei/bert_distill.git

class model(nn.Module):
	def __init__(self,input_dim,hidden_dim,output_dim):
		super(model,self).__init__()
		self.layer1 = nn.LSTM(input_dim,hidden_dim,output_dim,batch_first = True)
		self.layer2 = nn.Linear(hidden_dim,output_dim)
	def forward(self,inputs):
		layer1_output,layer1_hidden = self.layer1(inputs)
		layer2_output = self.layer2(layer1_output)
		layer2_output = layer2_output[:,-1,:]#取出一个batch中每个句子最后一个单词的输出向量即该句子的语义向量！！！！！！！!！
		return layer2_output

#建立小模型
model_student = model(input_dim = 2,hidden_dim = 8,output_dim = 4)

#建立大模型（此处仍然使用LSTM代替，可以使用训练好的BERT等复杂模型）
model_teacher = model(input_dim = 2,hidden_dim = 16,output_dim = 4)

#设置输入数据，此处只使用随机生成的数据代替
inputs = torch.randn(4,6,2)
true_label = torch.tensor([0,1,0,0])

#生成dataset
dataset = TensorDataset(inputs,true_label)

#生成dataloader
sampler = SequentialSampler(inputs)
dataloader = DataLoader(dataset = dataset,sampler = sampler,batch_size = 2)

loss_fun = CrossEntropyLoss()
criterion  = nn.KLDivLoss()#KL散度
optimizer = torch.optim.SGD(model_student.parameters(),lr = 0.1,momentum = 0.9)#优化器，优化器中只传入了学生模型的参数，因此此处只对学生模型进行参数更新，正好实现了教师模型参数不更新的目的

for step,batch in enumerate(dataloader):
	inputs = batch[0]
	labels = batch[1]  # tensor([0, 1])
	
	#分别使用学生模型和教师模型对输入数据进行计算
	output_student = model_student(inputs) # tensor([[ 0.1195, -0.0403, -0.1338, -0.1812], [ 0.1196, -0.0404, -0.1337, -0.1813]], grad_fn=<SliceBackward>)
	output_teacher = model_teacher(inputs)
	
	#计算学生模型和真实标签之间的交叉熵损失函数值
	loss_hard = loss_fun(output_student,labels)  # tensor(1.2145, grad_fn=<NllLossBackward>)
	
	#计算学生模型预测结果和教师模型预测结果之间的KL散度
	loss_soft = criterion(output_student,output_teacher)  # tensor(-0.0763, grad_fn=<KlDivBackward>)
	
	loss = 0.9*loss_soft + 0.1*loss_hard
	print(loss)  # tensor(0.0528, grad_fn=<AddBackward0>)
	optimizer.zero_grad()
	loss.backward()
	optimizer.step()

# 原文链接：https://blog.csdn.net/libaominshouzhang/article/details/109777317



