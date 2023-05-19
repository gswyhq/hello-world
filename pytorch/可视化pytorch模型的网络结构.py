#!/usr/bin/python3
# coding: utf-8

from torchvision.models import AlexNet
import torch
from torch import nn
from torchviz import make_dot

# sudo pip3 install torchviz -i http://pypi.douban.com/simple --trusted-host=pypi.douban.com
########################### 方法1：pytorch 模型可视化##################################################################################################
model = nn.Sequential()
model.add_module('W0', nn.Linear(8, 16))
model.add_module('tanh', nn.Tanh())
model.add_module('W1', nn.Linear(16, 1))

x = torch.randn(1, 8)

vise_graph = make_dot(model(x), params=dict(model.named_parameters()))
# 可视化， MLP的网络结构，输出的为一个PDF文件；
vise_graph.view(filename='MLP', directory='.')



model = AlexNet()

x = torch.randn(1, 3, 227, 227).requires_grad_(True)
y = model(x)

# pytorch 打印模型参数
params = list(model.parameters())
k = 0
for i in params:
    l = 1
    print("该层的结构：" + str(list(i.size())))
    for j in i.size():
        l *= j
    print("该层参数和：" + str(l))
    k = k + l
print("总参数数量和：" + str(k))

# 可视化模型结构
vise_graph = make_dot(y, params=dict(list(model.named_parameters()) + [('x', x)]))
vise_graph.view(filename='AlexNet', directory='.')

########################### 方法2：pytorch 模型可视化##################################################################################################
# 方法2： torchviz 可视化 pytorch 模型
# bert模型结构可视化；
import time
import torch
from torchviz import make_dot
from IPython.display import Image
from transformers.models.bert.modeling_bert import MaskedLMOutput
from transformers import BertForMaskedLM, PretrainedConfig

model2_config = {
    'bos_token_id': 0,
    'pad_token_id': 1,
    'eos_token_id': 2,
  'attention_probs_dropout_prob': 0.1,
 'directionality': 'bidi',
 'hidden_act': 'gelu',
 'hidden_dropout_prob': 0.1,
 'hidden_size': 312,
 'initializer_range': 0.02,
 'intermediate_size': 1248,
 'layer_norm_eps': 1e-12,
 'max_position_embeddings': 512,
 'model_type': 'roberta',
 'num_attention_heads': 12,
 'num_hidden_layers': 3,
 'pooler_fc_size': 768,
 'pooler_num_attention_heads': 12,
 'pooler_num_fc_layers': 3,
 'pooler_size_per_head': 128,
 'pooler_type': 'first_token_transform',
 'type_vocab_size': 2,
 'vocab_size': 8021}

model3 = BertForMaskedLM(PretrainedConfig(**model2_config))

inputs_c = {'input_ids': torch.tensor([[ 101,  791, 1921, 1921, 3698, 4696,  679, 7231, 8014, 2769,  812, 1343,
          3141, 3635, 1416, 8013,  102]]),
 'token_type_ids': torch.tensor([[0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]]),
 'attention_mask': torch.tensor([[1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1]])}
outputs_a = model3(**inputs_c, output_hidden_states=False)
vise=make_dot(outputs_a['logits'], params=dict([t for t in model3.named_parameters()] + [(k, v) for k, v in inputs_c.items()]))

time_str = time.strftime('%Y%m%d%H%M%S',time.localtime(time.time()))
print(time_str)
# vise.view()  # 直接在当前路径下保存 pdf 并打开
vise.render(filename="../result/images/{}".format(time_str), view=False, format='png')  # 保存 pdf或png
Image("./result/images/{}.png".format(time_str))  # ipython中展示图片

# 问题1：
# graphviz.backend.execute.ExecutableNotFound: failed to execute WindowsPath('dot'), make sure the Graphviz executables are on your systems' PATH
# 问题所在：一般大家习惯pip install graphviz去安装，但是graphviz是个软件，不能单独用Pip安装
# 解决方案：先将自己安装好的卸载
# pip uninstall graphviz
# 然后下载graphviz的安装包 ，网址：
# https://graphviz.org/download/

# 问题2：
# No such file or directory: 'xdg-open'
# 解决方案：
# apt-get  install xdg-utils
# 当然也可以选择在保存的时候，不打开即可，则不会报该错误；

########################### 方法3：pytorch 模型可视化##################################################################################################
# 方法3：使用torch.jit.trace把模型保存为.pt文件，然后在使用Netron进行打开。
script_model = torch.jit.trace(model, input_x)
script_model.save("model.pt")

# pip3 install netron
# netron命令行可视化模型，浏览器打开“http://localhost:8080/”，查看模型可视化结果
# ~$ netron model.pt

def main():
    pass


if __name__ == '__main__':
    main()