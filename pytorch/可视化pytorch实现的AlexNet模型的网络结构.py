#!/usr/bin/python3
# coding: utf-8

from torchvision.models import AlexNet
import torch
from torch import nn
from torchviz import make_dot

# sudo pip3 install torchviz -i http://pypi.douban.com/simple --trusted-host=pypi.douban.com

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

def main():
    pass


if __name__ == '__main__':
    main()