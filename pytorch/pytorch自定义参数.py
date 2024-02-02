#!/usr/bin/env python
# coding=utf-8

import torch
from torch import nn
# Pytorch自定义参数
# 如果想要灵活地使用模型，可能需要自定义参数，比如

class Net(nn.Module):
    def __init__(self):
        super(Net,self).__init__()
        self.A = torch.randn((2,3),requires_grad=True)
        self.B = nn.Linear(2,2)
    def forward(self,x):
         pass


这里在模型里定义了一个参数矩阵A，但输出模型的参数会发现

>>>net = Net()
>>>for i in net.parameters():
...    print(i)

Parameter containing:
tensor([[-0.6075,  0.5390],
        [ 0.5895, -0.3631]], requires_grad=True)
Parameter containing:
tensor([-0.4341, -0.1234], requires_grad=True)


模型中并没有A，而且模型训练的时候，也不会更新A，将模型移到GPU上时，A也不会跟着走，如果自定义参数，需要手动注册参数

class Net(nn.Module):
    def __init__(self):
        super(Net,self).__init__()
        A = torch.randn((2,3),requires_grad=True)
        self.A = torch.nn.Parameter(A)
        self.B = nn.Linear(2,2)
        self.register_parameter("Ablah",self.A)
    def forward(self,x):
         return x

这样就可以使模型包含参数A了

>>>net = Net()
>>>for i in net.parameters():
...    print(i)

Parameter containing:
tensor([[ 0.5211,  0.2569,  1.1290],
        [-0.5820,  0.1013, -1.3352]], requires_grad=True)
Parameter containing:
tensor([[-0.4867,  0.0765],
        [-0.0178,  0.5943]], requires_grad=True)
Parameter containing:
tensor([0.3423, 0.1557], requires_grad=True)

def main():
    pass


if __name__ == "__main__":
    main()
