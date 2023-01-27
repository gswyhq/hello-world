#!/usr/bin/env python3
# -*- coding: utf-8 -*-

# 冻结某些层的参数，有两种思路实现这个目标，一个是设置不要更新参数的网络层为false，另一个就是在定义优化器时只传入要更新的参数。
#
# 最优做法是，优化器只传入requires_grad=True的参数，这样占用的内存会更小一点，效率也会更高。
from torch import nn

# 定义一个简单的网络
class net(nn.Module):
    def __init__(self, num_class=10):
        super(net, self).__init__()
        self.fc1 = nn.Linear(8, 4)
        self.fc2 = nn.Linear(4, num_class)

    def forward(self, x):
        return self.fc2(self.fc1(x))

model = net()

# 将不更新的参数的requires_grad设置为False，同时不将该参数传入optimizer
#
# 将不更新的参数的requires_grad设置为False

# 冻结fc1层的参数
for name, param in model.named_parameters():
    if "fc1" in name:
        param.requires_grad = False

# 不将不更新的模型参数传入optimizer

# 定义一个fliter，只传入requires_grad=True的模型参数
optimizer = optim.SGD(filter(lambda p: p.requires_grad, model.parameters()), lr=1e-2)

# 最优写法
loss_fn = nn.CrossEntropyLoss()

# # 训练前的模型参数
print("model.fc1.weight", model.fc1.weight)
print("model.fc2.weight", model.fc2.weight)
print("model.fc1.weight.requires_grad:", model.fc1.weight.requires_grad)
print("model.fc2.weight.requires_grad:", model.fc2.weight.requires_grad)


for epoch in range(10):
    x = torch.randn((3, 8))
    label = torch.randint(0, 3, [3]).long()
    output = model(x)

    loss = loss_fn(output, label)
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

print("model.fc1.weight", model.fc1.weight)
print("model.fc2.weight", model.fc2.weight)
print("model.fc1.weight.requires_grad:", model.fc1.weight.requires_grad)
print("model.fc2.weight.requires_grad:", model.fc2.weight.requires_grad)

# 节省显存：不将不更新的参数传入optimizer
# 提升速度：将不更新的参数的requires_grad设置为False，节省了计算这部分参数梯度的时间


def main():
    pass


if __name__ == '__main__':
    main()
