


# 在google-research/bert/modeling.py中的GELU，是采用近似的方式计算： 
def gelu():
    cdf = 0.5 * (1.0 + np.tanh((np.sqrt(2 / np.pi) * (x + 0.044715 * np.power(x, 3)))))
    return x * cdf

# 而在pretrained-BERT-pytorch/modeling.py中，已经有了精确的计算方式：
def gelu(x):
    """Implementation of the gelu activation function.
        For information: OpenAI GPT's gelu is slightly different (and gives slightly different results):
        0.5 * x * (1 + torch.tanh(math.sqrt(2 / math.pi) * (x + 0.044715 * torch.pow(x, 3))))
        Also see https://arxiv.org/abs/1606.08415
    """
    return x * 0.5 * (1.0 + math.erf(x / math.sqrt(2.0)))

import matplotlib.pyplot as plt
import numpy as np
import math

# 画gelu函数的曲线
plt.plot([i/100 for i in range(-500, 500)],[gelu(i/100) for i in range(-500, 500)],'ro')
plt.show()

def numerical_diff(f, x):
    """
    对函数f在点x处求导
    f:一个函数
    x:函数定义域内一个点
    """
    h = 1e-10  # 计算机保留浮点数是有精度限制的，并且存在舍入误差
    # 因为计算不能保存无限接近于0的数字，也必然存在误差。
    # 为了减少这个误差，一种改进的计算方式是计算(f(x + h) - f(x))/h 改为 计算f在(x + h)和(x-h)之间的差分。
    # 由于这种方法计算式以x为中心，计算它左右两边的差分，所以也称为中心差分。
    return (f(x + h) - f(x-h))/(2*h)

# 近似画gelu函数的导数曲线
plt.plot([i/100 for i in range(-500, 500)],[numerical_diff(gelu, i/100) for i in range(-500, 500)],'ro')
plt.show()


