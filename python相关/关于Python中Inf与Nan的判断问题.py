#!/usr/bin/python3
# coding: utf-8

import math

# 在Python 中可以用如下方式表示正负无穷：

float("inf") # 正无穷
float("-inf") # 负无穷
float("nan")  # 不是一个数；

# 利用 inf(infinite) 乘以 0 会得到 not-a-number(NaN) 。如果一个数超出 infinite，那就是一个 NaN（not a number）数。

>>> inf = float("inf")

>>> ninf = float("-inf")

>>> nan = float("nan")

>>> inf is inf
True

>>> ninf is ninf

True

>>> nan is nan

True

>>> inf == inf

True

>>> ninf == ninf

True

>>> nan == nan

False

>>> inf is float("inf")

False

>>> ninf is float("-inf")

False

>>> nan is float("nan")

False

>>> inf == float("inf")

True

>>> ninf == float("-inf")

True

>>> nan == float("nan")

False

对于正负无穷和 NaN 自身与自身用 is 操作，结果都是 True，这里好像没有什么问题；但是如果用 == 操作，结果却不一样了， NaN 这时变成了 False。
如果分别用 float 重新定义一个变量来与它们再用 is 和 == 比较，结果仍然出人意料。

如果你希望正确的判断 Inf 和 Nan 值，那么你应该使用 math 模块的 math.isinf 和 math.isnan 函数：

>>> import math

>>> math.isinf(inf)
True

>>> math.isinf(ninf)
True

>>> math.isnan(nan)
True

>>> math.isinf(float("inf"))
True

>>> math.isinf(float("-inf"))
True

>>> math.isnan(float("nan"))
True

不要在 Python 中试图用 is 和 == 来判断一个对象是否是正负无穷或者 NaN。你就乖乖的用 math 模块吧，否则就是引火烧身。

当然也有别的方法来作判断，以下用 NaN 来举例，但仍然推荐用 math 模块，免得把自己弄糊涂。

用对象自身判断自己
>>> def isnan(num):

...  return num != num

...

>>> isnan(float("nan"))
True

用 numpy 模块的函数
>>> import numpy as np
>>>

>>> np.isnan(np.nan)
True

>>> np.isnan(float("nan"))
True

>>> np.isnan(float("inf"))
False

Numpy 的 isnan 函数还可以对整个 list 进行判断：
>>> lst = [1, float("nan"), 2, 3, np.nan, float("-inf"), 4, np.nan]

>>> lst
[1, nan, 2, 3, nan, -inf, 4, nan]

>>> np.isnan(lst)
array([False, True, False, False, True, False, False, True], dtype=bool)

这里的 np.isnan 返回布尔值数组，如果对应位置为 NaN，返回 True，否则返回 False。

参考资料：
https://blog.csdn.net/davidguoguo/article/details/85261172

def main():
    pass


if __name__ == '__main__':
    main()