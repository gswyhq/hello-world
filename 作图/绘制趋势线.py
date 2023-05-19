
#####################################################################################################################################################################
# 拟合多项式曲线，包含拟合直线

import numpy as np
from matplotlib import pyplot as plt

x = [0.01, 0.02, 0.04, 0.08, 0.16, 0.32, 0.64, 0.8, 1.0]
y = [0.9412, 0.9268, 0.929, 0.9164, 0.9135, 0.8909, 0.8895, 0.8723, 0.8817]
# 展示原始数据
plt.figure(figsize=(16, 10))  # 设置画布大小
plt.plot(x, y)

# 计算趋势线
z = np.polyfit(x, y, 2)  # 第三个参数是：拟合多项式的次数，若为1，则拟合为直线
p = np.poly1d(z)

# 展示趋势线
plt.plot(x+[1.2, 1.5, ], p(x+[1.2, 1.5, ]), linestyle='--', dashes=(5, 1))  # 绘制虚线，plot命令中的dashes=(length, interval space)参数直接指定虚线长度/空格。

plt.show()


#####################################################################################################################################################################
# 拟合指数曲线，拟合指数趋势线1

import numpy as np
import matplotlib.pyplot as plt
import scipy.optimize as optimize
pi = np.pi

# 模拟生成一组实验数据
x = [0.01, 0.02, 0.04, 0.08, 0.16, 0.32, 0.64, 0.8, 1.0]
y = [0.9412, 0.9268, 0.929, 0.9164, 0.9135, 0.8909, 0.8895, 0.8723, 0.8817]

fig, ax = plt.subplots(figsize=(16, 10))
ax.plot(x, y, 'b')


# 拟合指数曲线
def target_func(x, a0, a1, a2):
    return a0 * np.exp(-x / a1) + a2


a0 = max(y) - min(y)
a1 = x[round(len(x) / 2)]
a2 = min(y)
p0 = [a0, a1, a2]
print(p0)
para, cov = optimize.curve_fit(target_func, x, y, p0=p0)
print(para)
y_fit = [target_func(a, *para) for a in x+[1.2, 1.5, 2]]
ax.plot(x+[1.2, 1.5, 2], y_fit, 'y--')

plt.show()

#######################################################################################################################################################################
# 拟合指数曲线，拟合指数趋势线2
# 指数曲线拟合
import numpy as np
x = [0.01, 0.02, 0.04, 0.08, 0.16, 0.32, 0.64, 0.8, 1.0]
y = [0.9412, 0.9268, 0.929, 0.9164, 0.9135, 0.8909, 0.8895, 0.8723, 0.8817]
x = np.array([6, 12, 18, 24, 30])
y = np.array([4, 8, 12, 16, 20])
log_x = np.log(x)
log_y = np.log(y)

coefficients = np.polyfit(x, log_y, 1)
print(coefficients)

c = np.exp(coefficients[1]) * np.exp(coefficients[0]*np.array(x))

plt.figure(figsize=(16, 10))
plt.plot(x, y, "o")
plt.plot(x, c, 'y--')
plt.show()

###################################################################################################################################################################
#####################################################################################################################################################################

# 对数曲线拟合
import numpy as np
import matplotlib.pyplot as plt

x = [0.01, 0.02, 0.04, 0.08, 0.16, 0.32, 0.64, 0.8, 1.0]
y = [0.9412, 0.9268, 0.929, 0.9164, 0.9135, 0.8909, 0.8895, 0.8723, 0.8817]
log_x = np.log(x)
log_y = np.log(y)

coefficients = np.polyfit(log_x, y, 1)
print(coefficients)


c = coefficients[0] * np.log(x+[1.2, 1.5, 2]) + coefficients[1]

plt.figure(figsize=(16, 10))
plt.plot(x, y, "o")
plt.plot(x+[1.2, 1.5, 2], c, 'y--')
    
for idx, xi in enumerate(x+[1.2, 1.5, 2]):
    plt.annotate(round(c[idx], 4), xy = (xi, round(c[idx], 4)), xytext = (xi+0.001, c[idx]+0.001)) # 这里xy是需要标记的坐标，xytext是对应的标签坐标
    
plt.show()


