#!/usr/bin/python3
# coding: utf-8

import matplotlib.pyplot as plt
import numpy as np

{(0.668754957029771, 0.8999576010351087): 0.7843562790324399,
 (0.6761704738083262, 0.8645611453894139): 0.77036580959887,
 (0.7910324658957225, 0.9293400287936777): 0.9293400287936777,
 (0.8211349849630077, 0.9221173841261299): 0.8716261845445687,
 (0.8219881453287553, 0.9093923297340155): 0.8656902375313854,
 (0.8542550348947021, 0.9195152116739612): 0.9195152116739612,
 (0.8551091647248069, 0.9255374365032568): 0.9255374365032568,
 (0.9393115336098808, 0.785219193468478): 0.785219193468478,
 (0.9430897283965576, 0.8743074075456718): 0.9430897283965576,
 (0.9810906572995571, 0.8028890311324118): 0.8919898442159845}

# 多项式拟合范例：

x = np.arange(1, 17, 1)
y = np.array([4.00, 6.40, 8.00, 8.80, 9.22, 9.50, 9.70, 9.86, 10.00, 10.20, 10.32, 10.42, 10.50, 10.55, 10.58, 10.60])
z1 = np.polyfit(x, y, 3) # 用3次多项式拟合
p1 = np.poly1d(z1)
print(p1) #在屏幕上打印拟合多项式
# poly1d([ 0.00624491, -0.20371114,  2.18193147,  2.57208791])
# 0.00624491*x**3 -0.20371114*x**2+ 2.18193147*x+2.57208791
yvals=p1(x)#也可以使用yvals=np.polyval(z1,x)
plot1=plt.plot(x, y, '*',label='original values')
plot2=plt.plot(x, yvals, 'r',label='polyfit values')
plt.xlabel('x axis')
plt.ylabel('y axis')
plt.legend(loc=4)#指定legend的位置,读者可以自己help它的用法
plt.title('polyfitting')
plt.show()
plt.savefig('p1.png')

# 2.指定函数拟合

#使用非线性最小二乘法拟合
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit
import numpy as np
#用指数形式来拟合
x = np.arange(1, 17, 1)
y = np.array([4.00, 6.40, 8.00, 8.80, 9.22, 9.50, 9.70, 9.86, 10.00, 10.20, 10.32, 10.42, 10.50, 10.55, 10.58, 10.60])
def func(x,a,b):
    return a*np.exp(b/x)
popt, pcov = curve_fit(func, x, y)
a=popt[0]#popt里面是拟合系数，读者可以自己help其用法
b=popt[1]
yvals=func(x,a,b)
plot1=plt.plot(x, y, '*',label='original values')
plot2=plt.plot(x, yvals, 'r',label='curve_fit values')
plt.xlabel('x axis')
plt.ylabel('y axis')
plt.legend(loc=4)#指定legend的位置,读者可以自己help它的用法
plt.title('curve_fit')
plt.show()
plt.savefig('p2.png')


# 三维拟合：
import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit
def fitFunc(x, a, b, c, d):
    return a + b*x[0] + c*x[1] + d*x[0]*x[1]
x_3d = np.array([[1,2,3,4,6],[4,5,6,7,8]])
p0 = [5.11, 3.9, 5.3, 2]
fitParams, fitCovariances = curve_fit(fitFunc, x_3d, x_3d[1,:], p0)

# 多元线性回归：
from sklearn.linear_model import LinearRegression
X = [[6, 2], [8, 1], [10, 0], [14, 2], [18, 0]]
y = [[7], [9], [13], [17.5], [18]]
model = LinearRegression()
model.fit(X, y)
X_test = [[8, 2], [9, 0], [11, 2], [16, 2], [12, 0]]
y_test = [[11], [8.5], [15], [18], [11]]
predictions = model.predict(X_test)
for i, prediction in enumerate(predictions):
    print('Predicted: %s, Target: %s' % (prediction, y_test[i]))
print('R-squared: %.2f' % model.score(X_test, y_test))


def main():
    pass


if __name__ == '__main__':
    main()