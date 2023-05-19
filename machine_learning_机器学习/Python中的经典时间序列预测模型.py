#!/usr/bin/env python3
# -*- coding: utf-8 -*-

# 本文主要参考11种经典时间序列预测方法
# 自回归 (Holt Winter, AR)
# 移动平均线 (Moving Average, MA)
# 自回归移动平均线 (Autoregressive Moving Average, ARMA)
# 自回归综合移动平均线 (Autoregressive Integrated Moving Average, ARIMA)
# 季节性自回归整合移动平均线 (Seasonal Autoregressive Integrated Moving-Average, SARIMA)
# 具有外生回归量的季节性自回归整合移动平均线 (Seasonal Autoregressive Integrated Moving-Average with Exogenous Regressors, SARIMAX)
# 向量自回归 (Vector Autoregression, VAR)
# 向量自回归移动平均 (Vector Autoregression Moving-Average, VARMA)
# 具有外源回归量的向量自回归移动平均值 (Vector Autoregression Moving-Average with Exogenous Regressors, VARMAX)
# 简单指数平滑 (Simple Exponential Smoothing, SES)
# 霍尔特·温特的指数平滑 (Holt Winter’s Exponential Smoothing, HWES)

# 来源：https://blog.csdn.net/cdfunlove/article/details/124076294
# https://blog.csdn.net/cdfunlove/article/details/124076370
# https://blog.csdn.net/cdfunlove/article/details/124076423?spm=1001.2014.3001.5502

Autoregression (AR)该方法适用于没有趋势和季节性成分的单变量时间序列。Autoregression (AR) 方法将序列中的下一步预测结果为先前时间步长观测值的线性函数。模型的符号：模型 p 的阶数作为 AR 函数的参数，即 AR(p)。例如，AR(1) 是一阶Autoregression  model（自回归模型）。
# AR example
from statsmodels.tsa.ar_model import AutoReg
from random import random
# contrived dataset
data = [x + random() for x in range(1, 100)]
# fit model
model = AutoReg(data, lags=1)
model_fit = model.fit()
# make prediction
yhat = model_fit.predict(len(data), len(data))
print(yhat)


Moving Average (MA)该方法适用于没有趋势和季节性成分的单变量时间序列。Moving Average (MA) 方法将序列中的下一步预测结果为来自先前时间步骤的平均过程的残差的线性函数。Moving Average模型不同于计算时间序列的移动平均。模型的符号：将模型 q 的阶指定为 MA 函数的参数，即 MA(q)。例如，MA(1) 是一阶Moving Average模型。
# MA example
from statsmodels.tsa.arima.model import ARIMA
from random import random
# contrived dataset
data = [x + random() for x in range(1, 100)]
# fit model
model = ARIMA(data, order=(0, 0, 1))
model_fit = model.fit()
# make prediction
yhat = model_fit.predict(len(data), len(data))
print(yhat)

Autoregressive Moving Average (ARMA)该方法适用于没有趋势和季节性成分的单变量时间序列。ARIMA 方法将序列中的下一步预测结果为先前时间步的观测值和残差的线性函数。它结合了AR 和MA 模型。模型的符号：  AR(p) 和 MA(q) 模型的阶数作为 ARMA 函数的参数，例如 ARMA(p, q)。ARIMA 模型可用于开发 AR 或 MA 模型。

from statsmodels.tsa.arima.model import ARIMA
from random import random
# contrived dataset
data = [random() for x in range(1, 100)]
# fit model
model = ARIMA(data, order=(2, 0, 1))
model_fit = model.fit()
# make prediction
yhat = model_fit.predict(len(data), len(data))
print(yhat)

Autoregressive Integrated Moving Average (ARIMA)该方法适用于有趋势且无季节性成分的单变量时间序列。ARIMA方法将序列中的下一步建模为先前时间步长的差分观测值和残差的线性函数。ARIMA结合了AR和MA模型以及序列的差分预处理步骤，使序列平稳。模型的符号：指定 AR(p)、I(d) 和 MA(q) 模型的阶数作为 ARIMA 函数的参数，例如 ARIMA(p, d, q)。ARIMA 模型还可用于开发 AR、MA 和 ARMA 模型。
# ARIMA example
from statsmodels.tsa.arima.model import ARIMA
from random import random
# contrived dataset
data = [x + random() for x in range(1, 100)]
# fit model
model = ARIMA(data, order=(1, 1, 1))
model_fit = model.fit()
# make prediction
yhat = model_fit.predict(len(data), len(data), typ='levels')
print(yhat)

Seasonal Autoregressive Integrated Moving-Average (SARIMA)该方法适用于具有趋势 且/或 季节性成分的单变量时间序列。SARIMA方法将序列中的下一步预测值为先前时间步长的差异观测值、误差、差异季节性观测值和季节性误差的线性函数。SARIMA将 ARIMA 模型与在季节性水平上执行相同的自回归、差分和移动平均建模的能力相结合。模型的符号：指定 AR(p)、I(d) 和 MA(q) 模型的阶数作为 ARIMA 函数和 AR(P)、I(D)、MA(Q) 和 m 的参数季节级别的参数，例如 SARIMA(p, d, q)(P, D, Q)m 其中“m”是每个季节（季节期间）的时间步长数。SARIMA 模型可用于开发 AR、MA、ARMA 和 ARIMA 模型。
# SARIMA example
from statsmodels.tsa.statespace.sarimax import SARIMAX
from random import random
# contrived dataset
data = [x + random() for x in range(1, 100)]
# fit model
model = SARIMAX(data, order=(1, 1, 1), seasonal_order=(0, 0, 0, 0))
model_fit = model.fit(disp=False)
# make prediction
yhat = model_fit.predict(len(data), len(data))
print(yhat)

Seasonal Autoregressive Integrated Moving-Average with Exogenous Regressors (SARIMAX)适用于具有趋势 且/或 季节性成分以及外生变量的单变量时间序列。SARIMAX 模型 是 SARIMA 模型的扩展，其中还包括外生变量的建模。外生变量也称为协变量，可以被认为是并行输入序列，它们在与原始序列相同的时间步长中进行观察。初级序列可被称为内源数据以将其与外源序列进行对比。外生变量的观测值在每个时间步直接包含在模型中，并且不以与主要内生序列相同的方式建模（例如作为 AR、MA 等过程）。SARIMAX 方法也可用于对包含外生变量的包含模型进行建模，例如 ARX、MAX、ARMAX 和 ARIMAX。
# SARIMAX example
from statsmodels.tsa.statespace.sarimax import SARIMAX
from random import random
# contrived dataset
data1 = [x + random() for x in range(1, 100)]
data2 = [x + random() for x in range(101, 200)]
# fit model
model = SARIMAX(data1, exog=data2, order=(1, 1, 1), seasonal_order=(0, 0, 0, 0))
model_fit = model.fit(disp=False)
# make prediction
exog2 = [200 + random()]
yhat = model_fit.predict(len(data1), len(data1), exog=[exog2])
print(yhat)

Vector Autoregression (VAR)该方法适用于没有趋势和季节性成分的多元时间序列。VAR 方法使用 AR 模型对每个时间序列的下一步进行建模。它是 AR 对多个并行时间序列的推广，例如多元时间序列。模型的符号：指定 AR(p) 模型的阶数作为 VAR 函数的参数，例如 VAR(p)。
# VAR example
from statsmodels.tsa.vector_ar.var_model import VAR
from random import random
# contrived dataset with dependency
data = list()
for i in range(100):
    v1 = i + random()
    v2 = v1 + random()
    row = [v1, v2]
    data.append(row)
# fit model
model = VAR(data)
model_fit = model.fit()
# make prediction
yhat = model_fit.forecast(model_fit.y, steps=1)
print(yhat)

Vector Autoregression Moving-Average (VARMA)该方法适用于没有趋势和季节性成分的多元时间序列。VARMA 方法使用 ARMA 模型对每个时间序列中的下一步进行建模。它是 ARMA 对多个并行时间序列的推广，例如多元时间序列。该模型的符号：指定 AR(p) 和 MA(q) 模型的阶数作为 VARMA 函数的参数，例如 VARMA(p, q)。VARMA 模型也可用于开发 VAR 或 VMA 模型。
from statsmodels.tsa.statespace.varmax import VARMAX
from random import random
# contrived dataset with dependency
data = list()
for i in range(100):
    v1 = random()
    v2 = v1 + random()
    row = [v1, v2]
    data.append(row)
# fit model
model = VARMAX(data, order=(1, 1))
model_fit = model.fit(disp=False)
# make prediction
yhat = model_fit.forecast()
print(yhat)

Vector Autoregression Moving-Average with Exogenous Regressors (VARMAX)该方法适用于具有外生变量的没有趋势和季节性成分的多元时间序列。带有外生回归量的向量自回归移动平均 (VARMAX) 是 VARMA 模型的扩展，它还包括外生变量的建模。它是 ARMAX 方法的多元版本。外生变量也称为协变量，可以被认为是并行输入序列，它们在与原始序列相同的时间步长中进行观察。主要系列被称为内源数据，以将其与外源序列进行对比。外生变量的观测值在每个时间步直接包含在模型中，并且不以与主要内生序列相同的方式建模（例如作为 AR、MA 等过程）。VARMAX 方法也可用于对包含外生变量的包含模型进行建模，例如 VARX 和 VMAX。
# VARMAX example
from statsmodels.tsa.statespace.varmax import VARMAX
from random import random
# contrived dataset with dependency
data = list()
for i in range(100):
    v1 = random()
    v2 = v1 + random()
    row = [v1, v2]
    data.append(row)
data_exog = [x + random() for x in range(100)]
# fit model
model = VARMAX(data, exog=data_exog, order=(1, 1))
model_fit = model.fit(disp=False)
# make prediction
data_exog2 = [[100]]
yhat = model_fit.forecast(exog=data_exog2)
print(yhat)

Simple Exponential Smoothing (SES)该方法适用于没有趋势和季节性成分的单变量时间序列。简单指数平滑 (SES) 方法将下一个时间步预测结果为先前时间步观测值的指数加权线性函数。
# SES example
from statsmodels.tsa.holtwinters import SimpleExpSmoothing
from random import random
# contrived dataset
data = [x + random() for x in range(1, 100)]
# fit model
model = SimpleExpSmoothing(data)
model_fit = model.fit()
# make prediction
yhat = model_fit.predict(len(data), len(data))
print(yhat)


Holt Winter’s Exponential Smoothing (HWES)该方法适用于具有趋势 且/或 季节性成分的单变量时间序列。
HWES，也称为三重指数平滑法，将下一个时间步预测结果为先前时间步观测值的指数加权线性函数，同时考虑了趋势和季节性。
# HWES example
from statsmodels.tsa.holtwinters import ExponentialSmoothing
from random import random
# contrived dataset
data = [x + random() for x in range(1, 100)]
# fit model
model = ExponentialSmoothing(data)
model_fit = model.fit()
# make prediction
yhat = model_fit.predict(len(data), len(data))
print(yhat)

def main():
    pass


if __name__ == '__main__':
    main()