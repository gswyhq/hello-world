#!/usr/bin/env python3
# -*- coding: utf-8 -*-

# 均衡化的目的是将非均匀分布变成均匀分布
# 直方图均衡化的目的是将原始图像的直方图变为均衡分布的的形式，将一非均匀灰度概率密度分布图像，通过寻求某种灰度变换，变成一幅具有均匀概率密度分布的目的图像。

# 直方图均衡化：
#         直方图均衡化是使图像直方图变得平坦的操作。直方图均衡化能够有效地解决图像整体过暗、过亮的问题，增加图像的清晰度。
#         具体流程如下所示。其中S是总的像素数，Zmax是像素的最大取值(8位灰度图像为255)，h(i)为图像像素取值为 i 及 小于 i 的像素的总数。
# 直方图均衡化实质上是减少图像的灰度级来加大对比度，图像经均衡化处理之后，图像变得清晰，直方图中每个像素点的灰度级减少，但分布更加均匀，对比度更高。

# 但直方图均衡化技术仍存在如下缺点：
# （1）将原始函数的累积分布函数作为变换函数，只能产生近似均匀的直方图。
# （2）在某些情况下，并不一定需要具有均匀直方图的图像。

import cv2
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# histogram equalization
def hist_equal(img, z_max=255):
    H, W = img.shape
    # S is the total of pixels
    S = H * W  * 1.

    out = img.copy()

    sum_h = 0.

    for i in range(1, 255):
        ind = np.where(img == i)
        sum_h += len(img[ind])
        z_prime = z_max / S * sum_h
        out[ind] = z_prime

    out = out.astype(np.uint8)

    return out


# Read image
img = cv2.imread("../head_g_n.jpg",0).astype(np.float)

# histogram normalization
out = hist_equal(img)

# Display histogram
plt.hist(out.ravel(), bins=255, rwidth=0.8, range=(0, 255))
plt.savefig("out_his.png")
plt.show()

# Save result
cv2.imshow("result", out)
cv2.imwrite("out.jpg", out)
cv2.waitKey(0)
cv2.destroyAllWindows()

############################################################################################################################
def histogram_normalization(value_counts):
    '''直方图均衡化
    将不均衡分布的数据，生成对应的映射字典，使之转换为0-1之间均衡分布的数据'''
    z_max = max(value_counts.keys()) # Zmax是像素的最大取值
    S = sum(value_counts.values()) # 总的像素数，
    out = {}
    sum_h = 0.
    for i, ind in sorted(value_counts.items(), key=lambda x: x[0]):
        sum_h += ind
        z_prime = z_max / S * sum_h
        out[i] = z_prime
    return out

df = pd.DataFrame(np.array([0.841233  , 0.88738644, 0.8170942 , 0.8716165 , 0.9231366 ,
       0.81411755, 0.91710156, 0.70177686, 0.88332075, 0.62443334,
       0.7723776 , 0.71895975, 0.91075605, 0.93808734, 0.532212  ,
       0.86577326, 0.84620273, 0.7205562 , 0.80210435, 0.90355676,
       0.69313884, 0.77566993, 0.66142875, 0.89476097, 0.8559841 ,
       0.73479533, 0.8171928 , 0.7934405 , 0.93359643, 0.80570877,
       0.8039533 , 0.83225685, 0.88607836, 0.7703523 , 0.9316197 ,
       0.8889308 , 0.8178044 , 0.9142437 , 0.66317445, 0.47521225,
       0.926416  , 0.8567897 , 0.7335771 , 0.84692186, 0.7146002 ,
       0.75015545, 0.8430602 , 0.8351139 , 0.8055599 , 0.7521103 ]), columns=['pred'])
img = dict(df['pred'].value_counts().items())
out = histogram_normalization(img)

# 原始数据计算分位数
np.percentile(df['pred'].values, np.array([2.5, 25, 50, 75, 97.5]))
Out[60]: array([0.5529618 , 0.75064416, 0.8174986 , 0.88538896, 0.93315167])

# 直方图均衡化处理后，计算分位数
np.percentile([out[t] for t in df['pred'].values], np.array([2.5, 25, 50, 75, 97.5]))
Out[61]: array([0.04174489, 0.24859315, 0.47842454, 0.70825594, 0.9151042 ])

def main():
    pass


if __name__ == '__main__':
    main()
