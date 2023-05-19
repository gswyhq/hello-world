#!/usr/bin/env python3
# -*- coding: utf-8 -*-


import cv2
import numpy as np
import os
from PIL import Image

USERNAME = os.getenv("USERNAME")


def replace(img_path):
    '''红色像素替换为白色'''
    for path in os.listdir(img_path):
        path2 = img_path + path + "/"
        for pathd in os.listdir(path2):
            path3 = path2 + pathd
            img2 = Image.open(path3)
            img2 = img2.convert('RGBA')  # 图像格式转为RGBA
            pixdata = img2.load()
            for y in range(img2.size[1]):
                for x in range(img2.size[0]):
                    if pixdata[x, y][0] > 220:  # 红色像素
                        pixdata[x, y] = (255, 255, 255, 255)  # 替换为白色，参数分别为(R,G,B,透明度)
            img2 = img2.convert('RGB')  # 图像格式转为RGB
            print("替换文件", pathd)
            img2.save(path3)


def separate_color_red(img):
    # 颜色提取
    # gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)  # 色彩空间转换为hsv，便于分离
    lower_hsv = np.array([0, 43, 46])  # 提取颜色的低值
    high_hsv = np.array([10, 255, 255])  # 提取颜色的高值
    mask = cv2.inRange(hsv, lowerb=lower_hsv, upperb=high_hsv)
    print("颜色提取完成")
    return mask


def salt(img, n):
    # 椒盐去燥
    for k in range(n):
        i = int(np.random.random() * img.shape[1])
        j = int(np.random.random() * img.shape[0])
        if img.ndim == 2:
            img[j, i] = 255
        elif img.ndim == 3:
            img[j, i, 0] = 255
            img[j, i, 1] = 255
            img[j, i, 2] = 255
        print("去燥完成")
        return img


def show(name, img):
    # 显示图片
    cv2.namedWindow(str(name), cv2.WINDOW_NORMAL)
    cv2.resizeWindow(str(name), 800, 2000)  # 改变窗口大小
    cv2.imshow(str(name), img)


def lines(img):
    # 直线检测
    img2 = cv2.Canny(img, 20, 250)  # 边缘检测
    line = 4
    minLineLength = 50
    maxLineGap = 150
    # HoughLinesP函数是概率直线检测，注意区分HoughLines函数
    lines = cv2.HoughLinesP(img2, 1, np.pi / 180, 120, lines=line, minLineLength=minLineLength, maxLineGap=maxLineGap)
    if lines is None:
        return img
    lines1 = lines[:, 0, :]  # 降维处理
    # line 函数勾画直线
    # (x1,y1),(x2,y2)坐标位置
    # (0,255,0)设置BGR通道颜色
    # 2 是设置颜色粗浅度
    for x1, y1, x2, y2 in lines1:
        cv2.line(img, (x1, y1), (x2, y2), (255, 255, 255), 2)
    return img


def contour(img):
    # 检测轮廓
    # ret, thresh = cv2.threshold(cv2.cvtColor(img, 127, 255, cv2.THRESH_BINARY))
    box, angle = None, None
    contours, hier = cv2.findContours(img, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    for c in contours:  # 遍历轮廓
        rect = cv2.minAreaRect(c)  # 生成最小外接矩形
        box_ = cv2.boxPoints(rect)
        h = abs(box_[3, 1] - box_[1, 1])
        w = abs(box_[3, 0] - box_[1, 0])
        print("宽，高", w, h)
        # 只保留需要的轮廓
        if (h > 3000 or w > 2200):
            continue
        if (h < 2500 or w < 1500):
            continue
        box = cv2.boxPoints(rect)  # 计算最小面积矩形的坐标
        box = np.int0(box)  # 将坐标规范化为整数
        angle = rect[2]  # 获取矩形相对于水平面的角度
        if angle > 0:
            if abs(angle) > 45:
                angle = 90 - abs(angle)
        else:
            if abs(angle) > 45:
                angle = (90 - abs(angle))
        # 绘制矩形
        # cv2.drawContours(img, [box], 0, (255, 0, 255), 3)
    print("轮廓数量", len(contours))
    return img, box, angle


def rotate(img, angle):
    # 旋转图片
    (h, w) = img.shape[:2]  # 获得图片高，宽
    center = (w // 2, h // 2)  # 获得图片中心点
    img_ratete = cv2.getRotationMatrix2D(center, angle, 1)
    rotated = cv2.warpAffine(img, img_ratete, (w, h))
    return rotated


def cut1(img, box):
    # 从轮廓出裁剪图片
    x1, y1 = box[1]  # 获取左上角坐标
    x2, y2 = box[3]  # 获取右下角坐标
    img_cut = img[y1 + 10:y2 - 10, x1 + 10:x2 - 10]  # 切片裁剪图像
    return img_cut


def cut2(img, out_path, filed):
    # 裁剪方格中图像并保存
    # ＠config i_:i为裁剪图像的y坐标区域, j_:j为裁剪图像的x坐标区域

    if not os.path.isdir(out_path):  # 创建文件夹
        os.makedirs(out_path)
    h, w, _ = img.shape  # 获取图像通道
    print(h, w)
    s, i_ = 0, 0
    # 循环保存图像
    for i in range(h // 12, h, h // 12):
        j_ = 0
        for j in range(w // 8, w, w // 8):
            imgd = img[i_ + 5:i - 5, j_ + 5:j - 5]
            # img_list.append(img[i_+10:i-10,j_+10:j-10])
            out_pathd = out_path + filed[:-4] + "_" + str(s) + ".jpg"  # 图像保存路径
            cv2.imwrite(out_pathd, imgd)
            print("保存文件", out_pathd)
            s += 1
            j_ = j
        i_ = i


def test():
    img_path = rf"D:\Users\{USERNAME}\github_project\PJ_PREDICT_IMG\68747.png"
    out_path = rf"D:\Users\{USERNAME}\github_project\PJ_PREDICT_IMG\68747_cut"
    img = cv2.imread(img_path)  # 读取图像
    img_separate = separate_color_red(img)  # 提取红色框先
    mediu = cv2.medianBlur(img_separate, 19)  # 中值滤波,过滤除最外层框线以外的线条
    img_lines = lines(mediu)  # 直线检测，补充矩形框线
    img_contours, box, angle = contour(img_lines)  # 轮廓检测，获取最外层矩形框的偏转角度
    print("角度", angle, "坐标", box)
    img_rotate = rotate(img_lines, angle)  # 旋转图像至水平方向
    img_contours, box, _ = contour(img_rotate)  # 获取水平矩形框的坐标

    img_original_rotate = rotate(img, angle)  # 旋转原图至水平方向
    img_original_cut = cut1(img_original_rotate, box)  # 通过图像坐标从外层矩形框处裁剪原图
    cut2(img_original_cut, out_path, filed)  # 裁剪方格保存
    replace(put_path)  # 消除多余红色框线
    # show("img",img)
    # show("img_separate", img_separate)
    # show("mediu", mediu)
    # show("img_lines", img_lines)
    # show("img_contours", img_contours)
    # show("img__rotate",img_rotate)
    # show("img_original_cut",img_original_cut)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

################################################# 过滤筛选方框类型的轮廓 #######################################################################
import cv2
import itertools

def split_squares(contours, thr=5):
    '''将不是方框的轮廓进行分割
    方法步骤：1、忽略轮廓顶点＜4；
    2、顶点数为4的直接保留；
    3、像素位置不超过阈值的点合并；
    4、根据合并保留点，进行方框分割
    '''

    def unique_point(cnt, thr=5):
        '''对轮廓顶点进行去重，即小于阈值的点合并'''
        cnt = sorted(cnt.tolist(), key=lambda x: x[0][0] + x[0][1])
        new_cnt = []
        for c in cnt:
            w, h = c[0]
            if len(new_cnt) > 0 and (abs(new_cnt[-1][0][0]-w) <= thr and abs(new_cnt[-1][0][1]-h) <= thr):
                new_cnt[-1] = [[w, h]]
            else:
                new_cnt.append([[w, h]])
        return new_cnt

    def split_cnt(cnt, thr=5):
        cnt = unique_point(cnt, thr=thr)  # 像素小于阈值的顶点合并；
        x_list = sorted(set([t[0][0] for t in cnt]))
        y_list = sorted(set([t[0][1] for t in cnt]))
        new_x_list = x_list[:1]
        for x in x_list[1:]:
            if abs(new_x_list[-1] - x) <= thr:
                new_x_list[-1] = x
            else:
                new_x_list.append(x)
        new_y_list = y_list[:1]
        for y in y_list[1:]:
            if abs(new_y_list[-1] - y) <= thr:
                new_y_list[-1] = y
            else:
                new_y_list.append(y)

        new_cnt = []
        for idx, x2 in enumerate(new_x_list[1:], 1):
            x1 = new_x_list[idx-1]
            for idy, y2 in enumerate(new_y_list[1:], 1):
                y1 = new_y_list[idy-1]
                new_cnt.append([[[x1, y1]], [[x1, y2]], [[x2, y2]], [[x2, y1]]])
        return new_cnt

    new_contours = []
    for cnt in contours:
        if len(cnt) < 4:
            continue
        elif len(cnt) == 4:
            new_contours.append(cnt.tolist())
        else:
            new_contours.extend(split_cnt(cnt, thr=thr))
    return np.array(new_contours, dtype=np.int32)

def angle_cos(p0, p1, p2):
    '''计算两边夹角额cos值'''
    d1, d2 = (p0-p1).astype('float'), (p2-p1).astype('float')
    return abs( np.dot(d1, d2) / np.sqrt( np.dot(d1, d1)*np.dot(d2, d2) ) )


def find_squares(img):
    '''寻找图像中的方框轮廓'''
    squares = []
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    ret, binary = cv2.threshold(gray, 127, 255, cv2.THRESH_BINARY)
    contours, _hierarchy = cv2.findContours(binary, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    print("轮廓数量：%d" % len(contours))
    contours = split_squares(contours, thr=5)  # 对方框轮廓重新拆分
    print("对方框轮廓重新拆分后的轮廓数量：%d" % len(contours))
    index = 0
    # 轮廓遍历
    for cnt in contours:
        cnt_len = cv2.arcLength(cnt, True)  # 计算轮廓周长
        cnt = cv2.approxPolyDP(cnt, 0.02 * cnt_len, True)  # 多边形逼近
        # 条件判断逼近边的数量是否为4，轮廓面积是否大于1000，检测轮廓是否为凸的
        # 通过cv2.contourArea可以求得轮廓面积
        # isContourConvex(contours)是用来判断某个轮廓是不是一个凸轮廓，其返回的是一个布尔值，当输入轮廓是凸轮廓时返回true，不是凸轮廓时返回false。
        if len(cnt) == 4 and cv2.contourArea(cnt) > 1000 and cv2.isContourConvex(cnt):
            M = cv2.moments(cnt)  # 计算轮廓的矩
            cx = int(M['m10'] / M['m00'])
            cy = int(M['m01'] / M['m00'])  # 轮廓重心

            cnt = cnt.reshape(-1, 2)
            max_cos = np.max([angle_cos(cnt[i], cnt[(i + 1) % 4], cnt[(i + 2) % 4]) for i in range(4)])
            # 只检测矩形（cos90° = 0）
            if max_cos < 0.1:
                # 检测四边形（不限定角度范围）
                # if True:
                index = index + 1
                cv2.putText(img, ("#%d" % index), (cx, cy), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 0, 255), 2)
                squares.append(cnt)
    return squares, img

def test_squares():
    img_path = rf"D:\Users\{USERNAME}\github_project\PJ_PREDICT_IMG\68747.png"
    img = cv2.imread(img_path)
    squares, img = find_squares(img)
    cv2.drawContours(img, squares, -1, (0, 0, 255), 2)
    cv2.imwrite(rf"D:\Users\{USERNAME}\github_project\PJ_PREDICT_IMG\68747_squares.png", img)

########################################################################################################################

import cv2
import numpy as np

#定义形状检测函数
def ShapeDetection(img):
    contours,hierarchy = cv2.findContours(img, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)  #寻找轮廓点
    for obj in contours:
        area = cv2.contourArea(obj)  #计算轮廓内区域的面积
        if area <= 1000 or not cv2.isContourConvex(obj):
            continue
        cv2.drawContours(imgContour, obj, -1, (255, 0, 0), 4)  #绘制轮廓线
        perimeter = cv2.arcLength(obj,True)  #计算轮廓周长
        approx = cv2.approxPolyDP(obj,0.02*perimeter,True)  #获取轮廓角点坐标
        CornerNum = len(approx)   #轮廓角点的数量
        x, y, w, h = cv2.boundingRect(approx)  #获取坐标值和宽度、高度

        #轮廓对象分类
        if CornerNum ==3: objType ="triangle"
        elif CornerNum == 4:
            if w==h: objType= "Square"
            else:objType="Rectangle"
        elif CornerNum>4: objType= "Circle"
        else:objType="N"

        cv2.rectangle(imgContour,(x,y),(x+w,y+h),(0,0,255),2)  #绘制边界框
        cv2.putText(imgContour,objType,(x+(w//2),y+(h//2)),cv2.FONT_HERSHEY_COMPLEX,0.6,(0,0,0),1)  #绘制文字

path = rf"D:\Users\{USERNAME}\github_project\PJ_PREDICT_IMG\68747.png"
img = cv2.imread(path)
imgContour = img.copy()

imgGray = cv2.cvtColor(img,cv2.COLOR_RGB2GRAY)  #转灰度图
imgBlur = cv2.GaussianBlur(imgGray,(5,5),1)  #高斯模糊
imgCanny = cv2.Canny(imgBlur,60,60)  #Canny算子边缘检测
ShapeDetection(imgCanny)  #形状检测


cv2.imshow("imgCanny", imgCanny)
cv2.imshow("shape Detection", imgContour)

cv2.waitKey(0)

#####################################################################################################################################
img_path = rf"D:\Users\{USERNAME}\github_project\PJ_PREDICT_IMG\68747.png"
img = cv2.imread(img_path)
gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
ret,binary = cv2.threshold(gray, 127, 255, cv2.THRESH_BINARY)

contours, hierarchy = cv2.findContours(binary, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
# 其中，cv2.findContours() 的第二个参数主要有
# cv2.RETR_LIST：检测的轮廓不建立等级关系
# cv2.RETR_TREE：L建立一个等级树结构的轮廓。
# cv2.RETR_CCOMP：建立两个等级的轮廓，上面的一层为外边界，里面的一层为内孔的边界信息。
# cv2.RETR_EXTERNAL：表示只检测外轮廓
# cv2.findContours() 的第三个参数 method为轮廓的近似办法
# cv2.CHAIN_APPROX_NONE存储所有的轮廓点，相邻的两个点的像素位置差不超过1，即max（abs（x1-x2），abs（y2-y1））==1
# cv2.CHAIN_APPROX_SIMPLE压缩水平方向，垂直方向，对角线方向的元素，只保留该方向的终点坐标，例如一个矩形轮廓只需4个点来保存轮廓信息
# cv2.CHAIN_APPROX_TC89_L1，CV_CHAIN_APPROX_TC89_KCOS使用teh-Chinl chain 近似算法

# 返回值
# contour返回值
# cv2.findContours()函数首先返回一个list，list中每个元素都是图像中的一个轮廓，用numpy中的ndarray表示。若是方框的话，对应坐标左上为零点，依次为：左上, 左下, 右下, 右上，共四个顶点;
# hierarchy返回值
# 该函数还可返回一个可选的hiararchy结果，这是一个ndarray，其中的元素个数和轮廓个数相同，每个轮廓contours[i]对应4个hierarchy元素hierarchy[i][0] ~hierarchy[i][3]，分别表示后一个轮廓、前一个轮廓、父轮廓、内嵌轮廓的索引编号，如果没有对应项，则该值为负数。

# 绘制轮廓
draw_img0 = cv2.drawContours(img.copy(),contours,0,(0,255,255),3)
draw_img1 = cv2.drawContours(img.copy(),contours,1,(255,0,255),3)
draw_img2 = cv2.drawContours(img.copy(),contours,2,(255,255,0),3)
draw_img3 = cv2.drawContours(img.copy(), contours, -1, (0, 0, 255), 3)

# cv2.drawContours()函数
# cv2.drawContours(image, contours, contourIdx, color[, thickness[, lineType[, hierarchy[, maxLevel[, offset ]]]]])
#
# 第一个参数是指明在哪幅图像上绘制轮廓；
# 第二个参数是轮廓本身，在Python中是一个list。
# 第三个参数指定绘制轮廓list中的哪条轮廓，如果是-1，则绘制其中的所有轮廓。后面的参数很简单。其中thickness表明轮廓线的宽度，如果是-1（cv2.FILLED），则为填充模式。绘制参数将在以后独立详细介绍。
# 为了看到自己画了哪些轮廓，可以使用 cv2.boundingRect()函数获取轮廓的范围，即左上角原点，以及他的高和宽。然后用cv2.rectangle()方法画出矩形轮廓
#
"""
x, y, w, h = cv2.boundingRect(img)   
    参数：
    img  是一个二值图
    x，y 是矩阵左上点的坐标，
    w，h 是矩阵的宽和高

cv2.rectangle(img, (x,y), (x+w,y+h), (0,255,0), 2)
    img：       原图
    (x，y）：   矩阵的左上点坐标
    (x+w，y+h)：是矩阵的右下点坐标
    (0,255,0)： 是画线对应的rgb颜色
    2：         线宽
    
for i in range(0,len(contours)):  
    x, y, w, h = cv2.boundingRect(contours[i])   
    cv2.rectangle(image, (x,y), (x+w,y+h), (153,153,0), 5)   
    
获取物体最小外界矩阵
使用 cv2.minAreaRect(cnt) ，返回点集cnt的最小外接矩形，cnt是所要求最小外接矩形的点集数组或向量，这个点集不定个数。
其中：cnt = np.array([[x1,y1],[x2,y2],[x3,y3],[x4,y4]]) # 必须是array数组的形式

rect = cv2.minAreaRect(cnt) # 得到最小外接矩形的（中心(x,y), (宽,高), 旋转角度）
box = np.int0(cv2.boxPoints(rect)) #通过box会出矩形框      
"""

print ("contours:类型：",type(contours))
print ("第0 个contours:",type(contours[0]))
print ("contours 数量：",len(contours))

print ("contours[0]点的个数：",len(contours[0]))
print ("contours[1]点的个数：",len(contours[1]))

cv2.imshow("img", img)
cv2.imshow("draw_img0", draw_img0)
cv2.imshow("draw_img1", draw_img1)
cv2.imshow("draw_img2", draw_img2)
cv2.imshow("draw_img3", draw_img3)

cv2.waitKey(0)
cv2.destroyAllWindows()

def main():
    test()


if __name__ == '__main__':
    main()

#
# [[175.0, 564.0], [584.0, 564.0], [584.0, 599.0], [175.0, 599.0]]
# [[107.0, 619.0], [280.0, 619.0], [280.0, 654.0], [107.0, 654.0]]

# array([[[   0,    0]],
#         [[   0, 1510]],
#         [[2334, 1510]],
#         [[2334,    0]]], dtype=int32)
# 1510, 上下一半；
# 左上为零点，依次为：左上, 左下, 右下, 右上;

# [[[102, 557]],
#  [[102, 1030]],
#  [[679, 558]],
#  [[848, 558]],
#  [[975, 557]],
#  [[679, 1030]],
#  [[848, 1030]],
#  [[975, 1030]]]