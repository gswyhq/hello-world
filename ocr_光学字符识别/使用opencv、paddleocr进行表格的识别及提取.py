#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os
import cv2
import pandas as pd
import numpy as np

from paddleocr import PaddleOCR

ocr = PaddleOCR(use_angle_cls=True, lang="ch")


class TableOCR(object):
    def __init__(self, result_file=''):
        self.img_file = ''
        if not result_file:
            self.result_file = 'result.xlsx'

    def get_sorted_rect(self, rect):
        '''
        获取排序的四个坐标
        @param rect:
        @return:按照左上 右上 右下 左下排列返回
        '''
        mid_x = (max([x[1] for x in rect]) - min([x[1] for x in rect])) * 0.5 + min([x[1] for x in rect])  # 中间点坐标
        left_rect = [x for x in rect if x[1] < mid_x]
        left_rect.sort(key=lambda x: (x[0], x[1]))
        right_rect = [x for x in rect if x[1] > mid_x]
        right_rect.sort(key=lambda x: (x[0], x[1]))
        sorted_rect = left_rect[0], left_rect[1], right_rect[1], right_rect[0]

        return sorted_rect

    def get_table(self, gray, min_table_area=0):
        '''
        从灰度图获取表格坐标，[[右下→左下→左上→右上],..]
        边缘检测+膨胀---》找最外围轮廓点，根据面积筛选---》根据纵坐标排序---》计算轮廓的四个点，再次筛选
        @param gray:灰度图 如果是二值图会报错
        @return:
        '''

        canny = cv2.Canny(gray, 200, 255)  # 第一个阈值和第二个阈值
        kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))
        canny = cv2.dilate(canny, kernel)

        contours, HIERARCHY = cv2.findContours(canny, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        if not min_table_area:
            min_table_area = gray.shape[0] * gray.shape[1] * 0.01  # 50000  # 最小的矩形面积阈值
        candidate_table = [cnt for cnt in contours if cv2.contourArea(cnt) > min_table_area]  # 计算该轮廓的面积
        candidate_table = sorted(candidate_table, key=cv2.contourArea, reverse=True)

        area_list = [cv2.contourArea(cnt) for cnt in candidate_table]

        table = []
        for i in range(len(candidate_table)):
            # 遍历所有轮廓
            # cnt是一个点集
            cnt = candidate_table[i]
            # 找到最小的矩形，该矩形可能有方向
            rect = cv2.minAreaRect(cnt)
            # box是四个点的坐标
            box = cv2.boxPoints(rect)
            box = np.int0(box)
            sorted_box = self.get_sorted_rect(box)

            result = [sorted_box[2], sorted_box[3], sorted_box[0], sorted_box[1]]  # 右下 左下 左上 右上

            result = [x.tolist() for x in result]
            table.append(result)

        return table

    def perTran(self, image, rect):
        '''
        做透视变换
        image 图像
        rect  四个顶点位置:左上 右上 右下 左下
        '''

        tl, tr, br, bl = rect  # 左下 右下  左上 右上 || topleft topright 左上 右上 右下 左下
        # 计算宽度
        widthA = np.sqrt(((br[0] - bl[0]) ** 2) + ((br[1] - bl[1]) ** 2))
        widthB = np.sqrt(((tr[0] - tl[0]) ** 2) + ((tr[1] - tl[1]) ** 2))
        maxWidth = max(int(widthA), int(widthB))
        # 计算高度
        heightA = np.sqrt(((tr[0] - br[0]) ** 2) + ((tr[1] - br[1]) ** 2))
        heightB = np.sqrt(((tl[0] - bl[0]) ** 2) + ((tl[1] - bl[1]) ** 2))
        maxHeight = max(int(heightA), int(heightB))
        # 定义变换后新图像的尺寸
        dst = np.array([[0, 0], [maxWidth - 1, 0], [maxWidth - 1, maxHeight - 1],
                        [0, maxHeight - 1]], dtype='float32')
        # 变换矩阵
        rect = np.array(rect, dtype=np.float32)
        dst = np.array(dst, dtype=np.float32)
        M = cv2.getPerspectiveTransform(rect, dst)
        # 透视变换
        warped = cv2.warpPerspective(image, M, (maxWidth, maxHeight))
        return warped

    def recognize_bgkx(self, binary):
        rows, cols = binary.shape

        scale = 30  # 值越小 横线越少 40
        # 自适应获取核值
        # 识别横线:
        kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (cols // scale, 1))
        # 矩形

        eroded = cv2.erode(binary, kernel, iterations=1)  # 腐蚀
        dilated_col = cv2.dilate(eroded, kernel, iterations=1)  # 膨胀
        # cv2.imshow("excel_horizontal_line", dilated_col)
        # cv2.waitKey()

        # 识别竖线：
        scale = 20
        kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (1, rows // scale))
        eroded = cv2.erode(binary, kernel, iterations=1)
        dilated_row = cv2.dilate(eroded, kernel, iterations=1)
        # cv2.imshow("excel_vertical_line：", dilated_row)
        # cv2.waitKey()

        # 将识别出来的横竖线合起来 对二进制数据进行“与”操作
        bitwise_and = cv2.bitwise_and(dilated_col, dilated_row)
        # cv2.imshow("excel_bitwise_and", bitwise_and)
        # cv2.waitKey()

        # 标识表格轮廓
        # merge = cv2.add(dilated_col, dilated_row)  # 进行图片的加和
        # cv2.imshow("entire_excel_contour：", merge)
        # cv2.waitKey()

        # 两张图片进行减法运算，去掉表格框线
        # merge2 = cv2.subtract(binary, merge)
        # cv2.imshow("binary_sub_excel_rect", merge2)
        # cv2.waitKey()

        # new_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (2, 2))
        # erode_image = cv2.morphologyEx(merge2, cv2.MORPH_OPEN, new_kernel)
        # cv2.imshow('erode_image2', erode_image)
        # cv2.waitKey()

        # merge3 = cv2.add(erode_image, bitwise_and)
        # cv2.imshow('merge3', merge3)
        # cv2.waitKey()

        # 将焦点标识取出来
        ys, xs = np.where(bitwise_and > 0)

        # 横纵坐标数组
        y_point_arr = []
        x_point_arr = []
        # 通过排序，排除掉相近的像素点，只取相近值的最后一点
        # 这个10就是两个像素点的距离，不是固定的，根据不同的图片会有调整，基本上为单元格表格的高度（y坐标跳变）和长度（x坐标跳变）
        i = 0

        sort_x_point = np.sort(list(set(xs)))
        for i in range(len(sort_x_point) - 1):
            if sort_x_point[i + 1] - sort_x_point[i] > 10:
                x_point_arr.append(sort_x_point[i])
            i = i + 1
        # 要将最后一个点加入
        x_point_arr.append(sort_x_point[i])

        i = 0
        sort_y_point = np.sort(list(set(ys)))
        # print(np.sort(ys))
        for i in range(len(sort_y_point) - 1):
            if (sort_y_point[i + 1] - sort_y_point[i] > 10):
                y_point_arr.append(sort_y_point[i])
            i = i + 1
        y_point_arr.append(sort_y_point[i])

        return x_point_arr, y_point_arr

    def recognize_text_by_loop(self, gray, x_point_arr, y_point_arr):
        # y_point_arr = [x - 3 for x in y_point_arr]
        # 循环y坐标，x坐标分割表格
        data = [[] for i in range(len(y_point_arr))]
        for i in range(len(y_point_arr) - 1):
            # if i==0:
            #     continue
            for j in range(len(x_point_arr) - 1):
                # 在分割时，第一个参数为y坐标，第二个参数为x坐标
                cell = gray[
                       y_point_arr[i]:y_point_arr[i + 1],
                       x_point_arr[j]:x_point_arr[j + 1]
                       ]
                # cv2.imshow("sub_pic" + str(i) + str(j), cell)
                # cv2.waitKey()
                # cv2.destroyAllWindows()
                # img_path = "cell_image_" + str(i) + '_' + str(j) + ".png"
                # cv2.imwrite(img_path, cell)
                # 输入待识别图片路径

                # 输出结果保存路径
                result = ocr.ocr(cell)
                print('result', result)
                text1 = ''.join([x[0][1][0] for x in result if x])
                print(text1)
                data[i].append(text1)
        print(data)
        df = pd.DataFrame(data[1:-1], columns=data[0])
        return df

    def ocr(self, img_file):
        self.img_file = img_file
        img = cv2.imread(img_file)
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        rect = self.get_table(gray)
        sorted_rect = self.get_sorted_rect(rect[0])
        gray_z = self.perTran(gray, sorted_rect)  # 如果有多个table 可以循环
        cv2.imwrite('gray_z.jpg', gray_z)
        binary_z = cv2.adaptiveThreshold(~gray_z, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
                                         cv2.THRESH_BINARY, 15, -5)
        x_point_arr, y_point_arr = self.recognize_bgkx(binary_z)
        df = self.recognize_text_by_loop(gray_z, x_point_arr, y_point_arr)
        df.to_excel(self.result_file, index=False)



def main():
    USERNAME = os.getenv("USERNAME")

    img_path = rf"D:\Users\{USERNAME}\github_project\PJ_PREDICT_IMG\68747.png"
    tableOCR = TableOCR()
    tableOCR.ocr(img_path)
    print('结果保存在：', os.path.abspath(tableOCR.result_file))

    # 原文链接：https: // blog.csdn.net / weixin_38235865 / article / details / 125951185


if __name__ == '__main__':
    main()
