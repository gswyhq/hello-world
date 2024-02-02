#!/usr/bin/env python
# coding=utf-8

# 代码来源：https://blog.csdn.net/golden_knife/article/details/124434270
# 数据来源：https://github.com/Hareric/ClusterTool/blob/master/Other/c2.txt

import os
import numpy as np
from math import sqrt
import time
import matplotlib.pylab as pl

USERNAME = os.getenv("USERNAME")

# 定义一个簇单元
class ClusterUnit:
    def __init__(self):
        self.node_list = []  # 该簇存在的节点列表
        self.node_num = 0  # 该簇节点数
        self.centroid = None  # 该簇质心

    def add_node(self, node, node_vec):
        """
        为本簇添加指定节点，并更新簇心
         node_vec:该节点的特征向量
         node:节点
         return:null
        """
        self.node_list.append(node)
        try:
            self.centroid = (self.node_num * self.centroid + node_vec) / (self.node_num + 1)  # 更新簇心
        except TypeError:
            self.centroid = np.array(node_vec) * 1  # 初始化质心
        self.node_num += 1  # 节点数加1

    def remove_node(self, node):
        # 移除本簇指定节点
        try:
            self.node_list.remove(node)
            self.node_num -= 1
        except ValueError:
            raise ValueError("%s not in this cluster" % node)  # 该簇本身就不存在该节点，移除失败

    def move_node(self, node, another_cluster):
        # 将本簇中的其中一个节点移至另一个簇
        self.remove_node(node=node)
        another_cluster.add_node(node=node)


# cluster_unit = ClusterUnit()
# cluster_unit.add_node(1, [1, 1, 2])
# cluster_unit.add_node(5, [2, 1, 2])
# cluster_unit.add_node(3, [3, 1, 2])
# print(cluster_unit.centroid)


# 计算向量a与向量b的欧式距离
def euclidian_distance(vec_a, vec_b):
    diff = vec_a - vec_b
    return sqrt(np.dot(diff, diff))             # dot计算矩阵内积


class OnePassCluster:
    def __init__(self, t, vector_list):
        # t:一趟聚类的阈值
        self.threshold = t                      # 一趟聚类的阈值
        self.vectors = np.array(vector_list)    # 数据列表（向量列表）
        self.cluster_list = []                  # 聚类后簇的列表

        t1 = time.time()
        self.clustering()
        t2 = time.time()
        self.cluster_num = len(self.cluster_list)       # 聚类完成后 簇的个数
        self.spend_time = t2 - t1                       # 聚类花费的时间

    def clustering(self):
        self.cluster_list.append(ClusterUnit())                 # 初始新建一个簇
        self.cluster_list[0].add_node(0, self.vectors[0])       # 将读入的第一个节点归于该簇
        for index in range(len(self.vectors))[1:]:
            min_distance = euclidian_distance(vec_a=self.vectors[index],
                                              vec_b=self.cluster_list[0].centroid)  # 与簇的质心的最小距离
            min_cluster_index = 0  # 最小距离的簇的索引
            for cluster_index, cluster in enumerate(self.cluster_list[1:]):
                # enumerate会将数组或列表组成一个索引序列
                # 寻找距离最小的簇，记录下距离和对应的簇的索引
                distance = euclidian_distance(vec_a=self.vectors[index],
                                              vec_b=cluster.centroid)
                if distance < min_distance:
                    min_distance = distance
                    min_cluster_index = cluster_index + 1
            if min_distance < self.threshold:                   # 最小距离小于阈值，则归于该簇
                self.cluster_list[min_cluster_index].add_node(index, self.vectors[index])
            else:  # 否则新建一个簇
                new_cluster = ClusterUnit()
                new_cluster.add_node(index, self.vectors[index])
                self.cluster_list.append(new_cluster)
                del new_cluster

    def print_result(self, label_dict=None):
        # 打印出聚类结果
        # label_dict:节点对应的标签字典
        print("***********  single-pass的聚类结果展示  ***********")
        for index, cluster in enumerate(self.cluster_list):
            print("cluster:%s" % index)         # 簇的序号
            print(cluster.node_list)            # 该簇的节点列表
            if label_dict is not None:
                print(" ".join([label_dict[n] for n in cluster.node_list]))     # 若有提供标签字典，则输出该簇的标签
            print("node num: %s" % cluster.node_num)
            print( "-------------")
        print( "所有节点的个数为： %s" % len(self.vectors))
        print("簇类的个数为：%s" % self.cluster_num)
        print("花费的时间为： %.9fs" % (self.spend_time / 1000))

# 运行之后，一共聚类十类，聚类个数从0-9。
# 调用与画图
# 之后通过实例化类和调用函数，来实现聚类

# 读取测试集
data_file = rf"D:\Users\{USERNAME}\Downloads\data1.txt"
temperature_all_city = np.loadtxt(data_file, delimiter=",", usecols=(3, 4),encoding='utf-8')  # 读取聚类特征, 即最高温和最低温
xy = np.loadtxt(data_file, delimiter=",", usecols=(8, 9), encoding='utf-8')  # 读取各地经纬度
with open(data_file, 'r', encoding='utf-8')as f:
    lines = f.readlines()
    zone_dict = [i.split(',')[1] for i in lines]  # 读取地区并转化为字典


# 构建一趟聚类器
clustering = OnePassCluster(vector_list=temperature_all_city, t=9)
clustering.print_result(label_dict=zone_dict)
print(temperature_all_city)


# 将聚类结果导出图
fig, ax = pl.subplots()
c_map = pl.get_cmap('jet', clustering.cluster_num)
# c = 0

for c, cluster in zip([0, 7, 8, 4, 3, 9, 1, 5, 6, 2], clustering.cluster_list):
    for node in cluster.node_list:
        # 散点图，c设置点的颜色，s设置点的大小
        ax.scatter(xy[node][0], xy[node][1], c=c, s=3, cmap=c_map, vmin=0, vmax=clustering.cluster_num)
        # ax.scatter(xy[node][0], xy[node][1])
    # c += 1

pl.axis('off') # 不显示坐标轴
# 根据样本中经纬度的信息，并结合聚类算法的结果，可画图如下：
pl.savefig(rf'D:\Users\{USERNAME}\Downloads\test/map.jpg')
pl.show()

def main():
    pass


if __name__ == "__main__":
    main()
