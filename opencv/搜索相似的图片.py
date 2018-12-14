# -*- coding: utf-8 -*-
"""
http://visualgenome.org/api/v0/api_home.html

http://python.jobbole.com/80860/
"""

# import the necessary packages
import numpy as np
import cv2
import csv
import sys,os

class ColorDescriptor:
    '''用来封装所有用于提取图像中3D HSV颜色直方图的逻辑。

    定义图像描述符
    这里不使用标准的颜色直方图，而是对其进行一些修改，使其更加健壮和强大。
    这个图像描述符是HSV颜色空间的3D颜色直方图（色相、饱和度、明度）。'''
    def __init__(self, bins):
        # 参数——bins(组距)，即颜色直方图中组距的数目。
        self.bins = bins

    def describe(self, image):
        # 用于描述指定的图像
        # 将图像从RGB颜色空间（或是BGR颜色空间，OpenCV以NumPy数组的形式
        #反序表示RGB图像）转成HSV颜色空间。接着初始化用于量化图像的特征列表features。
        image = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
        features = []

        # 获取图像的维度，并计算图像中心(x, y)的位置。
        (h, w) = image.shape[:2]
        (cX, cY) = (int(w * 0.5), int(h * 0.5))

        # 这里不计算整个图像的3D HSV颜色直方图，而是计算图像中不同区域的3D HSV颜色直方图。
        #使用基于区域的直方图，而不是全局直方图的好处是：这样我们可以模拟各个区域的颜色分布。
        # 用于分别定义左上、右上、右下、和左下区域。
        segments = [(0, cX, 0, cY), (cX, w, 0, cY), (cX, w, cY, h),
                    (0, cX, cY, h)]

        # 需要构建一个椭圆用来表示图像的中央区域。在代码的27行，定义一个长短轴分别为图像长宽75%的椭圆。
        (axesX, axesY) = (int(w * 0.75 / 2), int(h * 0.75 / 2))
        #接着初始化一个空白图像（将图像填充0，表示黑色的背景），该图像与需要描述的图像大小相同
        ellipMask = np.zeros(image.shape[:2], dtype = "uint8")
        #使用cv2.ellipse函数绘制实际的椭圆。该函数需要8个不同的参数：
        #需要绘制椭圆的图像。这里使用了下面会介绍的“掩模”的概念。
        #两个元素的元组，用来表示图像的中心坐标。
        #两个元组的元组，表示椭圆的两个轴。在这里，椭圆的长短轴长度为图像长宽的75%。
        #椭圆的旋转角度。在本例中，椭圆无需旋转，所以值为0。
        #椭圆的起始角。
        #椭圆的终止角。看上一个参数，这意味着绘制的是完整的椭圆。
        #椭圆的颜色，255表示的绘制的是白色椭圆。
        #椭圆边框的大小。传递正数比会以相应的像素数目绘制椭圆边框。负数表示椭圆是填充模式。

        #我们的目标是分开描述图像的每个区域。表述不同区域最高效的方法是使用掩模。
        #对于图像中某个点(x, y)，只有掩模中该点位白色(255)时，该像素点才会用于计算直方图。
        #如果该像素点对应的位置在掩模中是黑色(0)，将会被忽略。另外参数必须为整数。
        cv2.ellipse(ellipMask, (cX, cY), (axesX, axesY), 0, 0, 360, 255, -1)

        # 独立检测每块区域，在迭代中移除每个矩形与图像中间的椭圆重叠的部分。
        for (startX, endX, startY, endY) in segments:
            # 每个角的掩模分配内存
            cornerMask = np.zeros(image.shape[:2], dtype = "uint8")
            #图像的每个角绘制白色矩形
            cv2.rectangle(cornerMask, (startX, startY), (endX, endY), 255, -1)
            #将矩形减去中间的椭圆。
            cornerMask = cv2.subtract(cornerMask, ellipMask)

            # 针对每个区域都调用直方图方法。
            #第一个参数是需要提取特征的图像，第二个参数是掩模区域，这样来提取颜色直方图。
            #histogram方法会返回当前区域的颜色直方图表示，我们将其添加到特征列表中。
            hist = self.histogram(image, cornerMask)
            features.extend(hist)

        # 提取图像中间（椭圆）区域的颜色直方图并更新features列表。
        hist = self.histogram(image, ellipMask)
        features.extend(hist)

        # 返回特征向量
        return features

    def histogram(self, image, mask):
        '''histogram方法需要两个参数，第一个是需要描述的图像，第二个是mask，描述需要描述的图像区域。'''
        # 通过调用cv2.calcHist计算图像掩模区域的直方图，使用构造器中的bin(组距)数目作为参数。
        hist = cv2.calcHist([image], [0, 1, 2], mask, self.bins,
                            [0, 180, 0, 256, 0, 256])

        #对直方图归一化。这意味着如果我们计算两幅相同的图像，其中一幅比另一幅大50%，
        #直方图会是相同的。对直方图进行归一化非常重要，这样每个直方图表示的就是图像中
        #每个bin的所占的比例，而不是每个bin的个数。同样，归一化能保证不同尺寸但内容
        #近似的图像也会在比较函数中认为是相似的。
        hist = cv2.normalize(hist,hist).flatten()

        # 返回归一化后的3D HSV颜色直方图
        return hist


class Searcher:
    def __init__(self, indexPath):
        # Searcher类的构造器只需一个参数，indexPath，用于表示index.csv文件在磁盘上的路径。
        self.indexPath = indexPath

    def search(self, queryFeatures, limit = 10):
        # 调用search方法。该方法需要两个参数，queryFeatures是提取自待搜索图像
        #（如向CBIR系统提交并请求返回相似图像的图像），和返回图像的数目的最大值。
        #初始化results字典。在这里，字典有很用的用途，每个图像有唯一的imageID，可以作为字典的键，而相似度作为字典的值。
        results = {}

        # open the index file for reading
        with open(self.indexPath) as f:
            # 打开index.csv文件,获取CSV读取器的句柄
            reader = csv.reader(f)

            # 循环读取index.csv文件的每一行。
            for row in reader:
                features = [float(x) for x in row[1:]]
                #用chi2_distance函数将其与待搜索的图像特征进行比较
                d = self.chi2_distance(features, queryFeatures)

                results[row[0]] = d

        # 使用唯一的图像文件名作为键，用与待查找图像的与索引后的图像的相似读作为值来更新results字典。
        #最后，将results字典根据相似读升序排序。
        #卡方相似度为零的图片表示完全相同。相似度数值越高，表示两幅图像差别越大。
        results = sorted([(v, k) for (k, v) in results.items()])

        # return our (limited) results
        return results[:limit]

    def chi2_distance(self, histA, histB, eps = 1e-10):
        #函数需要两个参数，即用来进行比较的两个直方图。可选的eps值用来预防除零错误。
        d = 0.5 * np.sum([((a - b) ** 2) / (a + b + eps)
            for (a, b) in zip(histA, histB)])

        # return the chi-squared distance
        return d

def index (outputfile='index.csv',dataset_path=''):
    #为了索引化我们的相册数据集，
    #完成后将会获得一个名为index.csv的新文件。
    #可以看到在.csv文件的每一行，第一项是文件名，第二项是一个数字列表。
    #这个数字列表就是用来表示并量化图像的特征向量。

    import glob  #用来获取图像的文件路径

    '''
    对数据集中的每幅图像提取特征（如颜色直方图）。
    提取特征并将其持久保存起来的过程一般称为“索引化”。
    '''

    # 初始化ColorDescriptor，8 bin拥有色相、12 bin用于饱和度、3 bin用于明度。
    cd = ColorDescriptor((8, 12, 3))

    # 打开输出文件
    with open(outputfile,'w',encoding='utf8')as output:

        # use glob to grab the image paths and loop over them
        #遍历数据集中的所有图像。
        for imagePath in glob.glob(dataset_path + "/*.png"):
            # 对于每幅图像，可以提取一个imageID，即图像的文件名。对于这个作为示例的搜索引擎，
            #我们假定每个文件名都是唯一的，也可以针对每幅图像生产一个UUID。
            imageID = imagePath[imagePath.rfind("/") + 1:]

            #将从磁盘上读取图像。
            image = cv2.imread(imagePath)

            # 图像使用图像描述符并提取特征。ColorDescriptor的describe方法返回由浮点数构成的列表，用来量化并表示图像。
            #这个数字列表，或者说特征向量，含有第一步中图像的5个区域的描述。
            #每个区域由一个直方图表示，含有8 × 12 × 3 = 288项。5个区域总共有5 × 288 = 1440维度。
            #因此每个图像使用1440个数字量化并表示。
            features = cd.describe(image)

            # 将图像的文件名和管理的特征向量写入文件。
            features = [str(f) for f in features]
            output.write("%s,%s\n" % (imageID, ",".join(features)))

def search(index_file='index.csv',query_image='',dataset_path=''):
    #这里将INRIA假期数据集（http://lear.inrialpes.fr/people/jegou/data.php）作为图像搜索的数据集。
    #这个数据集含有全世界许多地方的假期旅行，包括埃及金字塔、潜水、山区的森林、餐桌上的瓶子和盘子、游艇、海面上的日落。

    # 处理命令行参数。我们需要用一个–index来表示index.csv文件的位置。
    #还需要–query来表示带搜索图像的存储路径。该图像将与数据集中的每幅图像进行比较。
    #目标是找到数据集中欧给你与待搜索图像相似的图像。

    # 使用图像描述符提取相同的参数，就如同在索引化那一步做的一样。
    #如果我们是为了比较图像的相似度（事实也正是如此），就无需改变数据集中颜色直方图的bin的数目。
    cd = ColorDescriptor((8, 12, 3))

    # 从磁盘读取待搜索图像
    query = cv2.imread(query_image)
    features = cd.describe(query)#获取该图像的特征

    #使用提取到的特征进行搜索，返回经过排序后的结果列表。
    searcher = Searcher(index_file)
    results = searcher.search(features)

    # 显示出待搜索的照片
    cv2.imshow("Query", query)

    # 遍历搜索结果，将相应的图像显示在屏幕上。
    for (score, resultID) in results:
        # load the result image and display it
        if sys.platform[:5] == 'linux':
            result = cv2.imread(os.path.join(dataset_path,resultID))
        else:
            result = cv2.imread(resultID)

        cv2.imshow("Result", result)
        cv2.waitKey(0)

if __name__=='__main__':
    outputfile='/home/gswewf/VQA/index.csv'
    dataset_path='/home/gswewf/gow69/data/dataset'
    query_image='/home/gswewf/gow69/data/queries/108100.png'
    #index (outputfile,dataset_path)
    search(outputfile,query_image,dataset_path)


