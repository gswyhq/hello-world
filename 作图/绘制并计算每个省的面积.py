
import geopandas
import matplotlib.pyplot as plt
from shapely.geometry import Polygon

maps = geopandas.read_file('/home/Downloads/省级行政区.shx')
# 数据来源：链接：https://pan.baidu.com/s/1TzJT1sCi0L6HiKe6E_se8g  提取码：bfgh
# 读取的数据格式类似于
# geometry
# 0 POLYGON ((1329152.341 5619034.278, 1323327.591...
# 1 POLYGON ((-2189253.375 4611401.367, -2202922.3...
# 2 POLYGON ((761692.092 4443124.843, 760999.873 4...
# 3 POLYGON ((-34477.046 4516813.963, -41105.128 4...
# ... ...
maps.plot()
plt.savefig("test.png")

# 每一个省级行政区都被划分为一个区块，因此可以一行语句算出每个省级行政区所占面积：

print(maps.area)
# 0 4.156054e+11
# 1 1.528346e+12
# 2 1.487538e+11
# 3 4.781135e+10
# 4 1.189317e+12
# 5 1.468277e+11
# 6 1.597052e+11
# 7 9.770609e+10
# 8 1.385692e+11
# 9 1.846538e+11
# 10 1.015979e+11
# ... ...



