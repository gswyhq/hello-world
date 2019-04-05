# coding:utf-8
import xgboost as xgb
 
# 计算分类正确率
from sklearn.metrics import accuracy_score

# 二、数据读取
# XGBoost可以加载libsvm格式的文本数据，libsvm的文件格式（稀疏特征）如下：
# 1  101:1.2 102:0.03
# 0  1:2.1 10001:300 10002:400
# ...
# 每一行表示一个样本，第一行的开头的“1”是样本的标签。“101”和“102”为特征索引，'1.2'和'0.03' 为特征的值。
# 在两类分类中，用“1”表示正样本，用“0” 表示负样本。也支持[0,1]表示概率用来做标签，表示为正样本的概率。
#
# 下面的示例数据需要我们通过一些蘑菇的若干属性判断这个品种是否有毒。
# UCI数据描述：http://archive.ics.uci.edu/ml/machine-learning-databases/mushroom/ ，
# 每个样本描述了蘑菇的22个属性，比如形状、气味等等（将22维原始特征用加工后变成了126维特征，
# 并存为libsvm格式)，然后给出了这个蘑菇是否可食用。其中6513个样本做训练，1611个样本做测试。
#
# 注：libsvm格式文件说明如下 https://www.cnblogs.com/codingmengmeng/p/6254325.html
#
# XGBoost加载的数据存储在对象DMatrix中
# XGBoost自定义了一个数据矩阵类DMatrix，优化了存储和运算速度
# DMatrix文档：http://xgboost.readthedocs.io/en/latest/python/python_api.html

# curl http://archive.ics.uci.edu/ml/machine-learning-databases/mushroom/agaricus-lepiota.data -o agaricus-lepiota.data
#/home/gswyhq/Downloads/FormatDataLibsvm.xls

# read in data，数据在xgboost安装的路径下的demo目录,现在我们将其copy到当前代码下的data目录
my_workpath = './data/'
dtrain = xgb.DMatrix(my_workpath + 'agaricus.txt.train')
dtest = xgb.DMatrix(my_workpath + 'agaricus.txt.test')

 # 查看数据情况
dtrain.num_col()
 
dtrain.num_row()
 
dtest.num_row()

 # 训练参数设置
# max_depth： 树的最大深度。缺省值为6，取值范围为：[1,∞]
# eta：为了防止过拟合，更新过程中用到的收缩步长。在每次提升计算之后，算法会直接获得新特征的权重。
# eta通过缩减特征的权重使提升计算过程更加保守。缺省值为0.3，取值范围为：[0,1]
# silent：取0时表示打印出运行时信息，取1时表示以缄默方式运行，不打印运行时信息。缺省值为0
# objective： 定义学习任务及相应的学习目标，“binary:logistic” 表示二分类的逻辑回归问题，输出为概率。
#
# 其他参数取默认值。

# specify parameters via map
param = {'max_depth':2, 'eta':1, 'silent':0, 'objective':'binary:logistic' }
print(param)

 # 训练模型
# 设置boosting迭代计算次数
num_round = 2
 
import time
 
starttime = time.clock()
 
bst = xgb.train(param, dtrain, num_round)  # dtrain是训练数据集
 
endtime = time.clock()
print (endtime - starttime)

 # XGBoost预测的输出是概率。这里蘑菇分类是一个二类分类问题，输出值是样本为第一类的概率。
# 我们需要将概率值转换为0或1。
 
train_preds = bst.predict(dtrain)    #
print ("train_preds",train_preds)
 
train_predictions = [round(value) for value in train_preds]
print ("train_predictions",train_predictions)
 
y_train = dtrain.get_label()
print ("y_train",y_train)
 
train_accuracy = accuracy_score(y_train, train_predictions)
print ("Train Accuary: %.2f%%" % (train_accuracy * 100.0))
 
 # 模型训练好后，可以用训练好的模型对测试数据进行预测
# make prediction
preds = bst.predict(dtest)

# 检查模型在测试集上的正确率
# XGBoost预测的输出是概率，输出值是样本为第一类的概率。我们需要将概率值转换为0或1。


predictions = [round(value) for value in preds]
 
y_test = dtest.get_label()
 
test_accuracy = accuracy_score(y_test, predictions)
print("Test Accuracy: %.2f%%" % (test_accuracy * 100.0))


 # 模型可视化
# 调用XGBoost工具包中的plot_tree，在显示
# 要可视化模型需要安装graphviz软件包
# plot_tree（）的三个参数：
# 1. 模型
# 2. 树的索引，从0开始
# 3. 显示方向，缺省为竖直，‘LR'是水平方向

# from matplotlib import pyplot
# import graphviz
 
import graphviz
 
# xgb.plot_tree(bst, num_trees=0, rankdir='LR')
# pyplot.show()
 
# xgb.plot_tree(bst,num_trees=1, rankdir= 'LR' )
# pyplot.show()
# xgb.to_graphviz(bst,num_trees=0)
# xgb.to_graphviz(bst,num_trees=1)
 
# https://blog.csdn.net/u011630575/article/details/79418138

