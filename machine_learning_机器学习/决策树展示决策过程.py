




from sklearn.datasets import load_wine
from sklearn import metrics
import matplotlib.pyplot as plt
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
from sklearn import tree
import pandas as pd
​
from sklearn.datasets import load_wine
from sklearn import metrics
import matplotlib.pyplot as plt
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
from sklearn import tree
import pandas as pd

CSV_HEADER = [
    "age",
    "workclass",
    "fnlwgt",
    "education",
    "education_num",
    "marital_status",
    "occupation",
    "relationship",
    "race",
    "gender",
    "capital_gain",
    "capital_loss",
    "hours_per_week",
    "native_country",
    "income_bracket",
]
​
# 数据来源：https://archive.ics.uci.edu/ml/machine-learning-databases/adult/
​
train_data_url = "./data/adult/adult.data"
data = pd.read_csv(train_data_url, header=None)
print({name: idx for idx, name in enumerate(CSV_HEADER)})
data.head()
{'age': 0, 'workclass': 1, 'fnlwgt': 2, 'education': 3, 'education_num': 4, 'marital_status': 5, 'occupation': 6, 'relationship': 7, 'race': 8, 'gender': 9, 'capital_gain': 10, 'capital_loss': 11, 'hours_per_week': 12, 'native_country': 13, 'income_bracket': 14}
0	1	2	3	4	5	6	7	8	9	10	11	12	13	14
0	39	State-gov	77516	Bachelors	13	Never-married	Adm-clerical	Not-in-family	White	Male	2174	0	40	United-States	<=50K
1	50	Self-emp-not-inc	83311	Bachelors	13	Married-civ-spouse	Exec-managerial	Husband	White	Male	0	0	13	United-States	<=50K
2	38	Private	215646	HS-grad	9	Divorced	Handlers-cleaners	Not-in-family	White	Male	0	0	40	United-States	<=50K
3	53	Private	234721	11th	7	Married-civ-spouse	Handlers-cleaners	Husband	Black	Male	0	0	40	United-States	<=50K
4	28	Private	338409	Bachelors	13	Married-civ-spouse	Prof-specialty	Wife	Black	Female	0	0	40	Cuba	<=50K

data[1][data[1]==' Private']=0
data[1][data[1]==' Self-emp-not-inc']=1
data[1][data[1]==' Self-emp-inc']=2
data[1][data[1]==' Federal-gov']=3
data[1][data[1]==' Local-gov']=4
data[1][data[1]==' State-gov']=5
data[1][data[1]==' Without-pay']=6
data[1][data[1]==' Never-worked']=7

data[3][data[3]==' Bachelors']=0
data[3][data[3]==' Some-college']=1
data[3][data[3]==' 11th']=2
data[3][data[3]==' HS-grad']=3
data[3][data[3]==' Prof-school']=4
data[3][data[3]==' Assoc-acdm']=5
data[3][data[3]==' Assoc-voc']=6
data[3][data[3]==' 9th']=7
data[3][data[3]==' 7th-8th']=8
data[3][data[3]==' 12th']=9
data[3][data[3]==' Masters']=10
data[3][data[3]==' 1st-4th']=11
data[3][data[3]==' 10th']=12
data[3][data[3]==' Doctorate']=13
data[3][data[3]==' 5th-6th']=14
data[3][data[3]==' Preschool']=15
​
data[5][data[5]==' Married-civ-spouse']=0
data[5][data[5]==' Divorced']=1
data[5][data[5]==' Never-married']=2
data[5][data[5]==' Separated']=3
data[5][data[5]==' Widowed']=4
data[5][data[5]==' Married-spouse-absent']=5
data[5][data[5]==' Married-AF-spouse']=6
​
data[6][data[6]==' Tech-support']=0
data[6][data[6]==' Craft-repair']=1
data[6][data[6]==' Other-service']=2
data[6][data[6]==' Sales']=3
data[6][data[6]==' Exec-managerial']=4
data[6][data[6]==' Prof-specialty']=5
data[6][data[6]==' Handlers-cleaners']=6
data[6][data[6]==' Machine-op-inspct']=7
data[6][data[6]==' Adm-clerical']=8
data[6][data[6]==' Farming-fishing']=9
data[6][data[6]==' Transport-moving']=10
data[6][data[6]==' Priv-house-serv']=11
data[6][data[6]==' Protective-serv']=12
data[6][data[6]==' Armed-Forces']=13
​
data[7][data[7]==' Wife']=0
data[7][data[7]==' Own-child']=1
data[7][data[7]==' Husband']=2
data[7][data[7]==' Not-in-family']=3
data[7][data[7]==' Other-relative']=4
data[7][data[7]==' Unmarried']=5
​
data[8][data[8]==' White']=0
data[8][data[8]==' Asian-Pac-Islander']=1
data[8][data[8]==' Amer-Indian-Eskimo']=2
data[8][data[8]==' Other']=3
data[8][data[8]==' Black']=4
​
data[9][data[9]==' Male']=0
data[9][data[9]==' Female']=1
​
data[14][data[14]==' >50K']=0
data[14][data[14]==' <=50K']=1
建模（决策树）

X=data.iloc[:,[3,5,8,9]]#提取特征值
X=X.astype('int')#转换为整数型
Y=data.iloc[:,14]#提取标签值
Y=Y.astype('int')#转换为整数型
x_train, x_test, y_train, y_test = train_test_split(X, Y, test_size=0.2)#分割训练集和测试集
dt_clf = tree.DecisionTreeClassifier()#生成实例
dt_clf = dt_clf.fit(x_train, y_train)
tree_y_pred = dt_clf.predict(x_test)#生成预测值
metrics.accuracy_score(y_test,tree_y_pred)#真实值与预测值的一致性
0.8171349608475357
决策树直接展示

import graphviz
tree.plot_tree(dt_clf) 
feature_name=['education','marital-status','race','sex']
dot_data = tree.export_graphviz(dt_clf,filled=True,rounded=True,feature_names=feature_name,class_names=['>50k','<=50k'])
graph = graphviz.Source(dot_data)
graph#直接展示（法一）



#输出为pdf文件
dot_data=dot_data.replace('helvetica','"Microsoft Yahei"') # 修改字体，支持中文
graph = graphviz.Source(dot_data)
graph.render(r'./result/adult123', format='pdf')#将可视化结果输出至指定位置
'./result/adult123.pdf'

​

​

# 注意：
使用决策树展示决策过程的时候，需要将类别特征转换为数值型；
类别特征转换为数值方法：
1、类别特征的最优切分。
在枚举分割点之前，先把直方图按照每个类别对应的label均值进行排序；然后按照排序的结果依次枚举最优分割点。
类似频数编码排序，sum(y)/count(y)
2、转成数值特征。
a) 把类别特征转成one-hot coding扔到NN里训练个embedding；
b) 类似于CTR(点击率,Click-Through-Rate)特征，统计每个类别对应的label(训练目标)的均值。
3、其他的编码方法。
https://github.com/scikit-learn-contrib/category_encoders


XgBoost和Random Forest，不能直接处理categorical feature，必须先编码成为numerical feature。
lightgbm和CatBoost，可以直接处理categorical feature。
lightgbm： 需要先做label encoding。用特定算法（On Grouping for Maximum Homogeneity）找到optimal split，效果优于ONE。也可以选择采用one-hot encoding。https://lightgbm.readthedocs.io/en/latest/Features.html?highlight=categorical#optimal-split-for-categorical-features
CatBoost： 不需要先做label encoding。可以选择采用one-hot encoding，target encoding (with regularization)。https://catboost.ai/en/docs/concepts/algorithm-main-stages_cat-to-numberic




