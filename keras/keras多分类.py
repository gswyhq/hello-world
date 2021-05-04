import keras
from keras.models import Sequential # 导入序冠模型
from keras.layers import Dense, Dropout, Activation # 导入全连接层 、、激活函数
from keras.optimizers import SGD # 导入随机梯度下降

# 生成训练和测试数据
import numpy as np
x_train = np.random.random((1000, 20)) #random.random()生成0和1之间的随机浮点数float
y_train = keras.utils.to_categorical(np.random.randint(10, size=(1000, 1)), num_classes=10) #to_categorical函数将类别向量(从0到nb_classes的整数向量)映射为二值类别矩阵, 用于应用到以 categorical_crossentropy 为目标函数的模型中.
x_test = np.random.random((100, 20))
y_test = keras.utils.to_categorical(np.random.randint(10, size=(100, 1)), num_classes=10)

model = Sequential()
# Dense(64) is a fully-connected layer with 64 hidden units.
# in the first layer, you must specify the expected input data shape:
# here, 20-dimensional vectors.
model.add(Dense(64, activation='relu', input_dim=20))
model.add(Dropout(0.5))
model.add(Dense(64, activation='relu'))
model.add(Dropout(0.5))
model.add(Dense(10, activation='softmax'))

sgd = SGD(lr=0.01, decay=1e-6, momentum=0.9, nesterov=True)
model.compile(loss='categorical_crossentropy',
              optimizer=sgd,
              metrics=['accuracy'])

model.fit(x_train, y_train,
          epochs=20,
          batch_size=128)
score = model.evaluate(x_test, y_test, batch_size=128)

