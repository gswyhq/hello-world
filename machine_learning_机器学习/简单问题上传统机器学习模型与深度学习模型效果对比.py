
####################################################################################################################
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression

# 加载数据
iris = load_iris()
X = iris.data
y = iris.target

# 划分训练集和测试集
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# 创建模型
model = LogisticRegression(max_iter=1000)

# 训练模型
model.fit(X_train, y_train)

# 测试模型
score = model.score(X_test, y_test)
print('Accuracy:', score)

# 轻轻松松就做到了正确率：Accuracy: 1.0



####################################################################################################################



from keras.models import Sequential
from keras.layers import Dense, Dropout
from keras.optimizers import SGD
import matplotlib.pyplot as plt

# 创建模型
model = Sequential()
model.add(Dense(128, input_dim=(4), activation='relu'))
# model.add(Dense(128, activation='relu'))
# model.add(Dropout(0.5))
# 添加输入层
model.add(Dense(3, activation='softmax'))

# 编译模型
model.compile(loss='sparse_categorical_crossentropy', optimizer=SGD(lr=0.01), metrics=['accuracy'])

# 训练模型
history = model.fit(X_train, y_train, epochs=500, batch_size=32)

# 评估模型
loss, accuracy = model.evaluate(X_test, y_test)
print('Accuracy:', accuracy)

plt.plot(history.history["loss"], label="train_loss")
plt.xlabel("Epochs")
plt.ylabel("Loss")
plt.legend()
plt.show()


