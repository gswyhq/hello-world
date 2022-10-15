#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import tensorflow as tf
from tensorflow import keras

import os
import tempfile

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from pylab import mpl
mpl.rcParams['font.sans-serif'] = ['SimHei'] #指定默认字体   
mpl.rcParams['axes.unicode_minus'] = False #解决保存图像是负号'-'显示为方块的问题
import sklearn
from sklearn.metrics import confusion_matrix
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

mpl.rcParams['figure.figsize'] = (12, 10)
colors = plt.rcParams['axes.prop_cycle'].by_key()['color']

# 资料来源： https://tensorflow.google.cn/tutorials/structured_data/imbalanced_data

# 读取信用卡欺诈数据
file = tf.keras.utils
# raw_df = pd.read_csv('https://storage.googleapis.com/download.tensorflow.org/data/creditcard.csv')
raw_df = pd.read_csv(r"D:\Users\{}\Downloads\creditcard.csv".format(os.getenv("USERNAME")), sep=',', encoding='utf-8')
raw_df.head()

# 查看数据统计信息
raw_df[['Time', 'V1', 'V2', 'V3', 'V4', 'V5', 'V26', 'V27', 'V28', 'Amount', 'Class']].describe()

# 检查类标签不平衡
# 让我们看一下数据集的不平衡：
neg, pos = np.bincount(raw_df['Class'])
total = neg + pos
print('示例(Examples):\n    总量(Total): {}\n    正样本(Positive): {} ({:.2f}% 占比(of total))\n'.format(
    total, pos, 100 * pos / total))
# 这显示了一小部分正样本。

# 清理、拆分和规范化数据
# 原始数据有一些问题。首先，Time和Amount列的变量太大而无法直接使用。删除该Time列（因为不清楚它的含义）并记录该Amount列的对数以减小其范围。
cleaned_df = raw_df.copy()

# You don't want the `Time` column.
cleaned_df.pop('Time')

# The `Amount` column covers a huge range. Convert to log-space.
eps = 0.001 # 0 => 0.1¢
cleaned_df['Log Amount'] = np.log(cleaned_df.pop('Amount')+eps)
# 将数据集拆分为训练集、验证集和测试集。
# 在模型拟合期间使用验证集来评估损失和任何指标，但是模型不适合这些数据。
# 测试集在训练阶段完全未使用，仅在最后用于评估模型对新数据的泛化程度。这对于不平衡的数据集尤其重要，因为缺乏训练数据，过度拟合是一个重要问题。



# Use a utility from sklearn to split and shuffle your dataset.
train_df, test_df = train_test_split(cleaned_df, test_size=0.2)
train_df, val_df = train_test_split(train_df, test_size=0.2)

# Form np arrays of labels and features.
train_labels = np.array(train_df.pop('Class'))
bool_train_labels = train_labels != 0
val_labels = np.array(val_df.pop('Class'))
test_labels = np.array(test_df.pop('Class'))

train_features = np.array(train_df)
val_features = np.array(val_df)
test_features = np.array(test_df)
# 使用 sklearn StandardScaler 标准化输入特征。这会将平均值设置为 0，标准差设置为 1。
#
# 注意：仅StandardScaler适合使用train_features以确保模型不会偷看验证集或测试集。

scaler = StandardScaler()
train_features = scaler.fit_transform(train_features)

val_features = scaler.transform(val_features)
test_features = scaler.transform(test_features)

train_features = np.clip(train_features, -5, 5)
val_features = np.clip(val_features, -5, 5)
test_features = np.clip(test_features, -5, 5)


print('Training labels shape:', train_labels.shape)
print('Validation labels shape:', val_labels.shape)
print('Test labels shape:', test_labels.shape)

print('Training features shape:', train_features.shape)
print('Validation features shape:', val_features.shape)
print('Test features shape:', test_features.shape)


# 看数据分布
# 接下来比较一些特征上正例和负例的分布。此时要问自己的好问题是：
#
# 这些分布有意义吗？
# 是的。您已经对输入进行了归一化，这些输入大多集中在该+/- 2范围内。
# 你能看出分布之间的区别吗？
# 是的，正面的例子包含更高的极值率。

pos_df = pd.DataFrame(train_features[ bool_train_labels], columns=train_df.columns)
neg_df = pd.DataFrame(train_features[~bool_train_labels], columns=train_df.columns)

sns.jointplot(x=pos_df['V5'], y=pos_df['V6'],
              kind='hex', xlim=(-5,5), ylim=(-5,5))
plt.suptitle("正例样本分布(Positive distribution)")

sns.jointplot(x=neg_df['V5'], y=neg_df['V6'],
              kind='hex', xlim=(-5,5), ylim=(-5,5))
_ = plt.suptitle("负例样本分布(Negative distribution)")


# 定义模型和指标
# 定义一个函数，该函数创建一个简单的神经网络，该网络具有一个密集连接的隐藏层、一个用于减少过度拟合的dropout层，以及一个返回交易欺诈概率的输出 sigmoid 层：


METRICS = [
      keras.metrics.TruePositives(name='tp'),
      keras.metrics.FalsePositives(name='fp'),
      keras.metrics.TrueNegatives(name='tn'),
      keras.metrics.FalseNegatives(name='fn'),
      keras.metrics.BinaryAccuracy(name='accuracy'),
      keras.metrics.Precision(name='precision'),
      keras.metrics.Recall(name='recall'),
      keras.metrics.AUC(name='auc'),
      keras.metrics.AUC(name='prc', curve='PR'), # precision-recall curve
]

def make_model(metrics=None, output_bias=None):
    if metrics is None:
        metrics = METRICS
    if output_bias is not None:
        output_bias = tf.keras.initializers.Constant(output_bias)
    model = keras.Sequential([
      keras.layers.Dense(
          16, activation='relu',
          input_shape=(train_features.shape[-1],)),
      keras.layers.Dropout(0.5),
      keras.layers.Dense(1, activation='sigmoid',
                         bias_initializer=output_bias),
    ])

    model.compile(
      optimizer=keras.optimizers.Adam(learning_rate=1e-3),
      loss=keras.losses.BinaryCrossentropy(),
      metrics=metrics)

    return model

# 了解有用的指标
# 请注意，上面定义的一些指标可以由模型计算，这在评估性能时会有所帮助。
#
# 假阴性和假阳性是错误分类的样本
# 真阴性和真阳性是正确分类的样本
# 准确率是正确分类的示例的百分比 = (正确样本数true samples)/(总样本数total samples)
# 精度是正确分类 的预测阳性的百分比= (正确阳性true positives)/(正确阳性true positives+错误阳性false positives)
# 召回率是正确分类 的实际阳性的百分比 = (正确阳性true positives)/(正确阳性true positives+错误阴性false negatives)
# AUC是指受试者工作特征曲线 (ROC-AUC) 的曲线下面积。该指标等于分类器将随机正样本排名高于随机负样本的概率。
# AUPRC是指精确召回曲线的曲线下面积。该指标计算不同概率阈值的精确召回对。
# 注意：准确性不是此任务的有用指标。通过始终预测 False，您可以在此任务上获得 99.8% 以上的准确率。

# 基线模型
# 建立模型
# 现在使用之前定义的函数创建和训练您的模型。请注意，该模型使用大于默认批量大小 2048 进行拟合，这对于确保每个批次都有相当大的机会包含一些正样本非常重要。如果批量太小，他们可能没有欺诈交易可供学习。
#
# 注意：这个模型不能很好地处理类不平衡。您将在本教程的后面部分对其进行改进。

EPOCHS = 100
BATCH_SIZE = 2048

early_stopping = tf.keras.callbacks.EarlyStopping(
    monitor='val_prc',
    verbose=1,
    patience=10,
    mode='max',
    restore_best_weights=True)

model = make_model()
model.summary()

# 测试运行模型：
model.predict(train_features[:10])

# 可选：设置正确的初始偏差。
# 这些最初的猜测并不好。你知道数据集是不平衡的。设置输出层的偏差以反映这一点（请参阅：训练神经网络的秘诀：“init well”）。这有助于初始收敛。

# 使用默认偏差初始化，损失应该是math.log(2) = 0.69314
results = model.evaluate(train_features, train_labels, batch_size=BATCH_SIZE, verbose=0)
print("Loss: {:0.4f}".format(results[0]))

# Loss: 0.6775
# 要设置的正确偏差可以来自：
# p0 = pos/(pos + neg) = 1/(1+exp(-b0))
# b0 = -ln(1/p0-1)
# b0=ln(pos/neg)

initial_bias = np.log([pos/neg])
initial_bias

# array([-6.35935934])
# 将其设置为初始偏差，模型将给出更合理的初始猜测。

# 它应该在附近：pos/total = 0.0018


model = make_model(output_bias=initial_bias)
model.predict(train_features[:10])

# 通过此初始化，初始损失应约为：
# -p0*log(p0) - (1-p0)*log(1-p0) = 0.01317

results = model.evaluate(train_features, train_labels, batch_size=BATCH_SIZE, verbose=0)
print("Loss: {:0.4f}".format(results[0]))

# Loss: 0.0209
# 这个初始损失大约是初始初始化的 50 倍（0.69314/0.01317）。

# 通过这种方式，模型不需要花费前几个 epoch 来学习积极的例子是可能的。
# 这也使得在训练期间更容易阅读损失图。

# 检查初始权重
# 为了使各种训练运行更具可比性，将此初始模型的权重保存在检查点文件中，并在训练前将它们加载到每个模型中：
initial_weights = os.path.join(tempfile.mkdtemp(), 'initial_weights')
model.save_weights(initial_weights)

# 确认偏差修复有帮助
# 在继续之前，请快速确认仔细的偏差初始化确实有帮助。
#
# 对模型进行 20 个 epoch 的训练，无论有没有这种仔细的初始化，并比较损失：


model = make_model()
model.load_weights(initial_weights)
model.layers[-1].bias.assign([0.0])
zero_bias_history = model.fit(
    train_features,
    train_labels,
    batch_size=BATCH_SIZE,
    epochs=20,
    validation_data=(val_features, val_labels),
    verbose=0)

model = make_model()
model.load_weights(initial_weights)
careful_bias_history = model.fit(
    train_features,
    train_labels,
    batch_size=BATCH_SIZE,
    epochs=20,
    validation_data=(val_features, val_labels),
    verbose=0)

def plot_loss(history, label, n):
    # Use a log scale on y-axis to show the wide range of values.
    plt.semilogy(history.epoch, history.history['loss'],
               color=colors[n], label='Train ' + label)
    plt.semilogy(history.epoch, history.history['val_loss'],
               color=colors[n], label='Val ' + label,
               linestyle="--")
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()

plot_loss(zero_bias_history, "Zero Bias", 0)
plot_loss(careful_bias_history, "Careful Bias", 1)

# 上图清楚地表明：在验证损失方面，在这个问题上，这种经过设置的初始化具有明显的优势。

# 训练模型

model = make_model()
model.load_weights(initial_weights)
baseline_history = model.fit(
    train_features,
    train_labels,
    batch_size=BATCH_SIZE,
    epochs=EPOCHS,
    callbacks=[early_stopping],
    validation_data=(val_features, val_labels))

# 查看训练历史
# 在本节中，您将生成模型在训练和验证集上的准确度和损失图。这些对于检查过拟合很有用，您可以在过拟合和欠拟合教程中了解更多信息。
#
# 此外，您可以为上面创建的任何指标生成这些图。包括假阴性作为示例。


def plot_metrics(history):
  metrics = ['loss', 'prc', 'precision', 'recall']
  for n, metric in enumerate(metrics):
    name = metric.replace("_"," ").capitalize()
    plt.subplot(2,2,n+1)
    plt.plot(history.epoch, history.history[metric], color=colors[0], label='Train')
    plt.plot(history.epoch, history.history['val_'+metric],
             color=colors[0], linestyle="--", label='Val')
    plt.xlabel('Epoch')
    plt.ylabel(name)
    if metric == 'loss':
      plt.ylim([0, plt.ylim()[1]])
    elif metric == 'auc':
      plt.ylim([0.8,1])
    else:
      plt.ylim([0,1])

    plt.legend()

plot_metrics(baseline_history)

# 注意：验证曲线通常比训练曲线表现更好。这主要是因为在评估模型时 dropout 层没有激活。

# 评估指标
# 您可以使用混淆矩阵来总结实际标签与预测标签，其中 X 轴是预测标签，Y 轴是实际标签：


train_predictions_baseline = model.predict(train_features, batch_size=BATCH_SIZE)
test_predictions_baseline = model.predict(test_features, batch_size=BATCH_SIZE)


def plot_cm(labels, predictions, p=0.5):
    cm = confusion_matrix(labels, predictions > p)
    plt.figure(figsize=(5,5))
    sns.heatmap(cm, annot=True, fmt="d")
    plt.title('Confusion matrix @{:.2f}'.format(p))
    plt.ylabel('Actual label')
    plt.xlabel('Predicted label')

    print('Legitimate Transactions Detected (True Negatives): ', cm[0][0])
    print('Legitimate Transactions Incorrectly Detected (False Positives): ', cm[0][1])
    print('Fraudulent Transactions Missed (False Negatives): ', cm[1][0])
    print('Fraudulent Transactions Detected (True Positives): ', cm[1][1])
    print('Total Fraudulent Transactions: ', np.sum(cm[1]))
# 在测试数据集上评估您的模型并显示您在上面创建的指标的结果：


baseline_results = model.evaluate(test_features, test_labels,
                                  batch_size=BATCH_SIZE, verbose=0)
for name, value in zip(model.metrics_names, baseline_results):
  print(name, ': ', value)


plot_cm(test_labels, test_predictions_baseline)

# 如果模型完美地预测了一切，这将是一个对角矩阵，其中主对角线以外的值表示不正确的预测，将为零。在这种情况下，矩阵显示您的误报相对较少，这意味着被错误标记的合法交易相对较少。但是，尽管增加误报数量的成本，您可能希望得到更少的误报。这种权衡可能更可取，因为误报会允许欺诈交易通过，而误报可能会导致向客户发送一封电子邮件，要求他们验证他们的卡活动。
#
# 绘制 ROC
# 现在绘制ROC。该图很有用，因为它一目了然地显示了模型只需调整输出阈值即可达到的性能范围。


def plot_roc(name, labels, predictions, **kwargs):
  fp, tp, _ = sklearn.metrics.roc_curve(labels, predictions)

  plt.plot(100*fp, 100*tp, label=name, linewidth=2, **kwargs)
  plt.xlabel('False positives [%]')
  plt.ylabel('True positives [%]')
  plt.xlim([-0.5,20])
  plt.ylim([80,100.5])
  plt.grid(True)
  ax = plt.gca()
  ax.set_aspect('equal')

plot_roc("Train Baseline", train_labels, train_predictions_baseline, color=colors[0])
plot_roc("Test Baseline", test_labels, test_predictions_baseline, color=colors[0], linestyle='--')
plt.legend(loc='lower right');

# 绘制 AUPRC
# 现在绘制AUPRC。插值精度-召回曲线下的面积，通过针对不同的分类阈值绘制（召回，精度）点而获得。根据计算方式，PR AUC 可能等于模型的平均精度。


def plot_prc(name, labels, predictions, **kwargs):
    precision, recall, _ = sklearn.metrics.precision_recall_curve(labels, predictions)

    plt.plot(precision, recall, label=name, linewidth=2, **kwargs)
    plt.xlabel('Precision')
    plt.ylabel('Recall')
    plt.grid(True)
    ax = plt.gca()
    ax.set_aspect('equal')

plot_prc("Train Baseline", train_labels, train_predictions_baseline, color=colors[0])
plot_prc("Test Baseline", test_labels, test_predictions_baseline, color=colors[0], linestyle='--')
plt.legend(loc='lower right');

# 看起来精度相对较高，但召回率和 ROC 曲线下面积 (AUC) 并没有您想的那么高。
# 分类器在尝试最大化精度和召回率时经常面临挑战，在处理不平衡数据集时尤其如此。
# 在您关心的问题的背景下考虑不同类型错误的成本是很重要的。在此示例中，误报（错过欺诈交易）可能会产生财务成本，而误报（交易被错误地标记为欺诈）可能会降低用户的幸福感。


# # 类别权重
# 计算类别权重
# 目标是识别欺诈性交易，但您没有太多的正面样本可供使用，因此您希望分类器对可用的少数样本进行大量加权。您可以通过参数传递每个类的 Keras 权重来做到这一点。这些将导致模型“更加关注”来自代表性不足的类的示例。


# Scaling by total/2 helps keep the loss to a similar magnitude.
# The sum of the weights of all examples stays the same.
weight_for_0 = (1 / neg) * (total / 2.0)
weight_for_1 = (1 / pos) * (total / 2.0)

class_weight = {0: weight_for_0, 1: weight_for_1}

print('Weight for class 0: {:.2f}'.format(weight_for_0))
print('Weight for class 1: {:.2f}'.format(weight_for_1))

# 使用类权重训练模型
# 现在尝试使用类权重重新训练和评估模型，看看它如何影响预测。
#
# 注意：使用class_weights改变了损失的范围。这可能会影响训练的稳定性，具体取决于优化器。
# 步长取决于梯度大小的优化器，例如tf.keras.optimizers.SGD，可能会失败。
# 此处使用的优化器tf.keras.optimizers.Adam不受缩放变化的影响。
# 另请注意，由于加权，两个模型之间的总损失不具有可比性。


weighted_model = make_model()
weighted_model.load_weights(initial_weights)

weighted_history = weighted_model.fit(
    train_features,
    train_labels,
    batch_size=BATCH_SIZE,
    epochs=EPOCHS,
    callbacks=[early_stopping],
    validation_data=(val_features, val_labels),
    # The class weights go here
    class_weight=class_weight)

# 查看训练历史
plot_metrics(weighted_history)

# 评估指标

train_predictions_weighted = weighted_model.predict(train_features, batch_size=BATCH_SIZE)
test_predictions_weighted = weighted_model.predict(test_features, batch_size=BATCH_SIZE)

weighted_results = weighted_model.evaluate(test_features, test_labels,
                                           batch_size=BATCH_SIZE, verbose=0)
for name, value in zip(weighted_model.metrics_names, weighted_results):
  print(name, ': ', value)


plot_cm(test_labels, test_predictions_weighted)

# 在这里您可以看到，使用类权重，准确率和准确率较低，因为有更多的误报，但相反，召回率和 AUC 更高，因为模型还发现了更多的真阳性。
# 尽管准确率较低，但该模型具有较高的召回率（并识别出更多的欺诈交易）。
# 当然，这两种类型的错误都是有代价的（您也不希望通过将太多合法交易标记为欺诈来欺骗用户）。
# 为您的应用程序仔细考虑这些不同类型的错误之间的权衡。

# 绘制 ROC

plot_roc("Train Baseline", train_labels, train_predictions_baseline, color=colors[0])
plot_roc("Test Baseline", test_labels, test_predictions_baseline, color=colors[0], linestyle='--')

plot_roc("Train Weighted", train_labels, train_predictions_weighted, color=colors[1])
plot_roc("Test Weighted", test_labels, test_predictions_weighted, color=colors[1], linestyle='--')


plt.legend(loc='lower right');


# 绘制 AUPRC

plot_prc("Train Baseline", train_labels, train_predictions_baseline, color=colors[0])
plot_prc("Test Baseline", test_labels, test_predictions_baseline, color=colors[0], linestyle='--')

plot_prc("Train Weighted", train_labels, train_predictions_weighted, color=colors[1])
plot_prc("Test Weighted", test_labels, test_predictions_weighted, color=colors[1], linestyle='--')


plt.legend(loc='lower right');

# 过采样
# 过采样少数类
# 一种相关的方法是通过对少数类进行过采样来重新采样数据集。


pos_features = train_features[bool_train_labels]
neg_features = train_features[~bool_train_labels]

pos_labels = train_labels[bool_train_labels]
neg_labels = train_labels[~bool_train_labels]

# 使用 NumPy
# 您可以通过从正示例中选择正确数量的随机索引来手动平衡数据集：


ids = np.arange(len(pos_features))
choices = np.random.choice(ids, len(neg_features))

res_pos_features = pos_features[choices]
res_pos_labels = pos_labels[choices]

res_pos_features.shape

# (181953, 29)

resampled_features = np.concatenate([res_pos_features, neg_features], axis=0)
resampled_labels = np.concatenate([res_pos_labels, neg_labels], axis=0)

order = np.arange(len(resampled_labels))
np.random.shuffle(order)
resampled_features = resampled_features[order]
resampled_labels = resampled_labels[order]

resampled_features.shape

# (363906, 29)


# 使用tf.data
# 如果您使用tf.data生成平衡示例的最简单方法是从 apositive和一个negative数据集开始，然后合并它们。有关更多示例，请参阅tf.data 指南(https://tensorflow.google.cn/guide/data)。


BUFFER_SIZE = 100000

def make_ds(features, labels):
  ds = tf.data.Dataset.from_tensor_slices((features, labels))#.cache()
  ds = ds.shuffle(BUFFER_SIZE).repeat()
  return ds

pos_ds = make_ds(pos_features, pos_labels)
neg_ds = make_ds(neg_features, neg_labels)
# 每个数据集提供(feature, label)对：


for features, label in pos_ds.take(1):
  print("Features:\n", features.numpy())
  print('')
  print("Label: ", label.numpy())

# Features:
#  [-2.25920707 -0.0072081  -3.40638234  4.89298529  3.87783585 -3.25107162
#  -2.14285219  0.28594133 -3.52636688 -3.65233852  5.         -5.
#  -1.1397116  -5.          0.28999929 -1.26264317  0.74955735  1.1855264
#  -0.84786706 -0.13985126  0.33724841 -0.06842706 -0.3580829  -0.65930289
#   1.63937591  1.36920315  1.22563042  2.86007288 -1.4516158 ]

# Label:  1
# 使用将两者合并在一起tf.data.Dataset.sample_from_datasets：


resampled_ds = tf.data.experimental.sample_from_datasets([pos_ds, neg_ds], weights=[0.5, 0.5])
resampled_ds = resampled_ds.batch(BATCH_SIZE).prefetch(2)

for features, label in resampled_ds.take(1):
  print(label.numpy().mean())

# 0.498046875
# 要使用此数据集，您需要每个 epoch 的步数。

# 在这种情况下，“epoch”的定义不太清楚。假设它是查看每个负例一次所需的批次数：


resampled_steps_per_epoch = np.ceil(2.0*neg/BATCH_SIZE)
resampled_steps_per_epoch

# 训练过采样数据
# 现在尝试使用重新采样的数据集而不是使用类权重来训练模型，以查看这些方法的比较情况。
#
# 注意：因为通过复制正例来平衡数据，所以总数据集大小更大，并且每个 epoch 运行更多的训练步骤。

resampled_model = make_model()
resampled_model.load_weights(initial_weights)

# Reset the bias to zero, since this dataset is balanced.
output_layer = resampled_model.layers[-1]
output_layer.bias.assign([0])

val_ds = tf.data.Dataset.from_tensor_slices((val_features, val_labels)).cache()
val_ds = val_ds.batch(BATCH_SIZE).prefetch(2)

resampled_history = resampled_model.fit(
    resampled_ds,
    epochs=EPOCHS,
    steps_per_epoch=resampled_steps_per_epoch,
    callbacks=[early_stopping],
    validation_data=val_ds)

# 如果训练过程在每次梯度更新时考虑整个数据集，那么这种过采样将与类加权基本相同。
# 但是在批量训练模型时，正如您在此处所做的那样，过采样数据提供了更平滑的梯度信号：不是每个正样本都以较大的权重显示在一个批次中，而是每次都以许多不同的批次显示重量小。

# 这种更平滑的梯度信号使模型更容易训练。

# 查看训练历史
# 请注意，这里的指标分布会有所不同，因为训练数据与验证数据和测试数据的分布完全不同。


plot_metrics(resampled_history)

# 重新训练
# 由于在平衡数据上训练更容易，上述训练过程可能会很快过拟合。
#
# 因此，拆分 epoch 以tf.keras.callbacks.EarlyStopping更好地控制何时停止训练。


resampled_model = make_model()
resampled_model.load_weights(initial_weights)

# Reset the bias to zero, since this dataset is balanced.
output_layer = resampled_model.layers[-1]
output_layer.bias.assign([0])

resampled_history = resampled_model.fit(
    resampled_ds,
    # These are not real epochs
    steps_per_epoch=20,
    epochs=10*EPOCHS,
    callbacks=[early_stopping],
    validation_data=(val_ds))

# 重新检查训练历史

plot_metrics(resampled_history)

# 评估指标

train_predictions_resampled = resampled_model.predict(train_features, batch_size=BATCH_SIZE)
test_predictions_resampled = resampled_model.predict(test_features, batch_size=BATCH_SIZE)


resampled_results = resampled_model.evaluate(test_features, test_labels,
                                             batch_size=BATCH_SIZE, verbose=0)
for name, value in zip(resampled_model.metrics_names, resampled_results):
  print(name, ': ', value)
print('')

plot_cm(test_labels, test_predictions_resampled)

# 绘制 ROC

plot_roc("Train Baseline", train_labels, train_predictions_baseline, color=colors[0])
plot_roc("Test Baseline", test_labels, test_predictions_baseline, color=colors[0], linestyle='--')

plot_roc("Train Weighted", train_labels, train_predictions_weighted, color=colors[1])
plot_roc("Test Weighted", test_labels, test_predictions_weighted, color=colors[1], linestyle='--')

plot_roc("Train Resampled", train_labels, train_predictions_resampled, color=colors[2])
plot_roc("Test Resampled", test_labels, test_predictions_resampled, color=colors[2], linestyle='--')
plt.legend(loc='lower right');


# 绘制 AUPRC

plot_prc("Train Baseline", train_labels, train_predictions_baseline, color=colors[0])
plot_prc("Test Baseline", test_labels, test_predictions_baseline, color=colors[0], linestyle='--')

plot_prc("Train Weighted", train_labels, train_predictions_weighted, color=colors[1])
plot_prc("Test Weighted", test_labels, test_predictions_weighted, color=colors[1], linestyle='--')

plot_prc("Train Resampled", train_labels, train_predictions_resampled, color=colors[2])
plot_prc("Test Resampled", test_labels, test_predictions_resampled, color=colors[2], linestyle='--')
plt.legend(loc='lower right');

# 不平衡的数据分类本质上是一项艰巨的任务，因为要学习的样本很少。
# 您应该始终首先从数据开始，并尽最大努力收集尽可能多的样本，并充分考虑哪些特征可能相关，以便模型可以充分利用您的少数类。
# 在某些时候，您的模型可能难以改进并产生您想要的结果，因此请务必牢记问题的背景以及不同类型错误之间的权衡。

# 最后整个程序跑完，感觉通过设置类别权重，或者过采样的结果，还不如一开始的基线模型；

def main():
    pass


if __name__ == '__main__':
    main()