#!/usr/bin/python3
# coding: utf-8

# https://www.twblogs.net/a/5bde2b792b717720b51b93e1
# https://github.com/BenShuai/kerasTfPoj.git
import re
import os
import hashlib
import json
import random
import numpy as np
import pickle
import wave
from keras.layers import Dense, Dropout, Bidirectional, LSTM, Reshape
from keras.models import Sequential
import keras.losses
import keras.optimizers
import keras.utils
from keras.callbacks import ModelCheckpoint
from keras import regularizers

# 训练数据： https://pan.baidu.com/s/1Au85kI_oeDjode2hWumUvQ

SPEECH_FILE_PATH = '/home/gswyhq/data/speech_commands_v0.01'
TRAIN_TEST_SPLIT_FILE_PATH = '/home/gswyhq/data/speech_commands_v0.01/train_test_split.json'

CLASS_TAGS = ['cat', 'eight', 'go', 'left', 'nine', 'on', 'right', 'six', 'three', 'up', 'yes', 'bed', 'dog', 'five', 'happy', 'no', 'one', 'seven', 'stop', 'tree', 'zero', 'bird', 'down', 'four', 'house', 'marvin', 'off', 'sheila', 'two', 'wow']
# CLASS_TAGS = CLASS_TAGS[:3]

MAX_NUM_WAVS_PER_CLASS = 2**27 - 1  # ~134M

def which_set(filename, validation_percentage=10, testing_percentage=10):
  """确定文件应属于哪个数据分区。
    文件路径为“happy / 3cfc6b3a_nohash_2.wav”
    表示所说的单词是“happy”，发言者的id是“3cfc6b3a”，并且
    这是数据集中该发言者对该词的第三个话语。
    
  如果随着时间的推移添加新的,我们希望将文件保存在相同的训练，验证或测试集中。这使得测试的可能性降低
  重启长跑时，不小心会在训练中重复使用样本
  例如。为了保持这种稳定性，需要使用文件名的哈希值确定它应该属于哪个集合。这个决定只取决于
  名称和设置比例，因此不会随着其他文件的添加而改变。

  将特定文件关联起来（例如单词）也很有用由同一个人说出来，所以文件名中的'_nohash_'之后的任何内容都是设置确定被忽略。
    这确保'bobby_nohash_0.wav'和例如，'bobby_nohash_1.wav'总是在同一个集合中。

  ARGS：
    filename：数据样本的文件路径。
    validation_percentage：用于验证的数据集的数量。
    testing_percentage：用于测试的数据集的数量。

  Returns:
    String, one of 'training', 'validation', or 'testing'.
  """
  base_name = os.path.basename(filename)
  hash_name = re.sub(r'_nohash_.*$', '', base_name)
  hash_name_hashed = hashlib.sha1(hash_name.encode('utf-8')).hexdigest()
  percentage_hash = ((int(hash_name_hashed, 16) %
                      (MAX_NUM_WAVS_PER_CLASS + 1)) *
                     (100.0 / MAX_NUM_WAVS_PER_CLASS))
  if percentage_hash < validation_percentage:
    result = 'validation'
  elif percentage_hash < (testing_percentage + validation_percentage):
    result = 'testing'
  else:
    result = 'training'
  return result


def train_test_split(data_path=SPEECH_FILE_PATH, load_save=False):
    data = {}
    if load_save and os.path.isfile(TRAIN_TEST_SPLIT_FILE_PATH):
        with open(TRAIN_TEST_SPLIT_FILE_PATH, 'r')as f:
            data = json.load(f)
    else:
        for _class in CLASS_TAGS:
            file_path = os.path.join(data_path, _class)
            file_lists = os.listdir(file_path)
            for filename in file_lists:
                tag = which_set(filename, validation_percentage=10, testing_percentage=10)
                data.setdefault(tag, [])
                data[tag].append(os.path.join(file_path, filename))
        with open(TRAIN_TEST_SPLIT_FILE_PATH, 'w')as f:
            json.dump(data, f)

    return data

def get_wav_mfcc(wav_path):
    '''
    獲取語音文件的fmcc值
    :param wav_path: 
    :return: 
    '''
    f = wave.open(wav_path,'rb')
    params = f.getparams()
    # print("params:",params)
    nchannels, sampwidth, framerate, nframes = params[:4]
    strData = f.readframes(nframes)#讀取音頻，字符串格式
    waveData = np.fromstring(strData,dtype=np.int16)#將字符串轉化爲int
    waveData = waveData*1.0/(max(abs(waveData)))#wave幅值歸一化
    if np.isnan(waveData).any():
        # /home/gswyhq/data/speech_commands_v0.01/bird/3e7124ba_nohash_0.wav
        print("{} 文件有问题；".format(wav_path))
        return None
    waveData = np.reshape(waveData,[nframes,nchannels]).T
    f.close()

    ### 對音頻數據進行長度大小的切割，保證每一個的長度都是一樣的
    #【因爲訓練文件全部是1秒鐘長度，16000幀的，所以這裏需要把每個語音文件的長度處理成一樣的】
    data = list(np.array(waveData[0]))
    # print(len(data))
    while len(data)>16000:
        del data[len(waveData[0])-1]
        del data[0]
    # print(len(data))
    while len(data)<16000:
        data.append(0)
    # print(len(data))

    # 爲了在訓練的時候避免損失函數應爲負數導致輸出結果相差太大，需要把原始的mfcc全部轉爲正數，直接平方後在開方就是正值了。
    data=np.array(data)
    # 平方之後，開平方，取正數，值的範圍在  0-1  之間
    data = data ** 2
    data = data ** 0.5
    return data

# 訓練之前需要先讀取數據創建數據集和標籤集：
# 加載數據集 和 標籤[並返回標籤集的處理結果]
def create_datasets(data, split_num=0.2):
    # training 51088 /home/gswyhq/data/speech_commands_v0.01/cat/89ed36ab_nohash_0.wav
    # validation 6798 /home/gswyhq/data/speech_commands_v0.01/cat/9cde5de8_nohash_0.wav
    # testing 6835 /home/gswyhq/data/speech_commands_v0.01/cat/cc592808_nohash_1.wav
    result = {}
    for tag, files in data.items():
        wavs=[]
        labels=[] # labels 和 testlabels 這裏面存的值都是對應標籤的下標，下標對應的名字在 labsInd 和 testlabsInd 中
        random.shuffle(files)
        for filename in files[:int(len(files) * split_num)]:
            waveData = get_wav_mfcc(filename)
            if waveData is None:continue
            wavs.append(waveData)
            _class = filename.split('/')[-2]
            labels.append(CLASS_TAGS.index(_class))

        wavs=np.array(wavs)
        labels=np.array(labels)
        result.setdefault(tag, (wavs, labels))

    with open('/home/gswyhq/data/speech_commands_v0.01/train_test_data.pkl', 'wb')as f:
        pickle.dump(result, f)
    # with open('/home/gswyhq/data/speech_commands_v0.01/train_test_data.pkl', 'rb')as f:
    #   result = pickle.load(f, encoding='iso-8859-1')
    return result

def generator_datasets(data, tag, batch_size=1):
    # training 51088 /home/gswyhq/data/speech_commands_v0.01/cat/89ed36ab_nohash_0.wav
    # validation 6798 /home/gswyhq/data/speech_commands_v0.01/cat/9cde5de8_nohash_0.wav
    # testing 6835 /home/gswyhq/data/speech_commands_v0.01/cat/cc592808_nohash_1.wav
    # result = {}
    # for tag, files in data.items():
    files = data[tag]
    random.shuffle(files)
    while True:

        wavs = []
        labels = []  # labels 和 testlabels 這裏面存的值都是對應標籤的下標，下標對應的名字在 labsInd 和 testlabsInd 中
        for filename in files:
            waveData = get_wav_mfcc(filename)
            if waveData is None:continue
            wavs.append(waveData)
            _class = filename.split('/')[-2]
            labels.append(CLASS_TAGS.index(_class))

            # label = CLASS_TAGS.index(_class)

            if len(wavs) >= batch_size:
                wavs = np.array(wavs)
                labels = np.array(labels)
                labels = keras.utils.to_categorical(labels, len(CLASS_TAGS))
                yield wavs, labels
                wavs = []
                labels = []  # labels 和 testlabels 這裏面存的值都是對應標籤的下標，下標對應的名字在 labsInd 和 testlabsInd 中

# 拿到數據集之後就可以開始進行神經網絡的訓練了，keras提供了很多封裝好的可以直接使用的神經網絡，我們先建立神經網絡模型
def build_model():
    # 構建一個4層的模型
    model = Sequential()
    model.add(Dropout(0.2, input_shape=(16000,))) # 音頻爲16000幀的數據，這裏的維度就是16000，激活函數直接用常用的relu
    # print("model.output_shape: {}".format(model.output_shape))
    model.add(Dense(1024, activation='relu',
                kernel_regularizer=regularizers.l2(0.005)  # 0.001, 过拟合; 0.01, 欠拟合
                # bias_regularizer=regularizers.l1_l2(l1=0.01, l2=0.01),
                # activity_regularizer=regularizers.l1(0.001)
                    ))
    model.add(Dropout(0.2))

    model.add(Dense(512, activation='relu',
                kernel_regularizer=regularizers.l2(0.005)  # 0.001, 过拟合; 0.01, 欠拟合
                # bias_regularizer=regularizers.l1_l2(l1=0.01, l2=0.01),
                # activity_regularizer=regularizers.l1(0.001)
                    ))
    # 正则化,regularizer是正则项(优化过程中层的参数或输出值添加的惩罚项)
    # Kernel_regularizer 权值正则化
    # Bias_regularizer 偏置正则化
    # Activity_regularizer 激活正则化
    # 激活正则化是信号乘以权值加上偏置值得到的激活
    # 在神经网络中，参数包括每一层仿射变换的权重和偏置，我们通常只对权重做惩罚而不对偏置做正则惩罚。
    # 精确拟合偏置所需的数据通常比拟合权重少得多。每个权重会指定两个变量如何相互作用。我们需要在各种条件下观察这两个变量才能良好地拟合权重。
    # 而每个偏置仅控制一个单变量。这意味着，我们不对其进行正则化也不会导致太大的方差。
    # 另外，正则化偏置参数可能会导致明显的欠拟合。

    model.add(Dropout(0.2))
    model.add(Dense(256, activation='relu', init='normal',
                kernel_regularizer=regularizers.l2(0.0001)
                # bias_regularizer=regularizers.l1_l2(l1=0.01, l2=0.01),
                # activity_regularizer=regularizers.l1(0.001)
                    ))

    model.add(Dropout(0.2))
    model.add(Dense(64, activation='tanh', init='normal',
                # kernel_regularizer=regularizers.l2(0.001),
                # bias_regularizer=regularizers.l1_l2(l1=0.01, l2=0.01),
                # activity_regularizer=regularizers.l1(0.001)
                    ))
    model.add(Dropout(0.2))
    model.add(Dense(len(CLASS_TAGS), init='normal', activation='softmax'))
    # 二分类与多分类在前面的结构上都没有问题，就是需要改一下最后的全连接层，因为此时有5分类，所以需要Dense(5)，
    # 同时激活函数是softmax，如果是二分类就是dense(2)+sigmoid(激活函数)。
    # [編譯模型] 配置模型，損失函數採用交叉熵，優化採用Adadelta，將識別準確率作爲模型評估
    model.compile(loss=keras.losses.categorical_crossentropy, optimizer=keras.optimizers.Adadelta(), metrics=['accuracy'])
    print(model.summary())
    return model


def train(wavs, labels, valwavs, vallabels, testwavs, testlabels, batch_size=124, epochs=5, verbose=1):
    #  validation_data爲驗證集
    filepath = '/home/gswyhq/data/speech_commands_v0.01/model-ep{epoch:03d}-loss{loss:.3f}-val_loss{val_loss:.3f}.h5'
    checkpoint = ModelCheckpoint(filepath, monitor='val_loss', verbose=1, save_best_only=True, save_weights_only=False, mode='auto')
    # mode：‘auto’，‘min’，‘max’之一，在min模式下，如果检测值停止下降则中止训练。在max模式下，当检测值不再上升则停止训练。
    # 对于`val_acc`，这应该是`max`，对于`val_loss`这应该是是`min`等。在`auto`模式下，方向是从监控数量的名称自动推断。
    model = build_model()
    model.fit(wavs, labels, batch_size=batch_size, epochs=epochs, verbose=verbose, validation_data=(valwavs, vallabels),
              callbacks=[checkpoint]) ## 進行5輪訓練，每個批次124個

    # 開始評估模型效果 # verbose=0爲不輸出日誌信息
    score = model.evaluate(testwavs, testlabels, verbose=0)
    print('Test loss:', score[0])
    print('Test accuracy:', score[1]) # 準確度

    # 最後保存模型到文件：

    # model.save('/home/gswyhq/data/speech_commands_v0.01/asr_model_weights.h5') # 保存訓練模型
    return model

def generator_train(data, batch_size=124, epochs=5, verbose=1):
    #  validation_data爲驗證集
    filepath = '/home/gswyhq/data/speech_commands_v0.01/model-ep{epoch:03d}-loss{loss:.3f}-val_loss{val_loss:.3f}-acc{acc:.4f}.h5'
    checkpoint = ModelCheckpoint(filepath, monitor='val_loss', verbose=1, save_best_only=True, save_weights_only=False, mode='auto')
    # mode：‘auto’，‘min’，‘max’之一，在min模式下，如果检测值停止下降则中止训练。在max模式下，当检测值不再上升则停止训练。
    # 对于`val_acc`，这应该是`max`，对于`val_loss`这应该是是`min`等。在`auto`模式下，方向是从监控数量的名称自动推断。
    model = build_model()
    # 'training', 'validation', 'testing'
    history = model.fit_generator(generator_datasets(data, 'training', batch_size=batch_size),
                                  steps_per_epoch=int(51088/batch_size),
                                  # batch_size=batch_size,
                                  epochs=epochs,
                                  verbose=verbose,
                                  validation_data=generator_datasets(data, 'validation', batch_size=batch_size),
                                  validation_steps = int(6798/batch_size),
                                  callbacks=[checkpoint])

    # 開始評估模型效果 # verbose=0爲不輸出日誌信息
    score = model.evaluate_generator(generator_datasets(data, 'testing'), steps=6835, max_queue_size=10, workers=1, use_multiprocessing=False, verbose=0)
    print('Test loss:', score[0])
    print('Test accuracy:', score[1]) # 準確度

    # 最後保存模型到文件：

    # model.save('/home/gswyhq/data/speech_commands_v0.01/asr_model_weights.h5') # 保存訓練模型
    return history

# 先在訓練的模型已經有了，我們開始使用trunk中的文件進行試驗：
#
# 先加載之前訓練的模型：

# model = load_model('asr_model_weights.h5') # 加載訓練模型
#
# # 然後獲得當前需要試驗的文件的mfcc。並且將數據封裝成和訓練時一樣的維度。並且使用模型的predict函數輸出結果：
#
# wavs=[]
# wavs.append(get_wav_mfcc("D:\\wav\\trunk\\2c.wav")) # 使用某一個文件
# X=np.array(wavs)
# print(X.shape)
# result=model.predict(X[0:1])[0] # 識別出第一張圖的結果，多張圖的時候，把後面的[0] 去掉，返回的就是多張圖結果
# print("識別結果",result)
# # 結果輸出：
# # 在這裏插入圖片描述
# # 可以看出結果是一個2個數的數組，裏面返回的對應類別相似度，也就是說哪一個下標的值最大，就跟那個下標對應的標籤最相似。
# #
# # 之前訓練的時候，標籤的集是：[seven , stop]
# #
# # 所以如圖下標1的值達到了89.9%的相似度。
#
# #  因爲在訓練的時候，標籤集的名字 爲：  0：seven   1：stop    0 和 1 是下標
# name = ["seven","stop"] # 創建一個跟訓練時一樣的標籤集
# ind=0 # 結果中最大的一個數
# for i in range(len(result)):
#     if result[i] > result[ind]:
#         ind=1
# print("識別的語音結果是：",name[ind])
# # 在這裏插入圖片描述
#
# # 我們把試驗文件換成 1b.wav
#
# wavs.append(get_wav_mfcc("D:\\wav\\trunk\\1b.wav"))
# # 結果輸出：
# # 在這裏插入圖片描述
# # 本機的試驗的識別速度在2秒內。
#
# 本文相關的代碼已上傳github：https://github.com/BenShuai/kerasTfPoj/tree/master/kerasTfPoj/ASR

def main():
    data = train_test_split(data_path=SPEECH_FILE_PATH, load_save=False)
    # train_data = create_datasets(data, split_num=0.2)
    # wavs, labels = train_data['training']
    # testwavs, testlabels = train_data['testing']
    # valwavs, vallabels = train_data['validation']
    # labels = keras.utils.to_categorical(labels, len(CLASS_TAGS))
    # vallabels = keras.utils.to_categorical(vallabels, len(CLASS_TAGS))
    # testlabels = keras.utils.to_categorical(testlabels, len(CLASS_TAGS))
    #
    # model = train(wavs, labels, valwavs, vallabels, testwavs, testlabels, batch_size=12, epochs=5, verbose=1)

    history = generator_train(data, batch_size=32, epochs=50, verbose=1)


if __name__ == '__main__':
    main()

# 412/412 [==============================] - 477s 1s/step - loss: 1.1251e-04 - acc: 1.0000 - val_loss: 2.5347 - val_acc: 0.6341
# Epoch 00005: val_loss did not improve from 1.03783
# Test loss: 2.796665355714285
# Test accuracy: 0.6327724945135332

# 410/412 [============================>.] - ETA: 1s - loss: 0.8723 - acc: 0.7428
# 411/412 [============================>.] - ETA: 0s - loss: 0.8721 - acc: 0.7429
# 412/412 [==============================] - 256s 621ms/step - loss: 0.8718 - acc: 0.7430 - val_loss: 1.2094 - val_acc: 0.5820
# 
# Epoch 00005: val_loss did not improve from 1.04681
# Test loss: 1.2647096929143795
# Test accuracy: 0.5635698610095099