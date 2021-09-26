#!/usr/bin/python3

# 基于keras的seq2seq中英文翻译实现
# https://blog.csdn.net/PIPIXIU/article/details/81016974
# http://www.manythings.org/anki/cmn-eng.zip

import pandas as pd
import numpy as np

# 2.1 数据处理
# 首先要将数据处理成Keras中模型接受的三维向量。这里需要处理3个向量，分别是encoder的输入encoder_input，decoder的输入和输出decoder_input， decoder_output

 #读取cmn-eng.txt文件
data_path = '/home/gswyhq/data/cmn-eng/cmn.txt'
df = pd.read_table(data_path,header=None).iloc[:NUM_SAMPLES,:,]
df.columns=['inputs','targets']
#讲每句中文句首加上'\t'作为起始标志，句末加上'\n'作为终止标志
df['targets'] = df['targets'].apply(lambda x: '\t'+x+'\n')

input_texts = df.inputs.values.tolist()#英文句子列表
target_texts = df.targets.values.tolist()#中文句子列表

#确定中英文各自包含的字符。df.unique()直接取sum可将unique数组中的各个句子拼接成一个长句子
input_characters = sorted(list(set(df.inputs.unique().sum())))
target_characters = sorted(list(set(df.targets.unique().sum())))

# 每条句子经过对字母转换成one-hot编码后，生成了LSTM需要的三维输入[n_samples, timestamp, one-hot feature]

#encoder输入、decoder输入输出初始化为三维向量
encoder_input = np.zeros((NUM_SAMPLES,INUPT_LENGTH,INPUT_FEATURE_LENGTH))
decoder_input = np.zeros((NUM_SAMPLES,OUTPUT_LENGTH,OUTPUT_FEATURE_LENGTH))
decoder_output = np.zeros((NUM_SAMPLES,OUTPUT_LENGTH,OUTPUT_FEATURE_LENGTH))


# 其中：
#
# NUM_SAMPLES，样本条数，这里是输入的句子条数
# INPUT_LENGTH，输入数据的时刻t的长度，这里为最长的英文句子长度
# OUTPUT_LENGTH，输出数据的时刻t的长度，这里为最长的中文句子长度
# INPUT_FEATURE_LENGTH，每个时刻进入encoder的lstm单元的数据xtxt的维度，这里为英文中出现的字符数
# OUTPUT_FEATURE_LENGTH，每个时刻进入decoder的lstm单元的数据xtxt的维度，这里为中文中出现的字符数
# 对句子进行字符级one-hot编码，将输入输出数据向量化：


#encoder的输入向量one-hot
for seq_index,seq in enumerate(input_texts):
    for char_index, char in enumerate(seq):
        encoder_input[seq_index,char_index,input_dict[char]] = 1
#decoder的输入输出向量one-hot，训练模型时decoder的输入要比输出晚一个时间步，这样才能对输出监督
for seq_index,seq in enumerate(target_texts):
    for char_index,char in enumerate(seq):
        decoder_input[seq_index,char_index,target_dict[char]] = 1.0
        if char_index > 0:
            decoder_output[seq_index,char_index-1,target_dict[char]] = 1.0


这里，查看获得的三个向量如下：

In[0]:  ''.join([input_dict_reverse[np.argmax(i)] for i in encoder_input[0] if max(i) !=0])
Out[0]: 'Hi.'

In[1]:  ''.join([input_dict_reverse[np.argmax(i)] for i in decoder_output[0] if max(i) !=0])
Out[1]: '嗨。\n'

In[2]:  ''.join([input_dict_reverse[np.argmax(i)] for i in decoder_input[0] if max(i) !=0])
Out[2]: '\t嗨。\n'
1
2
3
4
5
6
7
8
其中input_dict和target_dict为中英文字符与其索引的对应词典；input_dict_reverse和target_dict_reverse与之相反，索引为键字符为值：

input_dict = {char:index for index,char in enumerate(input_characters)}
input_dict_reverse = {index:char for index,char in enumerate(input_characters)}
target_dict = {char:index for index,char in enumerate(target_characters)}
target_dict_reverse = {index:char for index,char in enumerate(target_characters)}
1
2
3
4
2.2 encoder-decoder模型搭建
我们预测过程分为训练阶段和推理阶段，模型也分为训练模型和推断模型，这两个模型encoder之间deocder之间权重共享。
为什么要这么划分的？我们仔细考虑训练过程，会发现训练阶段和预测阶段的差异。
在训练阶段，encoder的输入为time series数据，输出为最终的隐状态，decoder的输出应该是target序列。为了有监督的训练，decoder输入应该是比输入晚一个时间步，这样在预测时才能准确的将下一个时刻的数据预测出来。
在训练阶段，每一时刻decoder的输入包含了上一时刻单元的状态ht−1ht−1和ct−1ct−1，输出则包含了本时刻的状态hh和cc以及经过全连接层之后的输出数据。
训练时的流程：

预测时的流程：


2.2.1 用于训练的模型

    #encoder
    encoder_input = Input(shape = (None, n_input))
    #encoder输入维度n_input为每个时间步的输入xt的维度，这里是用来one-hot的英文字符数
    encoder = LSTM(n_units, return_state=True)
    #n_units为LSTM单元中每个门的神经元的个数，return_state设为True时才会返回最后时刻的状态h,c(详见参考文献5,6)
    _,encoder_h,encoder_c = encoder(encoder_input)
    encoder_state = [encoder_h,encoder_c]
    #保留下来encoder的末状态作为decoder的初始状态

    #decoder
    decoder_input = Input(shape = (None, n_output))
    #decoder的输入维度为中文字符数
    decoder = LSTM(n_units,return_sequences=True, return_state=True)
    #训练模型时需要decoder的输出序列来与结果对比优化，故return_sequences也要设为True
    decoder_output, _, _ = decoder(decoder_input,initial_state=encoder_state)
    #在训练阶段只需要用到decoder的输出序列，不需要用最终状态h.c
    decoder_dense = Dense(n_output,activation='softmax')
    decoder_output = decoder_dense(decoder_output)
    #输出序列经过全连接层得到结果

    #生成的训练模型
    model = Model([encoder_input,decoder_input],decoder_output)
    #第一个参数为训练模型的输入，包含了encoder和decoder的输入，第二个参数为模型的输出，包含了decoder的输出
1
2
3
4
5
6
7
8
9
10
11
12
13
14
15
16
17
18
19
20
21
22
23
24
25
训练模型：
其中英文字符（含数字和符号）共有73个，中文中字符（含数字和符号）共有2623个。


2.2.2 用于推理的模型
 #推断模型-encoder，预测时对序列predict，生成的state给decoder
 encoder_infer = Model(encoder_input,encoder_state)
1
2
encoder推断模型



#推断模型-decoder
decoder_state_input_h = Input(shape=(n_units,))
decoder_state_input_c = Input(shape=(n_units,))
decoder_state_input = [decoder_state_input_h, decoder_state_input_c] #上个时刻的状态h,c

decoder_infer_output, decoder_infer_state_h, decoder_infer_state_c = decoder(decoder_input,initial_state=decoder_state_input)
decoder_infer_state = [decoder_infer_state_h, decoder_infer_state_c]#当前时刻得到的状态
decoder_infer_output = decoder_dense(decoder_infer_output)#当前时刻的输出
decoder_infer = Model([decoder_input]+decoder_state_input,[decoder_infer_output]+decoder_infer_state)
1
2
3
4
5
6
7
8
9
10
11
decoder推断模型


把模型创建整理成一个函数：

def create_model(n_input,n_output,n_units):
    #训练阶段
    encoder_input = Input(shape = (None, n_input))
    encoder = LSTM(n_units, return_state=True)
    _,encoder_h,encoder_c = encoder(encoder_input)
    encoder_state = [encoder_h,encoder_c]

    #decoder
    decoder_input = Input(shape = (None, n_output))
    decoder = LSTM(n_units,return_sequences=True, return_state=True)
    decoder_output, _, _ = decoder(decoder_input,initial_state=encoder_state)
    decoder_dense = Dense(n_output,activation='softmax')
    decoder_output = decoder_dense(decoder_output)

    #生成的训练模型
    model = Model([encoder_input,decoder_input],decoder_output)

    #推理阶段，用于预测过程
    encoder_infer = Model(encoder_input,encoder_state)

    decoder_state_input_h = Input(shape=(n_units,))
    decoder_state_input_c = Input(shape=(n_units,))
    decoder_state_input = [decoder_state_input_h, decoder_state_input_c]#上个时刻的状态h,c

    decoder_infer_output, decoder_infer_state_h, decoder_infer_state_c = decoder(decoder_input,initial_state=decoder_state_input)
    decoder_infer_state = [decoder_infer_state_h, decoder_infer_state_c]#当前时刻得到的状态
    decoder_infer_output = decoder_dense(decoder_infer_output)#当前时刻的输出
    decoder_infer = Model([decoder_input]+decoder_state_input,[decoder_infer_output]+decoder_infer_state)

    return model, encoder_infer, decoder_infer
1
2
3
4
5
6
7
8
9
10
11
12
13
14
15
16
17
18
19
20
21
22
23
24
25
26
27
28
29
30
2.3 模型评价
def predict_chinese(source,encoder_inference, decoder_inference, n_steps, features):
    #先通过推理encoder获得预测输入序列的隐状态
    state = encoder_inference.predict(source)
    #第一个字符'\t',为起始标志
    predict_seq = np.zeros((1,1,features))
    predict_seq[0,0,target_dict['\t']] = 1

    output = ''
    #开始对encoder获得的隐状态进行推理
    #每次循环用上次预测的字符作为输入来预测下一次的字符，直到预测出了终止符
    for i in range(n_steps):#n_steps为句子最大长度
        #给decoder输入上一个时刻的h,c隐状态，以及上一次的预测字符predict_seq
        yhat,h,c = decoder_inference.predict([predict_seq]+state)
        #注意，这里的yhat为Dense之后输出的结果，因此与h不同
        char_index = np.argmax(yhat[0,-1,:])
        char = target_dict_reverse[char_index]
        output += char
        state = [h,c]#本次状态做为下一次的初始状态继续传递
        predict_seq = np.zeros((1,1,features))
        predict_seq[0,0,char_index] = 1
        if char == '\n':#预测到了终止符则停下来
            break
    return output
1
2
3
4
5
6
7
8
9
10
11
12
13
14
15
16
17
18
19
20
21
22
23
对100个值进行预测：

for i in range(1000,1210):
    test = encoder_input[i:i+1,:,:]#i:i+1保持数组是三维
    out = predict_chinese(test,encoder_infer,decoder_infer,OUTPUT_LENGTH,OUTPUT_FEATURE_LENGTH)
    print(input_texts[i])
    print(out)
1
2
3
4
5
我们发现预测的都很准（废话，训练集能不准嘛！）
翻译结果：

 ————————————————
版权声明：本文为CSDN博主「PIPIXIU」的原创文章，遵循CC 4.0 by-sa版权协议，转载请附上原文出处链接及本声明。
原文链接：https://blog.csdn.net/PIPIXIU/article/details/81016974