

from keras.models import Model
from keras.layers import Input, Dense, Concatenate, Reshape, Dropout, TimeDistributed
from keras.layers.embeddings import Embedding
from keras.preprocessing import sequence
from keras.callbacks import EarlyStopping, ModelCheckpoint
from keras.models import Sequential
from keras.layers.core import Dense, Dropout, Activation
from keras.layers import Dense,LSTM,Bidirectional
from keras.layers.recurrent import LSTM
from keras.callbacks import TensorBoard

# keras.__version__, tf.__version__
# ('2.2.4', '1.14.0')

# 双向LSTM 
input_1 = Input(shape=(6, 44,), name="input_1")    
lstm = LSTM(64,input_shape=(6,44),return_sequences=False) # 由于LSTM层设置了 return_sequences=False，只有最后一个节点的输出值会返回，每层LSTM返回64维向量，两层合并共128维，因此输出尺寸为 (None, 128)  
bilstm = Bidirectional(lstm)(input_1) 
dense = Dense(10,activation='softmax')(bilstm)

inputs = [input_1]
outputs = [dense]
model = Model(inputs, outputs)
model.compile(optimizer='adam',loss='categorical_crossentropy',metrics=['accuracy'])
model.summary()    

_________________________________________________________________
Layer (type)                 Output Shape              Param #   
=================================================================
input_1 (InputLayer)         (None, 6, 44)             0         
_________________________________________________________________
bidirectional_2 (Bidirection (None, 128)               55808     
_________________________________________________________________
dense_2 (Dense)              (None, 10)                1290      
=================================================================
Total params: 57,098
Trainable params: 57,098
Non-trainable params: 0
_________________________________________________________________


# 双层LSTM

input_1 = Input(shape=(6, 44,), name="input_1")    
lstm1 = LSTM(64,input_shape=(6,44),return_sequences=True)(input_1)  
lstm2 = LSTM(32,return_sequences=False)(lstm1)  
dense = Dense(10,activation='softmax')(lstm2)
inputs = [input_1]
outputs = [dense]
model = Model(inputs, outputs)
model.compile(optimizer='adam',loss='categorical_crossentropy',metrics=['accuracy'])
model.summary()       

# 由于第一个LSTM层设置了 return_sequences=True，每个节点的输出值都会返回，因此输出尺寸为 (None, 6, 64) 
# 由于第二个LSTM层设置了 return_sequences=False，只有最后一个节点的输出值会返回，因此输出尺寸为 (None, 32) 

_________________________________________________________________
Layer (type)                 Output Shape              Param #   
=================================================================
input_1 (InputLayer)         (None, 6, 44)             0         
_________________________________________________________________
lstm_7 (LSTM)                (None, 6, 64)             27904     
_________________________________________________________________
lstm_8 (LSTM)                (None, 32)                12416     
_________________________________________________________________
dense_3 (Dense)              (None, 10)                330       
=================================================================
Total params: 40,650
Trainable params: 40,650
Non-trainable params: 0
_________________________________________________________________

