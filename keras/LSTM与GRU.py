

<keras.engine.functional.Functional at 0x7fb3ecb8a910>

def build_model5(sequence_length=8):
    inputs = keras.Input(shape=(sequence_length, 128))
    x = layers.LSTM(32, recurrent_dropout=0.5, return_sequences=True)(inputs)
    x = layers.LSTM(32, recurrent_dropout=0.5)(x)
    x = layers.Dropout(0.5)(x)
    outputs = layers.Dense(1)(x)
    model = keras.Model(inputs, outputs)
    model.compile(optimizer="rmsprop", loss="mse", metrics=["mae"])
    model.summary()
    return model
​
keras.backend.clear_session()
build_model5()
Model: "model"
_________________________________________________________________
Layer (type)                 Output Shape              Param #   
=================================================================
input_1 (InputLayer)         [(None, 8, 128)]          0         
_________________________________________________________________
lstm (LSTM)                  (None, 8, 32)             20608     
_________________________________________________________________
lstm_1 (LSTM)                (None, 32)                8320      
_________________________________________________________________
dropout (Dropout)            (None, 32)                0         
_________________________________________________________________
dense (Dense)                (None, 1)                 33        
=================================================================
Total params: 28,961
Trainable params: 28,961
Non-trainable params: 0
_________________________________________________________________
<keras.engine.functional.Functional at 0x7fb3ee541430>

build_model6(sequence_length=8):
    inputs = keras.Input(shape=(sequence_length, 128))
    
def build_model6(sequence_length=8):
    inputs = keras.Input(shape=(sequence_length, 128))
    x = layers.GRU(32, recurrent_dropout=0.5, return_sequences=True)(inputs)
    x = layers.GRU(32, recurrent_dropout=0.5)(x)
    x = layers.Dropout(0.5)(x)
    outputs = layers.Dense(1)(x)
    model = keras.Model(inputs, outputs)
    model.compile(optimizer="rmsprop", loss="mse", metrics=["mae"])
    model.summary()
    return model
​
keras.backend.clear_session()
build_model6()
Model: "model"
_________________________________________________________________
Layer (type)                 Output Shape              Param #   
=================================================================
input_1 (InputLayer)         [(None, 8, 128)]          0         
_________________________________________________________________
gru (GRU)                    (None, 8, 32)             15552     
_________________________________________________________________
gru_1 (GRU)                  (None, 32)                6336      
_________________________________________________________________
dropout (Dropout)            (None, 32)                0         
_________________________________________________________________
dense (Dense)                (None, 1)                 33        
=================================================================
Total params: 21,921
Trainable params: 21,921
Non-trainable params: 0
_________________________________________________________________
<keras.engine.functional.Functional at 0x7fb3ee168a00>

