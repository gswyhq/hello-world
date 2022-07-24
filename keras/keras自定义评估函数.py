

def Precision(y_true, y_pred):
    """精确率"""
    tp= K.sum(K.round(K.clip(y_true * y_pred, 0, 1)))  # true positives
    pp= K.sum(K.round(K.clip(y_pred, 0, 1))) # predicted positives
    precision = tp/ (pp+ K.epsilon())
    return precision
    
def Recall(y_true, y_pred):
    """召回率"""
    tp = K.sum(K.round(K.clip(y_true * y_pred, 0, 1))) # true positives
    pp = K.sum(K.round(K.clip(y_true, 0, 1))) # possible positives
    recall = tp / (pp + K.epsilon())
    return recall
 
def F1(y_true, y_pred):
    """F1-score"""
    precision = Precision(y_true, y_pred)
    recall = Recall(y_true, y_pred)
    f1 = 2 * ((precision * recall) / (precision + recall + K.epsilon()))
    return f1

def wubao(y_true, y_pred):
    """误报率：误报/所有预警"""
    precision = Precision(y_true, y_pred)
    return 1 - precision

def loubao(y_true, y_pred):
    """漏报率：漏报/所有升级"""
    recall = Recall(y_true, y_pred)
    return 1 - recall

# 普通变量转换为张量进行计算
tf.compat.v1.disable_eager_execution()
with tf.compat.v1.Session() as sess:
    print(sess.run(loubao(tf.constant(np.array([0, 1, 0, 0, 1, 1, 1, 1, 1, 1, 0, 0, 1, 0, 0]),dtype='float'), 
        tf.constant(np.array([0.0, 0.7, 0.0446, 0.0, 0.06668, 0.6279, 0.8957, 0.2, 1.0, 0.9, 0.61, 0.0, 0.0768, 0.109, 0.8]), dtype='float'))))

# [[5 2]
#  [3 5]]
# 误报率：0.286
# 漏报率：0.375
# 预警率：0.467

model.compile(loss='binary_crossentropy',optimizer='adam', metrics=['accuracy', wubao, loubao])

Train on 65415 samples, validate on 16054 samples
Epoch 1/3
37472/65415 [================>.............] - ETA: 2:57 - loss: 0.2666 - acc: 0.8760 - wubao: 0.1172 - loubao: 0.1274

