#!/usr/bin/env python3
# -*- coding: utf-8 -*-


def get_xy_fd(hash_flag=False):
    user_feature_columns = [SparseFeat('user', 3), SparseFeat(
        'gender', 2), VarLenSparseFeat(
        SparseFeat('hist_item', vocabulary_size=3 + 1, embedding_dim=4, embedding_name='item'), maxlen=4,
        length_name="hist_len")]
    item_feature_columns = [SparseFeat('item', 3 + 1, embedding_dim=4, )]

    uid = np.array([0, 1, 2, 1])
    ugender = np.array([0, 1, 0, 1])
    iid = np.array([1, 2, 3, 1])  # 0 is mask value

    hist_iid = np.array([[1, 2, 3, 0], [1, 2, 3, 0], [1, 2, 0, 0], [3, 0, 0, 0]])
    hist_len = np.array([3, 3, 2, 1])

    feature_dict = {'user': uid, 'gender': ugender, 'item': iid,
                    'hist_item': hist_iid, "hist_len": hist_len}

    # feature_names = get_feature_names(feature_columns)
    x = feature_dict
    y = np.array([1, 0, 1, 1])
    return x, y, user_feature_columns, item_feature_columns

x, y, user_feature_columns, item_feature_columns = get_xy_fd()
model = DeepFM(user_feature_columns, item_feature_columns, task='binary', dnn_dropout=0.2) #调用deepctr库中的DeepFM模型，执行二分类任务
# model = DCNMix(linear_feature_columns, dnn_feature_columns,task='binary', dnn_dropout=0.2) #调用deepctr库中的DCNMix模型，执行二分类任务
# model = DIN(linear_feature_columns, dnn_feature_columns,task='binary', dnn_dropout=0.2) #调用deepctr库中的DIN模型，执行二分类任务
model.compile("adam", "binary_crossentropy",
              metrics=['binary_crossentropy', 'accuracy'], ) #设置优化器，损失函数类型和评估指标

history = model.fit(x, y, batch_size=2, epochs=1, verbose=1, validation_split=0.2, )

def main():
    pass


if __name__ == '__main__':
    main()
