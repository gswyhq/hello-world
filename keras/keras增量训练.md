
有的时候需要在原有基础上继续进行训练

在训练时保存对应的模型：
```python
    ...
    early_stopping = EarlyStopping(monitor='val_loss', patience=3)

    model_checkpoint = ModelCheckpoint(filepath=os.path.join(model_save_path,
                                                             'classify-{epoch:02d}-{val_loss:.2f}-{val_acc:.3f}.hdf5'),
                                       save_best_only=True, save_weights_only=False)

    tb = TensorBoard(log_dir=log_dir,  # log 目录
                     histogram_freq=0,  # 按照何等频率（epoch）来计算直方图，0为不计算
                     batch_size=32,  # 用多大量的数据计算直方图
                     write_graph=True,  # 是否存储网络结构图
                     write_grads=False,  # 是否可视化梯度直方图
                     write_images=False,  # 是否可视化参数
                     embeddings_freq=0,
                     embeddings_layer_names=None,
                     embeddings_metadata=None)


    model.fit_generator(
        train_D.__iter__(),
        # steps_per_epoch=len(train_D),
        steps_per_epoch = 300,
        epochs=30,
        validation_data=valid_D.__iter__(),
        # validation_steps=len(valid_D),
        validation_steps = 32,
        class_weight='auto',  # 设置class weight让每类的sample对损失的贡献相等。
        shuffle=True,
        callbacks=[early_stopping, model_checkpoint, tb]
    )
```    

需要继续训练的时候, 在 `compile`后再`load_weights`即可; 
另外在再次训练时`model.fit_generator`或`model.fit`函数需可以设置参数： initial_epoch

```python
    ...
    model = Model(model.input, output)

    model.compile(
        loss='categorical_crossentropy',
        optimizer=Adam(1e-5),  # 用足够小的学习率
        # optimizer=PiecewiseLinearLearningRate(Adam(1e-5), {1000: 1e-5, 2000: 6e-5}),
        metrics=['accuracy']
    )
    
    model.summary()
    
    model.load_weights(os.path.join(model_save_path, filepath))
    ...
    
    model.fit_generator(
        train_D.__iter__(),
        # steps_per_epoch=len(train_D),
        steps_per_epoch = 300,
        initial_epoch = 10,
        epochs=30,
        validation_data=valid_D.__iter__(),
        # validation_steps=len(valid_D),
        validation_steps = 32,
        class_weight='auto',  # 设置class weight让每类的sample对损失的贡献相等。
        shuffle=True,
        callbacks=[early_stopping, model_checkpoint, tb]
    )
```

当initial_epoch = 10, epochs=30, steps_per_epoch=300 时，再次训练时开始时的日志会是如下所示：

Total params: 15,603,785
Trainable params: 1,610,313
Non-trainable params: 13,993,472
__________________________________________________________________________________________________
Epoch 11/30

  1/300 [..............................] - ETA: 50:35 - loss: 0.4435 - acc: 0.8750
  2/300 [..............................] - ETA: 47:28 - loss: 0.3463 - acc: 0.9219
  3/300 [..............................] - ETA: 56:31 - loss: 0.3156 - acc: 0.9271

若没有中途退出训练，训练会在Epoch 30/30 之后停止。


增量训练：
```python
def generate_arrays_from_file(path, batch_size):
    input_1, input_2, output = [], [], []
    while True:
        with open(path) as f:
            for line in f:
                # 从文件中的每一行生成输入数据和标签的 numpy 数组，
                x1, x2, y = process_line(line)
                input_1.append(x1)
                input_2.append(x2)
                output.append(y)
                if len(input_1) == batch_size:
                    yield ({'input_1': input_1, 'input_2': input_2}, output)
                    input_1, input_2, output = [], [], []
        f.close()
        
historys = model.fit_generator(generate_arrays_from_file("train_data.txt", batch_size), 
                               steps_per_epoch=868,  # 在声明一个 epoch 完成并开始下一个 epoch 之前从 generator 产生的总步数（批次样本）。 它通常应该等于你的数据集的样本数量除以批量大小。len(train_data)/batch_size
                               validation_data=generate_arrays_from_file("dev_data.txt", batch_size), 
                               validation_steps=103,  # 仅当 validation_data 是一个生成器时才可用。 在停止前 generator 生成的总步数（样本批数）。len(dev_data)/batch_size
          epochs=20, 
#           batch_size=batch_size,  # 通过迭代器输入训练数据时候，不需要该项
          verbose=1,
          shuffle=True,
         )
print(historys.history)
```