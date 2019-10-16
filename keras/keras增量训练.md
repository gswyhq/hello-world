
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
