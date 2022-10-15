#!/usr/bin/env python3
# -*- coding: utf-8 -*-

# 来源： https://keras.io/examples/keras_recipes/bayesian_neural_networks/
import numpy as np
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
import tensorflow_datasets as tfds
import tensorflow_probability as tfp
import os
# 我们使用TensorFlow Datasets 中提供的Wine Quality(https://archive.ics.uci.edu/ml/datasets/wine+quality)数据集。我们使用 red wine 子集，其中包含 4,898 个示例。
# 该数据集有 11 个葡萄酒的数值物理化学特征，任务是预测葡萄酒的质量，这是一个介于 0 到 10 之间的分数。在这个例子中，我们将其视为回归任务。

# 创建训练和评估数据集
# 在这里，我们使用 加载wine_quality数据集tfds.load()，并将目标特征转换为浮点。
# 然后，我们打乱数据集并将其拆分为训练集和测试集。我们将第一个train_size示例作为训练拆分，其余的作为测试拆分。

# https://archive.ics.uci.edu/ml/machine-learning-databases/wine-quality/winequality-white.csv

def get_train_and_test_splits(train_size, batch_size=1):
    # We prefetch with a buffer the same size as the dataset because th dataset
    # is very small and fits into memory.

    # http://raw.githubusercontent.com/tensorflow/datasets/master/tensorflow_datasets/testing/metadata/wine_quality/white/1.0.0/dataset_info.json
    # https://archive.ics.uci.edu/ml/machine-learning-databases/wine-quality/
    # ls tensorflow_datasets/wine_quality/white/1.0.0/
    # dataset_info.json      winequality-white.csv  winequality.names
    dataset = (
        tfds.load(name="wine_quality", as_supervised=True, split="train", data_dir=r"D:\Users\{}\tensorflow_datasets".format(os.getenv('USERNAME')), download=False)
        .map(lambda x, y: (x, tf.cast(y, tf.float32)))
        .prefetch(buffer_size=dataset_size)
        .cache()
    )
    # We shuffle with a buffer the same size as the dataset.
    train_dataset = (
        dataset.take(train_size).shuffle(buffer_size=train_size).batch(batch_size)
    )
    test_dataset = dataset.skip(train_size).batch(batch_size)

    return train_dataset, test_dataset


# 编译、训练和评估模型
hidden_units = [8, 8]
learning_rate = 0.001


def run_experiment(model, loss, train_dataset, test_dataset):

    model.compile(
        optimizer=keras.optimizers.RMSprop(learning_rate=learning_rate),
        loss=loss,
        metrics=[keras.metrics.RootMeanSquaredError()],
    )

    print("Start training the model...")
    model.fit(train_dataset, epochs=num_epochs, validation_data=test_dataset)
    print("Model training finished.")
    _, rmse = model.evaluate(train_dataset, verbose=0)
    print(f"Train RMSE: {round(rmse, 3)}")

    print("Evaluating model performance...")
    _, rmse = model.evaluate(test_dataset, verbose=0)
    print(f"Test RMSE: {round(rmse, 3)}")


# 创建模型输入
FEATURE_NAMES = [
    "fixed acidity",
    "volatile acidity",
    "citric acid",
    "residual sugar",
    "chlorides",
    "free sulfur dioxide",
    "total sulfur dioxide",
    "density",
    "pH",
    "sulphates",
    "alcohol",
]


def create_model_inputs():
    inputs = {}
    for feature_name in FEATURE_NAMES:
        inputs[feature_name] = layers.Input(
            name=feature_name, shape=(1,), dtype=tf.float32
        )
    return inputs

# 实验一：标准神经网络
# 我们创建了一个标准的确定性神经网络模型作为基线。

def create_baseline_model():
    inputs = create_model_inputs()
    input_values = [value for _, value in sorted(inputs.items())]
    features = keras.layers.concatenate(input_values)
    features = layers.BatchNormalization()(features)

    # Create hidden layers with deterministic weights using the Dense layer.
    for units in hidden_units:
        features = layers.Dense(units, activation="sigmoid")(features)
    # The output is deterministic: a single point estimate.
    outputs = layers.Dense(units=1)(features)

    model = keras.Model(inputs=inputs, outputs=outputs)
    return model


# 让我们将 wine 数据集分成训练集和测试集，分别包含 85% 和 15% 的示例。

dataset_size = 4898
batch_size = 256
train_size = int(dataset_size * 0.85)
train_dataset, test_dataset = get_train_and_test_splits(train_size, batch_size)


# 现在让我们训练基线模型。我们使用MeanSquaredError 作为损失函数。

num_epochs = 100
mse_loss = keras.losses.MeanSquaredError()
baseline_model = create_baseline_model()
run_experiment(baseline_model, mse_loss, train_dataset, test_dataset)
# Start training the model...
# Epoch 1/100
# 17/17 [==============================] - 1s 53ms/step - loss: 37.5710 - root_mean_squared_error: 6.1294 - val_loss: 35.6750 - val_root_mean_squared_error: 5.9729
# Epoch 2/100
# 17/17 [==============================] - 0s 7ms/step - loss: 35.5154 - root_mean_squared_error: 5.9594 - val_loss: 34.2430 - val_root_mean_squared_error: 5.8518
# Epoch 3/100
# 17/17 [==============================] - 0s 7ms/step - loss: 33.9975 - root_mean_squared_error: 5.8307 - val_loss: 32.8003 - val_root_mean_squared_error: 5.7272
# Epoch 4/100
# 17/17 [==============================] - 0s 12ms/step - loss: 32.5928 - root_mean_squared_error: 5.7090 - val_loss: 31.3385 - val_root_mean_squared_error: 5.5981
# Epoch 5/100
# 17/17 [==============================] - 0s 7ms/step - loss: 30.8914 - root_mean_squared_error: 5.5580 - val_loss: 29.8659 - val_root_mean_squared_error: 5.4650
#
# ...
#
# Epoch 95/100
# 17/17 [==============================] - 0s 6ms/step - loss: 0.6927 - root_mean_squared_error: 0.8322 - val_loss: 0.6901 - val_root_mean_squared_error: 0.8307
# Epoch 96/100
# 17/17 [==============================] - 0s 6ms/step - loss: 0.6929 - root_mean_squared_error: 0.8323 - val_loss: 0.6866 - val_root_mean_squared_error: 0.8286
# Epoch 97/100
# 17/17 [==============================] - 0s 6ms/step - loss: 0.6582 - root_mean_squared_error: 0.8112 - val_loss: 0.6797 - val_root_mean_squared_error: 0.8244
# Epoch 98/100
# 17/17 [==============================] - 0s 6ms/step - loss: 0.6733 - root_mean_squared_error: 0.8205 - val_loss: 0.6740 - val_root_mean_squared_error: 0.8210
# Epoch 99/100
# 17/17 [==============================] - 0s 7ms/step - loss: 0.6623 - root_mean_squared_error: 0.8138 - val_loss: 0.6713 - val_root_mean_squared_error: 0.8193
# Epoch 100/100
# 17/17 [==============================] - 0s 6ms/step - loss: 0.6522 - root_mean_squared_error: 0.8075 - val_loss: 0.6666 - val_root_mean_squared_error: 0.8165
# Model training finished.
# Train RMSE: 0.809
# Evaluating model performance...
# Test RMSE: 0.816
# 我们从测试集中抽取一个样本，使用模型来获得对它们的预测。请注意，由于基线模型是确定性的，我们得到每个测试示例的单个 点估计预测，没有关于模型不确定性和预测的信息。

sample = 10
examples, targets = list(test_dataset.unbatch().shuffle(batch_size * 10).batch(sample))[
    0
]

predicted = baseline_model(examples).numpy()
for idx in range(sample):
    print(f"Predicted: {round(float(predicted[idx][0]), 1)} - Actual: {targets[idx]}")
# Predicted: 6.0 - Actual: 6.0
# Predicted: 6.2 - Actual: 6.0
# Predicted: 5.8 - Actual: 7.0
# Predicted: 6.0 - Actual: 5.0
# Predicted: 5.7 - Actual: 5.0
# Predicted: 6.2 - Actual: 7.0
# Predicted: 5.6 - Actual: 5.0
# Predicted: 6.2 - Actual: 6.0
# Predicted: 6.2 - Actual: 6.0
# Predicted: 6.2 - Actual: 7.0



# 实验二：贝叶斯神经网络（BNN）
# 对神经网络建模的贝叶斯方法的目标是捕捉认知不确定性，即由于训练数据有限而导致的模型适应度的不确定性。
# 这个想法是，贝叶斯方法不是学习神经网络中的特定权重（和偏差）值，而是学习权重分布 - 我们可以从中采样以产生给定输入的输出 - 以编码权重不确定性。
# 因此，我们需要定义这些权重的先验分布和后验分布，训练过程就是学习这些分布的参数。

# Define the prior weight distribution as Normal of mean=0 and stddev=1.
# Note that, in this example, the we prior distribution is not trainable,
# as we fix its parameters.
def prior(kernel_size, bias_size, dtype=None):
    n = kernel_size + bias_size
    prior_model = keras.Sequential(
        [
            tfp.layers.DistributionLambda(
                lambda t: tfp.distributions.MultivariateNormalDiag(
                    loc=tf.zeros(n), scale_diag=tf.ones(n)
                )
            )
        ]
    )
    return prior_model


# Define variational posterior weight distribution as multivariate Gaussian.
# Note that the learnable parameters for this distribution are the means,
# variances, and covariances.
def posterior(kernel_size, bias_size, dtype=None):
    n = kernel_size + bias_size
    posterior_model = keras.Sequential(
        [
            tfp.layers.VariableLayer(
                tfp.layers.MultivariateNormalTriL.params_size(n), dtype=dtype
            ),
            tfp.layers.MultivariateNormalTriL(n),
        ]
    )
    return posterior_model


# 我们在神经网络模型中使用tfp.layers.DenseVariational层而不是标准 层。keras.layers.Dense

def create_bnn_model(train_size):
    inputs = create_model_inputs()
    features = keras.layers.concatenate(list(inputs.values()))
    features = layers.BatchNormalization()(features)

    # Create hidden layers with weight uncertainty using the DenseVariational layer.
    for units in hidden_units:
        features = tfp.layers.DenseVariational(
            units=units,
            make_prior_fn=prior,
            make_posterior_fn=posterior,
            kl_weight=1 / train_size,
            activation="sigmoid",
        )(features)

    # The output is deterministic: a single point estimate.
    outputs = layers.Dense(units=1)(features)
    model = keras.Model(inputs=inputs, outputs=outputs)
    return model


# 随着我们增加训练数据的大小，可以减少认知不确定性。也就是说，BNN 模型看到的数据越多，它对权重（分布参数）的估计就越确定。让我们通过在训练集的一个小子集上训练 BNN 模型来测试这种行为，然后在整个训练集上训练，以比较输出方差。
#
# 用一个小的训练子集训练 BNN。
num_epochs = 500
train_sample_size = int(train_size * 0.3)
small_train_dataset = train_dataset.unbatch().take(train_sample_size).batch(batch_size)

bnn_model_small = create_bnn_model(train_sample_size)
run_experiment(bnn_model_small, mse_loss, small_train_dataset, test_dataset)
# Start training the model...
# Epoch 1/500
# 5/5 [==============================] - 2s 123ms/step - loss: 34.5497 - root_mean_squared_error: 5.8764 - val_loss: 37.1164 - val_root_mean_squared_error: 6.0910
# Epoch 2/500
# 5/5 [==============================] - 0s 28ms/step - loss: 36.0738 - root_mean_squared_error: 6.0007 - val_loss: 31.7373 - val_root_mean_squared_error: 5.6322
# Epoch 3/500
# 5/5 [==============================] - 0s 29ms/step - loss: 33.3177 - root_mean_squared_error: 5.7700 - val_loss: 36.2135 - val_root_mean_squared_error: 6.0164
# Epoch 4/500
# 5/5 [==============================] - 0s 30ms/step - loss: 35.1247 - root_mean_squared_error: 5.9232 - val_loss: 35.6158 - val_root_mean_squared_error: 5.9663
# Epoch 5/500
# 5/5 [==============================] - 0s 23ms/step - loss: 34.7653 - root_mean_squared_error: 5.8936 - val_loss: 34.3038 - val_root_mean_squared_error: 5.8556
#
# ...
#
# Epoch 495/500
# 5/5 [==============================] - 0s 24ms/step - loss: 0.6978 - root_mean_squared_error: 0.8162 - val_loss: 0.6258 - val_root_mean_squared_error: 0.7723
# Epoch 496/500
# 5/5 [==============================] - 0s 22ms/step - loss: 0.6448 - root_mean_squared_error: 0.7858 - val_loss: 0.6372 - val_root_mean_squared_error: 0.7808
# Epoch 497/500
# 5/5 [==============================] - 0s 23ms/step - loss: 0.6871 - root_mean_squared_error: 0.8121 - val_loss: 0.6437 - val_root_mean_squared_error: 0.7825
# Epoch 498/500
# 5/5 [==============================] - 0s 23ms/step - loss: 0.6213 - root_mean_squared_error: 0.7690 - val_loss: 0.6581 - val_root_mean_squared_error: 0.7922
# Epoch 499/500
# 5/5 [==============================] - 0s 22ms/step - loss: 0.6604 - root_mean_squared_error: 0.7913 - val_loss: 0.6522 - val_root_mean_squared_error: 0.7908
# Epoch 500/500
# 5/5 [==============================] - 0s 22ms/step - loss: 0.6190 - root_mean_squared_error: 0.7678 - val_loss: 0.6734 - val_root_mean_squared_error: 0.8037
# Model training finished.
# Train RMSE: 0.805
# Evaluating model performance...
# Test RMSE: 0.801
# 由于我们已经训练了一个 BNN 模型，因此每次我们使用相同的输入调用该模型时都会产生不同的输出，因为每次从分布中采样一组新的权重来构建网络并产生输出。模式权重越不确定，我们将在相同输入的输出中看到更多的可变性（范围更广）。

def compute_predictions(model, iterations=100):
    predicted = []
    for _ in range(iterations):
        predicted.append(model(examples).numpy())
    predicted = np.concatenate(predicted, axis=1)

    prediction_mean = np.mean(predicted, axis=1).tolist()
    prediction_min = np.min(predicted, axis=1).tolist()
    prediction_max = np.max(predicted, axis=1).tolist()
    prediction_range = (np.max(predicted, axis=1) - np.min(predicted, axis=1)).tolist()

    for idx in range(sample):
        print(
            f"Predictions mean: {round(prediction_mean[idx], 2)}, "
            f"min: {round(prediction_min[idx], 2)}, "
            f"max: {round(prediction_max[idx], 2)}, "
            f"range: {round(prediction_range[idx], 2)} - "
            f"Actual: {targets[idx]}"
        )


compute_predictions(bnn_model_small)
# Predictions mean: 5.63, min: 4.92, max: 6.15, range: 1.23 - Actual: 6.0
# Predictions mean: 6.35, min: 6.01, max: 6.54, range: 0.53 - Actual: 6.0
# Predictions mean: 5.65, min: 4.84, max: 6.25, range: 1.41 - Actual: 7.0
# Predictions mean: 5.74, min: 5.21, max: 6.25, range: 1.04 - Actual: 5.0
# Predictions mean: 5.99, min: 5.26, max: 6.29, range: 1.03 - Actual: 5.0
# Predictions mean: 6.26, min: 6.01, max: 6.47, range: 0.46 - Actual: 7.0
# Predictions mean: 5.28, min: 4.73, max: 5.86, range: 1.12 - Actual: 5.0
# Predictions mean: 6.34, min: 6.06, max: 6.53, range: 0.47 - Actual: 6.0
# Predictions mean: 6.23, min: 5.91, max: 6.44, range: 0.53 - Actual: 6.0
# Predictions mean: 6.33, min: 6.05, max: 6.54, range: 0.48 - Actual: 7.0


# 用整个训练集训练 BNN。
num_epochs = 500
bnn_model_full = create_bnn_model(train_size)
run_experiment(bnn_model_full, mse_loss, train_dataset, test_dataset)

compute_predictions(bnn_model_full)
# Start training the model...
# Epoch 1/500
# 17/17 [==============================] - 2s 32ms/step - loss: 25.4811 - root_mean_squared_error: 5.0465 - val_loss: 23.8428 - val_root_mean_squared_error: 4.8824
# Epoch 2/500
# 17/17 [==============================] - 0s 7ms/step - loss: 23.0849 - root_mean_squared_error: 4.8040 - val_loss: 24.1269 - val_root_mean_squared_error: 4.9115
# Epoch 3/500
# 17/17 [==============================] - 0s 7ms/step - loss: 22.5191 - root_mean_squared_error: 4.7449 - val_loss: 23.3312 - val_root_mean_squared_error: 4.8297
# Epoch 4/500
# 17/17 [==============================] - 0s 7ms/step - loss: 22.9571 - root_mean_squared_error: 4.7896 - val_loss: 24.4072 - val_root_mean_squared_error: 4.9399
# Epoch 5/500
# 17/17 [==============================] - 0s 6ms/step - loss: 21.4049 - root_mean_squared_error: 4.6245 - val_loss: 21.1895 - val_root_mean_squared_error: 4.6027
#
# ...
#
# Epoch 495/500
# 17/17 [==============================] - 0s 7ms/step - loss: 0.5799 - root_mean_squared_error: 0.7511 - val_loss: 0.5902 - val_root_mean_squared_error: 0.7572
# Epoch 496/500
# 17/17 [==============================] - 0s 6ms/step - loss: 0.5926 - root_mean_squared_error: 0.7603 - val_loss: 0.5961 - val_root_mean_squared_error: 0.7616
# Epoch 497/500
# 17/17 [==============================] - 0s 7ms/step - loss: 0.5928 - root_mean_squared_error: 0.7595 - val_loss: 0.5916 - val_root_mean_squared_error: 0.7595
# Epoch 498/500
# 17/17 [==============================] - 0s 7ms/step - loss: 0.6115 - root_mean_squared_error: 0.7715 - val_loss: 0.5869 - val_root_mean_squared_error: 0.7558
# Epoch 499/500
# 17/17 [==============================] - 0s 7ms/step - loss: 0.6044 - root_mean_squared_error: 0.7673 - val_loss: 0.6007 - val_root_mean_squared_error: 0.7645
# Epoch 500/500
# 17/17 [==============================] - 0s 7ms/step - loss: 0.5853 - root_mean_squared_error: 0.7550 - val_loss: 0.5999 - val_root_mean_squared_error: 0.7651
# Model training finished.
# Train RMSE: 0.762
# Evaluating model performance...
# Test RMSE: 0.759
# Predictions mean: 5.41, min: 5.06, max: 5.9, range: 0.84 - Actual: 6.0
# Predictions mean: 6.5, min: 6.16, max: 6.61, range: 0.44 - Actual: 6.0
# Predictions mean: 5.59, min: 4.96, max: 6.0, range: 1.04 - Actual: 7.0
# Predictions mean: 5.67, min: 5.25, max: 6.01, range: 0.76 - Actual: 5.0
# Predictions mean: 6.02, min: 5.68, max: 6.39, range: 0.71 - Actual: 5.0
# Predictions mean: 6.35, min: 6.11, max: 6.52, range: 0.41 - Actual: 7.0
# Predictions mean: 5.21, min: 4.85, max: 5.68, range: 0.83 - Actual: 5.0
# Predictions mean: 6.53, min: 6.35, max: 6.64, range: 0.28 - Actual: 6.0
# Predictions mean: 6.3, min: 6.05, max: 6.47, range: 0.42 - Actual: 6.0
# Predictions mean: 6.44, min: 6.19, max: 6.59, range: 0.4 - Actual: 7.0
# 请注意，与使用训练数据集的子集训练的模型相比，使用完整训练数据集训练的模型在相同输入的预测值中显示出更小的范围（不确定性）。

# 实验三：概率贝叶斯神经网络
# 到目前为止，我们构建的标准和贝叶斯 NN 模型的输出是确定性的，即产生一个点估计作为给定示例的预测。我们可以通过让模型输出分布来创建概率 NN。在这种情况下，模型也捕获了偶然的不确定性，这是由于数据中不可减少的噪声，或者是由于生成数据的过程的随机性。
#
# 在此示例中，我们将输出建模为IndependentNormal具有可学习的均值和方差参数的分布。如果任务是分类，我们将使用IndependentBernoulli二元类和OneHotCategorical 多个类来模拟模型输出的分布。

def create_probablistic_bnn_model(train_size):
    inputs = create_model_inputs()
    features = keras.layers.concatenate(list(inputs.values()))
    features = layers.BatchNormalization()(features)

    # Create hidden layers with weight uncertainty using the DenseVariational layer.
    for units in hidden_units:
        features = tfp.layers.DenseVariational(
            units=units,
            make_prior_fn=prior,
            make_posterior_fn=posterior,
            kl_weight=1 / train_size,
            activation="sigmoid",
        )(features)

    # Create a probabilisticå output (Normal distribution), and use the `Dense` layer
    # to produce the parameters of the distribution.
    # We set units=2 to learn both the mean and the variance of the Normal distribution.
    distribution_params = layers.Dense(units=2)(features)
    outputs = tfp.layers.IndependentNormal(1)(distribution_params)

    model = keras.Model(inputs=inputs, outputs=outputs)
    return model


# 由于模型的输出是一个分布，而不是点估计，我们使用负对数似然 作为我们的损失函数来计算从模型产生的估计分布中看到真实数据（目标）的可能性。

def negative_loglikelihood(targets, estimated_distribution):
    return -estimated_distribution.log_prob(targets)


num_epochs = 1000
prob_bnn_model = create_probablistic_bnn_model(train_size)
run_experiment(prob_bnn_model, negative_loglikelihood, train_dataset, test_dataset)
# Start training the model...
# Epoch 1/1000
# 17/17 [==============================] - 2s 36ms/step - loss: 11.2378 - root_mean_squared_error: 6.6758 - val_loss: 8.5554 - val_root_mean_squared_error: 6.6240
# Epoch 2/1000
# 17/17 [==============================] - 0s 7ms/step - loss: 11.8285 - root_mean_squared_error: 6.5718 - val_loss: 8.2138 - val_root_mean_squared_error: 6.5256
# Epoch 3/1000
# 17/17 [==============================] - 0s 7ms/step - loss: 8.8566 - root_mean_squared_error: 6.5369 - val_loss: 5.8749 - val_root_mean_squared_error: 6.3394
# Epoch 4/1000
# 17/17 [==============================] - 0s 7ms/step - loss: 7.8191 - root_mean_squared_error: 6.3981 - val_loss: 7.6224 - val_root_mean_squared_error: 6.4473
# Epoch 5/1000
# 17/17 [==============================] - 0s 7ms/step - loss: 6.2598 - root_mean_squared_error: 6.4613 - val_loss: 5.9415 - val_root_mean_squared_error: 6.3466
#
# ...
#
# Epoch 995/1000
# 17/17 [==============================] - 0s 7ms/step - loss: 1.1323 - root_mean_squared_error: 1.0431 - val_loss: 1.1553 - val_root_mean_squared_error: 1.1060
# Epoch 996/1000
# 17/17 [==============================] - 0s 7ms/step - loss: 1.1613 - root_mean_squared_error: 1.0686 - val_loss: 1.1554 - val_root_mean_squared_error: 1.0370
# Epoch 997/1000
# 17/17 [==============================] - 0s 7ms/step - loss: 1.1351 - root_mean_squared_error: 1.0628 - val_loss: 1.1472 - val_root_mean_squared_error: 1.0813
# Epoch 998/1000
# 17/17 [==============================] - 0s 7ms/step - loss: 1.1324 - root_mean_squared_error: 1.0858 - val_loss: 1.1527 - val_root_mean_squared_error: 1.0578
# Epoch 999/1000
# 17/17 [==============================] - 0s 7ms/step - loss: 1.1591 - root_mean_squared_error: 1.0801 - val_loss: 1.1483 - val_root_mean_squared_error: 1.0442
# Epoch 1000/1000
# 17/17 [==============================] - 0s 7ms/step - loss: 1.1402 - root_mean_squared_error: 1.0554 - val_loss: 1.1495 - val_root_mean_squared_error: 1.0389
# Model training finished.
# Train RMSE: 1.068
# Evaluating model performance...
# Test RMSE: 1.068
# 现在让我们从给定测试示例的模型中生成输出。输出现在是一个分布，我们可以使用它的均值和方差来计算预测的置信区间 (CI)。

prediction_distribution = prob_bnn_model(examples)
prediction_mean = prediction_distribution.mean().numpy().tolist()
prediction_stdv = prediction_distribution.stddev().numpy()

# The 95% CI is computed as mean ± (1.96 * stdv)
upper = (prediction_mean + (1.96 * prediction_stdv)).tolist()
lower = (prediction_mean - (1.96 * prediction_stdv)).tolist()
prediction_stdv = prediction_stdv.tolist()

for idx in range(sample):
    print(
        f"Prediction mean: {round(prediction_mean[idx][0], 2)}, "
        f"stddev: {round(prediction_stdv[idx][0], 2)}, "
        f"95% CI: [{round(upper[idx][0], 2)} - {round(lower[idx][0], 2)}]"
        f" - Actual: {targets[idx]}"
    )


def main():
    pass


if __name__ == '__main__':
    main()