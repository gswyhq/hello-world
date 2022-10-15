#!/usr/bin/env python3
# -*- coding: utf-8 -*-

# 资料来源：https://keras.io/examples/nlp/multimodal_entailment/
# 在社交媒体平台上，为了审核和审核内容，我们可能希望近乎实时地找到以下问题的答案：
#
# 一条给定的信息是否与另一条相矛盾？
# 一条给定的信息是否暗示另一条信息？
# 在 NLP 中，此任务称为分析文本蕴涵。但是，仅当信息来自文本内容时。在实践中，可用的信息通常不仅来自文本内容，还来自文本、图像、音频、视频等的多模态组合。 多模态蕴涵只是文本蕴涵对各种新输入模态的扩展。

from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import os

import tensorflow as tf
import tensorflow_hub as hub
import tensorflow_text as text
from tensorflow import keras


from pylab import mpl
mpl.rcParams['font.sans-serif'] = ['SimHei'] #指定默认字体   
mpl.rcParams['axes.unicode_minus'] = False #解决保存图像是负号'-'显示为方块的问题

# 定义标签映射
label_map = {"Contradictory": 0, "Implies": 1, "NoEntailment": 2}

# 收集数据集
# 原始数据集可 在此处（https://github.com/google-research-datasets/recognizing-multimodal-entailment）获得。它带有图片的 URL，这些图片托管在 Twitter 的照片存储系统上，称为 Photo Blob Storage（简称 PBS）。我们将使用下载的图像以及原始数据集附带的其他数据。感谢 负责准备图像数据的Nilabhra Roy Chowdhury 。

image_base_path = keras.utils.get_file(
    "tweet_images",
    "https://github.com/sayakpaul/Multimodal-Entailment-Baseline/releases/download/v1.0.0/tweet_images.tar.gz",
    untar=True,
)

# 读取数据集并应用基本预处理
df = pd.read_csv(
    "https://github.com/sayakpaul/Multimodal-Entailment-Baseline/raw/main/csvs/tweets.csv"
)
df.sample(10)


# 我们感兴趣的列如下：
# text_1
# image_1
# text_2
# image_2
# label

# 蕴含任务表述如下：
# 给定 ( text_1, image_1) 和 ( text_2, image_2) 对，它们是否相互蕴含（或不蕴含或矛盾）？
#
# 我们已经下载了图像。image_1下载id1为其文件名，并以其文件名image2下载id2。在下一步中，我们将再添加两列df- image_1s 和image_2s 的文件路径。

images_one_paths = []
images_two_paths = []

for idx in range(len(df)):
    current_row = df.iloc[idx]
    id_1 = current_row["id_1"]
    id_2 = current_row["id_2"]
    extentsion_one = current_row["image_1"].split(".")[-1]
    extentsion_two = current_row["image_2"].split(".")[-1]

    image_one_path = os.path.join(image_base_path, str(id_1) + f".{extentsion_one}")
    image_two_path = os.path.join(image_base_path, str(id_2) + f".{extentsion_two}")

    images_one_paths.append(image_one_path)
    images_two_paths.append(image_two_path)

df["image_1_path"] = images_one_paths
df["image_2_path"] = images_two_paths

# Create another column containing the integer ids of
# the string labels.
df["label_idx"] = df["label"].apply(lambda x: label_map[x])


# 数据集可视化
def visualize(idx):
    current_row = df.iloc[idx]
    image_1 = plt.imread(current_row["image_1_path"])
    image_2 = plt.imread(current_row["image_2_path"])
    text_1 = current_row["text_1"]
    text_2 = current_row["text_2"]
    label = current_row["label"]

    plt.subplot(1, 2, 1)
    plt.imshow(image_1)
    plt.axis("off")
    plt.title("Image One")
    plt.subplot(1, 2, 2)
    plt.imshow(image_1)
    plt.axis("off")
    plt.title("Image Two")
    plt.show()

    print(f"Text one: {text_1}")
    print(f"Text two: {text_2}")
    print(f"Label: {label}")


random_idx = np.random.choice(len(df))
visualize(random_idx)

random_idx = np.random.choice(len(df))
visualize(random_idx)

# 训练/测试拆分
# 数据集存在 类不平衡问题。我们可以在下面的单元格中确认这一点。
#
# df["label"].value_counts()
# NoEntailment     1182
# Implies           109
# Contradictory     109
# Name: label, dtype: int64
# 为了解决这个问题，我们将进行分层拆分。

# 10% for test
train_df, test_df = train_test_split(
    df, test_size=0.1, stratify=df["label"].values, random_state=42
)
# 5% for validation
train_df, val_df = train_test_split(
    train_df, test_size=0.05, stratify=train_df["label"].values, random_state=42
)

print(f"Total training examples: {len(train_df)}")
print(f"Total validation examples: {len(val_df)}")
print(f"Total test examples: {len(test_df)}")


# 数据输入管道
# TensorFlow Hub 提供 各种 BERT 系列模型。这些模型中的每一个都带有相应的预处理层。
# 您可以从此资源（https://www.tensorflow.org/text/tutorials/bert_glue#loading_models_from_tensorflow_hub）了解有关这些模型及其预处理层的更多信息 。

# 为了保持这个示例的运行时间相对较短，我们将使用原始 BERT 模型的较小变体。

# Define TF Hub paths to the BERT encoder and its preprocessor
bert_model_path = (
    "https://tfhub.dev/tensorflow/small_bert/bert_en_uncased_L-2_H-256_A-4/1"
)
bert_preprocess_path = "https://tfhub.dev/tensorflow/bert_en_uncased_preprocess/3"

# 我们的文本预处理代码主要来自（https://www.tensorflow.org/text/tutorials/bert_glue）。强烈建议您查看教程以了解有关输入预处理的更多信息。

def make_bert_preprocessing_model(sentence_features, seq_length=128):
    """Returns Model mapping string features to BERT inputs.

  Args:
    sentence_features: A list with the names of string-valued features.
    seq_length: An integer that defines the sequence length of BERT inputs.

  Returns:
    A Keras Model that can be called on a list or dict of string Tensors
    (with the order or names, resp., given by sentence_features) and
    returns a dict of tensors for input to BERT.
  """

    input_segments = [
        tf.keras.layers.Input(shape=(), dtype=tf.string, name=ft)
        for ft in sentence_features
    ]

    # Tokenize the text to word pieces.
    bert_preprocess = hub.load(bert_preprocess_path)
    tokenizer = hub.KerasLayer(bert_preprocess.tokenize, name="tokenizer")
    segments = [tokenizer(s) for s in input_segments]

    # Optional: Trim segments in a smart way to fit seq_length.
    # Simple cases (like this example) can skip this step and let
    # the next step apply a default truncation to approximately equal lengths.
    truncated_segments = segments

    # Pack inputs. The details (start/end token ids, dict of output tensors)
    # are model-dependent, so this gets loaded from the SavedModel.
    packer = hub.KerasLayer(
        bert_preprocess.bert_pack_inputs,
        arguments=dict(seq_length=seq_length),
        name="packer",
    )
    model_inputs = packer(truncated_segments)
    return keras.Model(input_segments, model_inputs)


bert_preprocess_model = make_bert_preprocessing_model(["text_1", "text_2"])
keras.utils.plot_model(bert_preprocess_model, show_shapes=True, show_dtype=True)


# 在样本输入上运行预处理器
idx = np.random.choice(len(train_df))
row = train_df.iloc[idx]
sample_text_1, sample_text_2 = row["text_1"], row["text_2"]
print(f"Text 1: {sample_text_1}")
print(f"Text 2: {sample_text_2}")

test_text = [np.array([sample_text_1]), np.array([sample_text_2])]
text_preprocessed = bert_preprocess_model(test_text)

print("Keys           : ", list(text_preprocessed.keys()))
print("Shape Word Ids : ", text_preprocessed["input_word_ids"].shape)
print("Word Ids       : ", text_preprocessed["input_word_ids"][0, :16])
print("Shape Mask     : ", text_preprocessed["input_mask"].shape)
print("Input Mask     : ", text_preprocessed["input_mask"][0, :16])
print("Shape Type Ids : ", text_preprocessed["input_type_ids"].shape)
print("Type Ids       : ", text_preprocessed["input_type_ids"][0, :16])

# 我们现在tf.data.Dataset将从数据框创建对象。
#
# 请注意，文本输入将作为数据输入管道的一部分进行预处理。但预处理模块也可以是其相应 BERT 模型的一部分。这有助于减少训练/服务偏差，并让我们的模型使用原始文本输入进行操作。按照本教程 了解有关如何将预处理模块直接合并到模型中的更多信息。

def dataframe_to_dataset(dataframe):
    columns = ["image_1_path", "image_2_path", "text_1", "text_2", "label_idx"]
    dataframe = dataframe[columns].copy()
    labels = dataframe.pop("label_idx")
    ds = tf.data.Dataset.from_tensor_slices((dict(dataframe), labels))
    ds = ds.shuffle(buffer_size=len(dataframe))
    return ds

# 预处理实用程序
resize = (128, 128)
bert_input_features = ["input_word_ids", "input_type_ids", "input_mask"]


def preprocess_image(image_path):
    extension = tf.strings.split(image_path)[-1]

    image = tf.io.read_file(image_path)
    if extension == b"jpg":
        image = tf.image.decode_jpeg(image, 3)
    else:
        image = tf.image.decode_png(image, 3)
    image = tf.image.resize(image, resize)
    return image


def preprocess_text(text_1, text_2):
    text_1 = tf.convert_to_tensor([text_1])
    text_2 = tf.convert_to_tensor([text_2])
    output = bert_preprocess_model([text_1, text_2])
    output = {feature: tf.squeeze(output[feature]) for feature in bert_input_features}
    return output


def preprocess_text_and_image(sample):
    image_1 = preprocess_image(sample["image_1_path"])
    image_2 = preprocess_image(sample["image_2_path"])
    text = preprocess_text(sample["text_1"], sample["text_2"])
    return {"image_1": image_1, "image_2": image_2, "text": text}

# 创建最终数据集
batch_size = 32
auto = tf.data.AUTOTUNE


def prepare_dataset(dataframe, training=True):
    ds = dataframe_to_dataset(dataframe)
    if training:
        ds = ds.shuffle(len(train_df))
    ds = ds.map(lambda x, y: (preprocess_text_and_image(x), y)).cache()
    ds = ds.batch(batch_size).prefetch(auto)
    return ds


train_ds = prepare_dataset(train_df)
validation_ds = prepare_dataset(val_df, False)
test_ds = prepare_dataset(test_df, False)


# 模型构建实用程序
# 我们的最终模型将接受两个图像及其文本副本。虽然图像将直接馈送到模型中，但文本输入将首先进行预处理，然后将其放入模型中。
# 该模型由以下元素组成：
# 图像的独立编码器。为此，我们将使用 在ImageNet -1k 数据集上预训练的 ResNet50V2。
# 文本的独立编码器。为此，将使用预训练的 BERT。
# 提取单个嵌入后，它们将被投影到相同的空间中。最后，它们的投影将被连接起来并馈送到最终的分类层。

# 这是一个涉及以下类的多类分类问题：
# 无条件
# 暗示
# 矛盾的
# project_embeddings(), create_vision_encoder(), 和实用程序是从这个例子create_text_encoder()中引用的。

# 对象向量化
def project_embeddings(
    embeddings, num_projection_layers, projection_dims, dropout_rate
):
    projected_embeddings = keras.layers.Dense(units=projection_dims)(embeddings)
    for _ in range(num_projection_layers):
        x = tf.nn.gelu(projected_embeddings)
        x = keras.layers.Dense(projection_dims)(x)
        x = keras.layers.Dropout(dropout_rate)(x)
        x = keras.layers.Add()([projected_embeddings, x])
        projected_embeddings = keras.layers.LayerNormalization()(x)
    return projected_embeddings

# 图片编码
def create_vision_encoder(
    num_projection_layers, projection_dims, dropout_rate, trainable=False
):
    # Load the pre-trained ResNet50V2 model to be used as the base encoder.
    resnet_v2 = keras.applications.ResNet50V2(
        include_top=False, weights="imagenet", pooling="avg"
    )
    # Set the trainability of the base encoder.
    for layer in resnet_v2.layers:
        layer.trainable = trainable

    # Receive the images as inputs.
    image_1 = keras.Input(shape=(128, 128, 3), name="image_1")
    image_2 = keras.Input(shape=(128, 128, 3), name="image_2")

    # Preprocess the input image.
    preprocessed_1 = keras.applications.resnet_v2.preprocess_input(image_1)
    preprocessed_2 = keras.applications.resnet_v2.preprocess_input(image_2)

    # Generate the embeddings for the images using the resnet_v2 model
    # concatenate them.
    embeddings_1 = resnet_v2(preprocessed_1)
    embeddings_2 = resnet_v2(preprocessed_2)
    embeddings = keras.layers.Concatenate()([embeddings_1, embeddings_2])

    # Project the embeddings produced by the model.
    outputs = project_embeddings(
        embeddings, num_projection_layers, projection_dims, dropout_rate
    )
    # Create the vision encoder model.
    return keras.Model([image_1, image_2], outputs, name="vision_encoder")

# 文本编码
def create_text_encoder(
    num_projection_layers, projection_dims, dropout_rate, trainable=False
):
    # Load the pre-trained BERT model to be used as the base encoder.
    bert = hub.KerasLayer(bert_model_path, name="bert",)
    # Set the trainability of the base encoder.
    bert.trainable = trainable

    # Receive the text as inputs.
    bert_input_features = ["input_type_ids", "input_mask", "input_word_ids"]
    inputs = {
        feature: keras.Input(shape=(128,), dtype=tf.int32, name=feature)
        for feature in bert_input_features
    }

    # Generate embeddings for the preprocessed text using the BERT model.
    embeddings = bert(inputs)["pooled_output"]

    # Project the embeddings produced by the model.
    outputs = project_embeddings(
        embeddings, num_projection_layers, projection_dims, dropout_rate
    )
    # Create the text encoder model.
    return keras.Model(inputs, outputs, name="text_encoder")

# 多模型融合
def create_multimodal_model(
    num_projection_layers=1,
    projection_dims=256,
    dropout_rate=0.1,
    vision_trainable=False,
    text_trainable=False,
):
    # Receive the images as inputs.
    image_1 = keras.Input(shape=(128, 128, 3), name="image_1")
    image_2 = keras.Input(shape=(128, 128, 3), name="image_2")

    # Receive the text as inputs.
    bert_input_features = ["input_type_ids", "input_mask", "input_word_ids"]
    text_inputs = {
        feature: keras.Input(shape=(128,), dtype=tf.int32, name=feature)
        for feature in bert_input_features
    }

    # Create the encoders.
    vision_encoder = create_vision_encoder(
        num_projection_layers, projection_dims, dropout_rate, vision_trainable
    )
    text_encoder = create_text_encoder(
        num_projection_layers, projection_dims, dropout_rate, text_trainable
    )

    # Fetch the embedding projections.
    vision_projections = vision_encoder([image_1, image_2])
    text_projections = text_encoder(text_inputs)

    # Concatenate the projections and pass through the classification layer.
    concatenated = keras.layers.Concatenate()([vision_projections, text_projections])
    outputs = keras.layers.Dense(3, activation="softmax")(concatenated)
    return keras.Model([image_1, image_2, text_inputs], outputs)


multimodal_model = create_multimodal_model()
keras.utils.plot_model(multimodal_model, show_shapes=True)


# expand_nested您也可以通过设置 to 的参数plot_model()来检查各个编码器的结构 True。鼓励您使用构建此模型所涉及的不同超参数，并观察最终性能如何受到影响。

# 编译和训练模型
multimodal_model.compile(
    optimizer="adam", loss="sparse_categorical_crossentropy", metrics="accuracy"
)

history = multimodal_model.fit(train_ds, validation_data=validation_ds, epochs=10)
# Epoch 1/10
# 38/38 [==============================] - 49s 789ms/step - loss: 1.0014 - accuracy: 0.8229 - val_loss: 0.5514 - val_accuracy: 0.8571
# Epoch 2/10
# 38/38 [==============================] - 3s 90ms/step - loss: 0.4019 - accuracy: 0.8814 - val_loss: 0.5866 - val_accuracy: 0.8571
# Epoch 3/10
# 38/38 [==============================] - 3s 90ms/step - loss: 0.3557 - accuracy: 0.8897 - val_loss: 0.5929 - val_accuracy: 0.8571
# Epoch 4/10
# 38/38 [==============================] - 3s 91ms/step - loss: 0.2877 - accuracy: 0.9006 - val_loss: 0.6272 - val_accuracy: 0.8571
# Epoch 5/10
# 38/38 [==============================] - 3s 91ms/step - loss: 0.1796 - accuracy: 0.9398 - val_loss: 0.8545 - val_accuracy: 0.8254
# Epoch 6/10
# 38/38 [==============================] - 3s 91ms/step - loss: 0.1292 - accuracy: 0.9566 - val_loss: 1.2276 - val_accuracy: 0.8413
# Epoch 7/10
# 38/38 [==============================] - 3s 91ms/step - loss: 0.1015 - accuracy: 0.9666 - val_loss: 1.2914 - val_accuracy: 0.7778
# Epoch 8/10
# 38/38 [==============================] - 3s 92ms/step - loss: 0.1253 - accuracy: 0.9524 - val_loss: 1.1944 - val_accuracy: 0.8413
# Epoch 9/10
# 38/38 [==============================] - 3s 92ms/step - loss: 0.3064 - accuracy: 0.9131 - val_loss: 1.2162 - val_accuracy: 0.8095
# Epoch 10/10
# 38/38 [==============================] - 3s 92ms/step - loss: 0.2212 - accuracy: 0.9248 - val_loss: 1.1080 - val_accuracy: 0.8413

# 评估模型
_, acc = multimodal_model.evaluate(test_ds)
print(f"Accuracy on the test set: {round(acc * 100, 2)}%.")
# 5/5 [==============================] - 6s 1s/step - loss: 0.8390 - accuracy: 0.8429
# Accuracy on the test set: 84.29%.

# 关于训练的附加说明
# 结合正则化：
#
# 训练日志表明该模型开始过度拟合，并且可能受益于正则化。Dropout（Srivastava 等人）是一种简单而强大的正则化技术，我们可以在我们的模型中使用它。但是我们应该如何在这里应用它呢？
#
# 我们总是可以在模型的不同层之间引入 Dropout ( keras.layers.Dropout)。但这是另一个食谱。我们的模型需要来自两种不同数据模式的输入。如果在推理过程中不存在任何一种模式怎么办？为了解决这个问题，我们可以在单个投影被连接之前将 Dropout 引入：

vision_projections = keras.layers.Dropout(rate)(vision_projections)
text_projections = keras.layers.Dropout(rate)(text_projections)
concatenated = keras.layers.Concatenate()([vision_projections, text_projections])
# 关注重要事项：
#
# 图像的所有部分是否与它们的文本对应部分相同？情况可能并非如此。为了使我们的模型只关注与相应文本部分相关的图像中最重要的部分，我们可以使用“交叉注意力”：

# Embeddings.
vision_projections = vision_encoder([image_1, image_2])
text_projections = text_encoder(text_inputs)

# Cross-attention (Luong-style).
query_value_attention_seq = keras.layers.Attention(use_scale=True, dropout=0.2)(
    [vision_projections, text_projections]
)
# Concatenate.
concatenated = keras.layers.Concatenate()([vision_projections, text_projections])
contextual = keras.layers.Concatenate()([concatenated, query_value_attention_seq])


# 处理类不平衡：
#
# 数据集存在类不平衡。调查上述模型的混淆矩阵表明它在少数类上表现不佳。如果我们使用加权损失，那么训练会更有指导性。你可以查看 这个 在模型训练期间考虑到类不平衡的笔记本。
#
# 仅使用文本输入：
#
# 另外，如果我们只为蕴含任务合并文本输入怎么办？由于社交媒体平台上遇到的文本输入的性质，仅文本输入就会损害最终性能。在类似的训练设置下，通过仅使用文本输入，我们在相同的测试集上获得了 67.14% 的 top-1 准确率。有关详细信息，请参阅 此笔记本 。
#
# 最后，下表比较了为蕴含任务采取的不同方法：
#
# 类型	标准交叉熵	损失加权交叉熵	焦点损失
# 多式联运	77.86%	67.86%	86.43%
# 只有文字	67.14%	11.43%	37.86%


def main():
    pass


if __name__ == '__main__':
    main()