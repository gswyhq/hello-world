import tensorflow as tf
import tensorflow_datasets
from transformers import *
import numpy as np
from transformers.modeling_bert import BertForSequenceClassification
from transformers.data.processors.glue import MrpcProcessor, InputExample
from tensorflow_datasets.text.glue import *

# Load dataset, tokenizer, model from pretrained model/vocabulary

# 下面两步会下载对应的预训练模型，模型路径在：/root/.cache/torch/transformers，并且对应名字会改写
tokenizer = BertTokenizer.from_pretrained('bert-base-cased')
model = TFBertForSequenceClassification.from_pretrained('bert-base-cased')

# 在线下载太慢，或者本地加载失败；
# data = tensorflow_datasets.load('glue/mrpc', data_dir='/root/glue/mrpc', download=False)

# MRPC_TRAIN = 'https://dl.fbaipublicfiles.com/senteval/senteval_data/msr_paraphrase_train.txt' > train.tsv
# MRPC_TEST = 'https://dl.fbaipublicfiles.com/senteval/senteval_data/msr_paraphrase_test.txt' > dev.tsv
# https://firebasestorage.googleapis.com/v0/b/mtl-sentence-representations.appspot.com/o/data%2Fmrpc_dev_ids.tsv?alt=media&token=ec5c0836-31d5-48f4-b431-7480817f1adc

# root@9c5b943d6f4d:~/glue/mrpc# ls |xargs -i md5sum {}
# e437fdddb92535b820fe8852e2df8a49  dev.tsv
# 7ab59a1b04bd7cb773f98a0717106c9b  mrpc_dev_ids.tsv
# e437fdddb92535b820fe8852e2df8a49  msr_paraphrase_test.txt
# 793daf7b6224281e75fe61c1f80afe35  msr_paraphrase_train.txt
# 793daf7b6224281e75fe61c1f80afe35  train.tsv

# 训练、校验文件路径；
mrpc_files = {
          "dev_ids": '/root/glue/mrpc/mrpc_dev_ids.tsv',
          "train": '/root/glue/mrpc/train.tsv',
          "test": '/root/glue/mrpc/dev/tsv',
      }

def _generate_example_mrpc_files(mrpc_files, split):
    if split == "test":
        with tf.io.gfile.GFile(mrpc_files["test"]) as f:
            reader = csv.DictReader(f, delimiter="\t", quoting=csv.QUOTE_NONE)
            for n, row in enumerate(reader):
                yield {
                    "sentence1": row["#1 String"],
                    "sentence2": row["#2 String"],
                    "label": -1,
                    "idx": n,
                }
    else:
        with tf.io.gfile.GFile(mrpc_files["dev_ids"]) as f:
            reader = csv.reader(f, delimiter="\t", quoting=csv.QUOTE_NONE)
            dev_ids = [[row[0], row[1]] for row in reader]
        with tf.io.gfile.GFile(mrpc_files["train"]) as f:
            # The first 3 bytes are the utf-8 BOM \xef\xbb\xbf, which messes with
            # the Quality key.
            f.seek(3)
            reader = csv.DictReader(f, delimiter="\t", quoting=csv.QUOTE_NONE)
            for n, row in enumerate(reader):
                is_row_in_dev = [row["#1 ID"], row["#2 ID"]] in dev_ids
                if is_row_in_dev == (split == "dev"):
                    yield {
                        "sentence1": row["#1 String"],
                        "sentence2": row["#2 String"],
                        "label": int(row["Quality"]),
                        "idx": n,
                    }

def _generate_example_train():
    return _generate_example_mrpc_files(
          mrpc_files=mrpc_files, split='train')
def _generate_example_dev():
    return _generate_example_mrpc_files(
          mrpc_files=mrpc_files, split='dev')

def load_mrpc_data():
    data = {}
    data['train'] = tf.data.Dataset.from_generator(_generate_example_train,
                                                   output_types={"sentence1": tf.string,
                                                                    "sentence2": tf.string,
                                                                    "label": tf.int8,
                                                                    "idx": tf.int16}
                                                   )
    data['validation'] = tf.data.Dataset.from_generator(_generate_example_dev,
                                                        output_types={"sentence1": tf.string,
                                                                        "sentence2": tf.string,
                                                                        "label": tf.int8,
                                                                        "idx": tf.int16}
                                                        )
    return data



data = load_mrpc_data()
# tf.data.Dataset.

# Prepare dataset for GLUE as a tf.data.Dataset instance
train_dataset = glue_convert_examples_to_features(data['train'], tokenizer, max_length=128, task='mrpc')
valid_dataset = glue_convert_examples_to_features(data['validation'], tokenizer, max_length=128, task='mrpc')
train_dataset = train_dataset.shuffle(100).batch(32).repeat(2)
valid_dataset = valid_dataset.batch(64)

# Prepare training: Compile tf.keras model with optimizer, loss and learning rate schedule
optimizer = tf.keras.optimizers.Adam(learning_rate=3e-5, epsilon=1e-08, clipnorm=1.0)
loss = tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True)
metric = tf.keras.metrics.SparseCategoricalAccuracy('accuracy')
model.compile(optimizer=optimizer, loss=loss, metrics=[metric])

# Train and evaluate using tf.keras.Model.fit()
history = model.fit(train_dataset, epochs=2, steps_per_epoch=115,
                            validation_data=valid_dataset, validation_steps=7)

# Load the TensorFlow model in PyTorch for inspection
model.save_pretrained('./save/')

pytorch_model = BertForSequenceClassification.from_pretrained('./save/', from_tf=True)

# PyTorch 加载 TensorFlow model 来用于预测；
import os
os.environ['USE_TORCH'] = 'YES'
os.environ['USE_TF'] = 'NO'

tokenizer = BertTokenizer.from_pretrained('bert-base-cased')

# Quickly test a few predictions - MRPC is a paraphrasing task, let's see if our model learned the task
sentence_0 = "This research was consistent with his findings."
sentence_1 = "His findings were compatible with this research."
sentence_2 = "His findings were not compatible with this research."
inputs_1 = tokenizer.encode_plus(sentence_0, sentence_1, add_special_tokens=True, return_tensors='pt')
inputs_2 = tokenizer.encode_plus(sentence_0, sentence_2, add_special_tokens=True, return_tensors='pt')
inputs_3 = tokenizer.encode_plus(sentence_1, sentence_2, add_special_tokens=True, return_tensors='pt')

pred_1 = pytorch_model(inputs_1['input_ids'], token_type_ids=inputs_1['token_type_ids'])[0].argmax().item()
pred_2 = pytorch_model(inputs_2['input_ids'], token_type_ids=inputs_2['token_type_ids'])[0].argmax().item()
pred_3 = pytorch_model(inputs_3['input_ids'], token_type_ids=inputs_3['token_type_ids'])[0].argmax().item()

print("sentence_1 is", "a paraphrase" if pred_1 else "not a paraphrase", "of sentence_0")
print("sentence_2 is", "a paraphrase" if pred_2 else "not a paraphrase", "of sentence_0")


# Epoch 1/2
# 115/115 [==============================] - 22280s 194s/step - loss: 0.5610 - accuracy: 0.7045 - val_loss: 0.3718 - val_accuracy: 0.8407
# Epoch 2/2
# 115/115 [==============================] - 22207s 193s/step - loss: 0.3126 - accuracy: 0.8716 - val_loss: 0.3329 - val_accuracy: 0.8627

# root@9c5b943d6f4d:~# head /etc/issue
# Debian GNU/Linux 9 \n \l
#
# root@9c5b943d6f4d:~# python3 --version
# Python 3.6.8

# Package                  Version
# ------------------------ -----------
# Keras-Preprocessing      1.1.2
# numpy                    1.18.4
# tensorboard              2.2.2
# tensorflow               2.2.0
# tensorflow-datasets      3.1.0
# torch                    1.5.0
# torchvision              0.6.0
# transformers             2.10.0

# 资料来源：
# https://github.com/huggingface/transformers

# 训练模型：
# root@9c5b943d6f4d:~# python3 transformers_mrpc.py

# 或者
# docker run --rm -it -w /root -v $PWD/save:/root/save gswyhq/transformers:mrpc python3 transformers_mrpc.py
