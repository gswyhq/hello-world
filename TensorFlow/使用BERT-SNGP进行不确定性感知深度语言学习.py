#!/usr/bin/env python
# coding: utf-8


# https://tensorflow.google.cn/text/tutorials/uncertainty_quantification_with_sngp_bert


get_ipython().system('pip uninstall -y tensorflow tf-text')


# In[3]:


get_ipython().system('pip install "tensorflow-text==2.8.*"')


# In[4]:


get_ipython().system('pip install -U tf-models-official==2.7.0')


# In[5]:


import matplotlib.pyplot as plt

import sklearn.metrics
import sklearn.calibration

import tensorflow_hub as hub
import tensorflow_datasets as tfds

import numpy as np
import tensorflow as tf

import official.nlp.modeling.layers as layers
import official.nlp.optimization as optimization

from pylab import mpl
mpl.rcParams['font.sans-serif'] = ['SimHei'] #指定默认字体   
mpl.rcParams['axes.unicode_minus'] = False #解决保存图像是负号'-'显示为方块的问题




#@title Standard BERT model

PREPROCESS_HANDLE = 'https://tfhub.dev/tensorflow/bert_en_uncased_preprocess/3'
MODEL_HANDLE = 'https://tfhub.dev/tensorflow/bert_en_uncased_L-12_H-768_A-12/3'

class BertClassifier(tf.keras.Model):
  def __init__(self, 
               num_classes=150, inner_dim=768, dropout_rate=0.1,
               **classifier_kwargs):
    
    super().__init__()
    self.classifier_kwargs = classifier_kwargs

    # Initiate the BERT encoder components.
    self.bert_preprocessor = hub.KerasLayer(PREPROCESS_HANDLE, name='preprocessing')
    self.bert_hidden_layer = hub.KerasLayer(MODEL_HANDLE, trainable=True, name='bert_encoder')

    # Defines the encoder and classification layers.
    self.bert_encoder = self.make_bert_encoder()
    self.classifier = self.make_classification_head(num_classes, inner_dim, dropout_rate)

  def make_bert_encoder(self):
    text_inputs = tf.keras.layers.Input(shape=(), dtype=tf.string, name='text')
    encoder_inputs = self.bert_preprocessor(text_inputs)
    encoder_outputs = self.bert_hidden_layer(encoder_inputs)
    return tf.keras.Model(text_inputs, encoder_outputs)

  def make_classification_head(self, num_classes, inner_dim, dropout_rate):
    return layers.ClassificationHead(
        num_classes=num_classes, 
        inner_dim=inner_dim,
        dropout_rate=dropout_rate,
        **self.classifier_kwargs)

  def call(self, inputs, **kwargs):
    encoder_outputs = self.bert_encoder(inputs)
    classifier_inputs = encoder_outputs['sequence_output']
    return self.classifier(classifier_inputs, **kwargs)


# ### Build SNGP model

# To implement a BERT-SNGP model, you only need to replace the `ClassificationHead` with the built-in [`GaussianProcessClassificationHead`](https://github.com/tensorflow/models/blob/master/official/nlp/modeling/layers/cls_head.py). Spectral normalization is already pre-packaged into this classification head. Like in the [SNGP tutorial](https://www.tensorflow.org/tutorials/uncertainty/sngp), add a covariance reset callback to the model, so the model automatically reset the covariance estimator at the begining of a new epoch to avoid counting the same data twice.

# In[10]:


class ResetCovarianceCallback(tf.keras.callbacks.Callback):

  def on_epoch_begin(self, epoch, logs=None):
    """Resets covariance matrix at the begining of the epoch."""
    if epoch > 0:
      self.model.classifier.reset_covariance_matrix()


# In[11]:


class SNGPBertClassifier(BertClassifier):

  def make_classification_head(self, num_classes, inner_dim, dropout_rate):
    return layers.GaussianProcessClassificationHead(
        num_classes=num_classes, 
        inner_dim=inner_dim,
        dropout_rate=dropout_rate,
        gp_cov_momentum=-1,
        temperature=30.,
        **self.classifier_kwargs)

  def fit(self, *args, **kwargs):
    """Adds ResetCovarianceCallback to model callbacks."""
    kwargs['callbacks'] = list(kwargs.get('callbacks', []))
    kwargs['callbacks'].append(ResetCovarianceCallback())

    return super().fit(*args, **kwargs)


# Note: The `GaussianProcessClassificationHead` takes a new argument `temperature`. It corresponds to the $\lambda$ parameter in the __mean-field approximation__ introduced in the [SNGP tutorial](https://www.tensorflow.org/tutorials/understanding/sngp). In practice, this value is usually treated as a hyperparameter, and is finetuned to optimize the model's calibration performance.

# ### Load CLINC OOS dataset

# Now load the [CLINC OOS](https://www.tensorflow.org/datasets/catalog/clinc_oos) intent detection dataset. This dataset contains 15000 user's spoken queries collected over 150 intent classes, it also contains 1000 out-of-domain (OOD) sentences that are not covered by any of the known classes.

# In[12]:


(clinc_train, clinc_test, clinc_test_oos), ds_info = tfds.load(
    'clinc_oos', split=['train', 'test', 'test_oos'], with_info=True, batch_size=-1)


# Make the train and test data.

# In[13]:


train_examples = clinc_train['text']
train_labels = clinc_train['intent']

# Makes the in-domain (IND) evaluation data.
ind_eval_data = (clinc_test['text'], clinc_test['intent'])


# Create a OOD evaluation dataset. For this, combine the in-domain test data `clinc_test` and the out-of-domain data `clinc_test_oos`. We will also assign label 0 to the in-domain examples, and label 1 to the out-of-domain examples. 

# In[14]:


test_data_size = ds_info.splits['test'].num_examples
oos_data_size = ds_info.splits['test_oos'].num_examples

# Combines the in-domain and out-of-domain test examples.
oos_texts = tf.concat([clinc_test['text'], clinc_test_oos['text']], axis=0)
oos_labels = tf.constant([0] * test_data_size + [1] * oos_data_size)

# Converts into a TF dataset.
ood_eval_dataset = tf.data.Dataset.from_tensor_slices(
    {"text": oos_texts, "label": oos_labels})


# ### Train and evaluate

# First set up the basic training configurations.

# In[15]:


TRAIN_EPOCHS = 3
TRAIN_BATCH_SIZE = 32
EVAL_BATCH_SIZE = 256


# In[16]:


#@title

def bert_optimizer(learning_rate, 
                   batch_size=TRAIN_BATCH_SIZE, epochs=TRAIN_EPOCHS, 
                   warmup_rate=0.1):
  """Creates an AdamWeightDecay optimizer with learning rate schedule."""
  train_data_size = ds_info.splits['train'].num_examples
  
  steps_per_epoch = int(train_data_size / batch_size)
  num_train_steps = steps_per_epoch * epochs
  num_warmup_steps = int(warmup_rate * num_train_steps)  

  # Creates learning schedule.
  lr_schedule = tf.keras.optimizers.schedules.PolynomialDecay(
      initial_learning_rate=learning_rate,
      decay_steps=num_train_steps,
      end_learning_rate=0.0)  
  
  return optimization.AdamWeightDecay(
      learning_rate=lr_schedule,
      weight_decay_rate=0.01,
      epsilon=1e-6,
      exclude_from_weight_decay=['LayerNorm', 'layer_norm', 'bias'])


# In[17]:


optimizer = bert_optimizer(learning_rate=1e-4)
loss = tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True)
metrics = tf.metrics.SparseCategoricalAccuracy()


# In[18]:


fit_configs = dict(batch_size=TRAIN_BATCH_SIZE,
                   epochs=TRAIN_EPOCHS,
                   validation_batch_size=EVAL_BATCH_SIZE, 
                   validation_data=ind_eval_data)


# In[19]:


sngp_model = SNGPBertClassifier()
sngp_model.compile(optimizer=optimizer, loss=loss, metrics=metrics)
sngp_model.fit(train_examples, train_labels, **fit_configs)


# ### Evaluate OOD performance

# Evaluate how well the model can detect the unfamiliar out-of-domain queries. For rigorous evaluation, use the OOD evaluation dataset `ood_eval_dataset` built earlier.

# In[20]:


#@title

def oos_predict(model, ood_eval_dataset, **model_kwargs):
  oos_labels = []
  oos_probs = []

  ood_eval_dataset = ood_eval_dataset.batch(EVAL_BATCH_SIZE)
  for oos_batch in ood_eval_dataset:
    oos_text_batch = oos_batch["text"]
    oos_label_batch = oos_batch["label"] 

    pred_logits = model(oos_text_batch, **model_kwargs)
    pred_probs_all = tf.nn.softmax(pred_logits, axis=-1)
    pred_probs = tf.reduce_max(pred_probs_all, axis=-1)

    oos_labels.append(oos_label_batch)
    oos_probs.append(pred_probs)

  oos_probs = tf.concat(oos_probs, axis=0)
  oos_labels = tf.concat(oos_labels, axis=0) 

  return oos_probs, oos_labels


# Computes the OOD probabilities as $1 - p(x)$, where $p(x)=softmax(logit(x))$ is the predictive probability.

# In[21]:


sngp_probs, ood_labels = oos_predict(sngp_model, ood_eval_dataset)


# In[22]:


ood_probs = 1 - sngp_probs


# Now evaluate how well the model's uncertainty score `ood_probs` predicts the out-of-domain label. First compute the Area under precision-recall curve (AUPRC) for OOD probability v.s. OOD detection accuracy.

# In[23]:


precision, recall, _ = sklearn.metrics.precision_recall_curve(ood_labels, ood_probs)


# In[24]:


auprc = sklearn.metrics.auc(recall, precision)
print(f'SNGP AUPRC: {auprc:.4f}')


# This matches the SNGP performance reported at the CLINC OOS benchmark under the [Uncertainty Baselines](https://github.com/google/uncertainty-baselines).

# Next, examine the model's quality in [uncertainty calibration](https://scikit-learn.org/stable/modules/calibration.html), i.e., whether the model's predictive probability corresponds to its predictive accuracy. A well-calibrated model is considered trust-worthy, since, for example, its predictive probability $p(x)=0.8$ means that the model is correct 80% of the time.

# In[25]:


prob_true, prob_pred = sklearn.calibration.calibration_curve(
    ood_labels, ood_probs, n_bins=10, strategy='quantile')


# In[26]:


plt.plot(prob_pred, prob_true)

plt.plot([0., 1.], [0., 1.], c='k', linestyle="--")
plt.xlabel('Predictive Probability')
plt.ylabel('Predictive Accuracy')
plt.title('Calibration Plots, SNGP')

plt.show()


# ## Resources and further reading

# * See the [SNGP tutorial](https://www.tensorflow.org/tutorials/understanding/sngp) for an detailed walkthrough of implementing SNGP from scratch. 
# * See [Uncertainty Baselines](https://github.com/google/uncertainty-baselines)  for the implementation of SNGP model (and many other uncertainty methods) on a wide variety of benchmark datasets (e.g., [CIFAR](https://www.tensorflow.org/datasets/catalog/cifar100), [ImageNet](https://www.tensorflow.org/datasets/catalog/imagenet2012), [Jigsaw toxicity detection](https://www.tensorflow.org/datasets/catalog/wikipedia_toxicity_subtypes), etc).
# * For a deeper understanding of the SNGP method, check out the paper [Simple and Principled Uncertainty Estimation with Deterministic Deep Learning via Distance Awareness](https://arxiv.org/abs/2006.10108).
# 
