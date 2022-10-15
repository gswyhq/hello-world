#!/usr/bin/env python3
# -*- coding: utf-8 -*-

# 来源：https://pytorch-forecasting.readthedocs.io/en/latest/tutorials/stallion.html
import os

import copy
from pathlib import Path
import warnings

import numpy as np
import pandas as pd
import pytorch_lightning as pl
from pytorch_lightning.callbacks import EarlyStopping, LearningRateMonitor
from pytorch_lightning.loggers import TensorBoardLogger
import torch

from pytorch_forecasting import Baseline, TemporalFusionTransformer, TimeSeriesDataSet
from pytorch_forecasting.data import GroupNormalizer
from pytorch_forecasting.metrics import SMAPE, PoissonLoss, QuantileLoss
from pytorch_forecasting.models.temporal_fusion_transformer.tuning import optimize_hyperparameters

USERNAME = os.getenv("USERNAME")

# 加载数据
# 首先，我们需要将时间序列转换为 pandas 数据帧，其中每一行都可以用时间步长和时间序列来标识。幸运的是，大多数数据集已经采用这种格式。在本教程中，我们将使用来自 Kaggle 的 Stallion 数据集来描述各种饮料的销售情况。我们的任务是按库存单位 (SKU) 对销售量进行六个月的预测，即由代理机构（即商店）销售的产品。每月约有 21 000 条历史销售记录。除了历史销售，我们还有销售价格、代理商所在地、节假日等特殊日子以及整个行业的销售量等信息。
#
# 数据集的格式已经正确，但缺少一些重要特征。最重要的是，我们需要添加一个时间索引，每个时间步增加一。此外，添加日期特征是有益的，在这种情况下，这意味着从日期记录中提取月份。

from pytorch_forecasting.data.examples import get_stallion_data

# data = get_stallion_data()
# 数据来源： https://github.com/jdb78/pytorch-forecasting/examples/data/stallion.parquet
# agency和sku是group id，用于节点多序列中的没有给序列，一个agency+sku的组合代表了一个单序列，volume是我们要预测的销量数据
data = pd.read_parquet(rf"D:\Users\{USERNAME}\github_project\pytorch-forecasting\examples\data\stallion.parquet")

# add time index
data["time_idx"] = data["date"].dt.year * 12 + data["date"].dt.month
data["time_idx"] -= data["time_idx"].min()

# add additional features
data["month"] = data.date.dt.month.astype(str).astype("category")  # categories have be strings
data["log_volume"] = np.log(data.volume + 1e-8)
data["avg_volume_by_sku"] = data.groupby(["time_idx", "sku"], observed=True).volume.transform("mean")
data["avg_volume_by_agency"] = data.groupby(["time_idx", "agency"], observed=True).volume.transform("mean")

# we want to encode special days as one variable and thus need to first reverse one-hot encoding
special_days = [
    "easter_day",
    "good_friday",
    "new_year",
    "christmas",
    "labor_day",
    "independence_day",
    "revolution_day_memorial",
    "regional_games",
    "fifa_u_17_world_cup",
    "football_gold_cup",
    "beer_capital",
    "music_fest",
]
data[special_days] = data[special_days].apply(lambda x: x.map({0: "-", 1: x.name})).astype("category")
data.sample(10, random_state=521)
# 机构	库存单位	体积	日期	工业量	苏打水量	avg_max_temp	price_regular	price_actual	折扣	...	足球金杯	啤酒资本	音乐节	折扣百分比	时间序列	time_idx	月	日志卷	avg_volume_by_sku	avg_volume_by_agency
# 291	机构_25	SKU_03	0.5076	2013-01-01	492612703	718394219	25.845238	1264.162234	1152.473405	111.688829	...	-	-	-	8.835008	228	0	1	-0.678062	1225.306376	99.650400
# 871	机构_29	SKU_02	8.7480	2015-01-01	498567142	762225057	27.584615	1316.098485	1296.804924	19.293561	...	-	-	-	1.465966	177	24	1	2.168825	1634.434615	11.397086
# 19532	机构_47	SKU_01	4.9680	2013-09-01	454252482	789624076	30.665957	1269.250000	1266.490490	2.759510	...	-	-	-	0.217413	322	8	9	1.603017	2625.472644	48.295650
# 2089	机构_53	SKU_07	21.6825	2013-10-01	480693900	791658684	29.197727	1193.842373	1128.124395	65.717978	...	-	啤酒资本	-	5.504745	240	9	10	3.076505	38.529107	2511.035175
# 9755	机构_17	SKU_02	960.5520	2015-03-01	515468092	871204688	23.608120	1338.334248	1232.128069	106.206179	...	-	-	音乐节	7.935699	259	26	3	6.867508	2143.677462	396.022140
# 7561	机构_05	SKU_03	1184.6535	2014-02-01	425528909	734443953	28.668254	1369.556376	1161.135214	208.421162	...	-	-	-	15.218151	21	13	2	7.077206	1566.643589	1881.866367
# 19204	机构_11	SKU_05	5.5593	2017-08-01	623319783	1049868815	31.915385	1922.486644	1651.307674	271.178970	...	-	-	-	14.105636	17	55	8	1.715472	1385.225478	109.699200
# 8781	机构_48	SKU_04	4275.1605	2013-03-01	509281531	892192092	26.767857	1761.258209	1546.059670	215.198539	...	-	-	音乐节	12.218455	151	2	3	8.360577	1757.950603	1925.272108
# 2540	机构_07	SKU_21	0.0000	2015-10-01	544203593	761469815	28.987755	0.000000	0.000000	0.000000	...	-	-	-	0.000000	300	33	10	-18.420681	0.000000	2418.719550
# 12084	机构_21	SKU_03	46.3608	2017-04-01	589969396	940912941	32.478910	1675.922116	1413.571789	262.350327	...	-	-	-	15.654088	181	51	4	3.836454	2034.293024	109.381800
# 10行×31列

data.describe()
# 体积	工业量	苏打水量	avg_max_temp	price_regular	price_actual	折扣	avg_population_2017	avg_yearly_household_income_2017	折扣百分比	时间序列	time_idx	日志卷	avg_volume_by_sku	avg_volume_by_agency
# 数数	21000.000000	2.100000e+04	2.100000e+04	21000.000000	21000.000000	21000.000000	21000.000000	2.100000e+04	21000.000000	21000.000000	21000.00000	21000.000000	21000.000000	21000.000000	21000.000000
# 意思是	1492.403982	5.439214e+08	8.512000e+08	28.612404	1451.536344	1267.347450	184.374146	1.045065e+06	151073.494286	10.574884	174.50000	29.500000	2.464118	1492.403982	1492.403982
# 性病	2711.496882	6.288022e+07	7.824340e+07	3.972833	683.362417	587.757323	257.469968	9.291926e+05	50409.593114	9.590813	101.03829	17.318515	8.178218	1051.790829	1328.239698
# 分钟	0.000000	4.130518e+08	6.964015e+08	16.731034	0.000000	-3121.690141	0.000000	1.227100e+04	90240.000000	0.000000	0.00000	0.000000	-18.420681	0.000000	0.000000
# 25%	8.272388	5.090553e+08	7.890880e+08	25.374816	1311.547158	1178.365653	54.935108	6.018900e+04	110057.000000	3.749628	87.00000	14.750000	2.112923	932.285496	113.420250
# 50%	158.436000	5.512000e+08	8.649196e+08	28.479272	1495.174592	1324.695705	138.307225	1.232242e+06	131411.000000	8.948990	174.50000	29.500000	5.065351	1402.305264	1730.529771
# 75%	1774.793475	5.893715e+08	9.005551e+08	31.568405	1725.652080	1517.311427	272.298630	1.729177e+06	206553.000000	15.647058	262.00000	44.250000	7.481439	2195.362302	2595.316500
# 最大限度	22526.610000	6.700157e+08	1.049869e+09	45.290476	19166.625000	4925.404000	19166.625000	3.137874e+06	247220.000000	226.740147	349.00000	59.000000	10.022453	4332.363750	5884.717375

# 创建数据集和数据加载器
# 下一步是将数据帧转换为 PyTorch Forecasting TimeSeriesDataSet。
# 除了告诉数据集哪些特征是分类的还是连续的，哪些是静态的还是随时间变化的，我们还必须决定如何规范化数据。
# 在这里，我们分别对每个时间序列进行标准缩放，并表明值始终为正。
# 一般来说，EncoderNormalizer，在您训练时在每个编码器序列上动态缩放，最好避免归一化引起的前瞻偏差。
# 但是，如果您难以找到合理稳定的归一化，您可能会接受前瞻偏差，例如，因为您的数据中有很多零。
# 或者您期望推理中的标准化更稳定。在后一种情况下，您确保不会学习在运行推理时不会出现的“奇怪”跳跃，从而在更真实的数据集上进行训练。
#
# 我们还选择使用过去六个月作为验证集。

max_prediction_length = 6
max_encoder_length = 24
training_cutoff = data["time_idx"].max() - max_prediction_length

training = TimeSeriesDataSet(
    data[lambda x: x.time_idx <= training_cutoff],
    time_idx="time_idx",
    target="volume",
    group_ids=["agency", "sku"],
    min_encoder_length=max_encoder_length // 2,  # keep encoder length long (as it is in the validation set)
    max_encoder_length=max_encoder_length,
    min_prediction_length=1,
    max_prediction_length=max_prediction_length,
    static_categoricals=["agency", "sku"],
    static_reals=["avg_population_2017", "avg_yearly_household_income_2017"],
    time_varying_known_categoricals=["special_days", "month"],
    variable_groups={"special_days": special_days},  # group of categorical variables can be treated as one variable
    time_varying_known_reals=["time_idx", "price_regular", "discount_in_percent"],
    time_varying_unknown_categoricals=[],
    time_varying_unknown_reals=[
        "volume",
        "log_volume",
        "industry_volume",
        "soda_volume",
        "avg_max_temp",
        "avg_volume_by_agency",
        "avg_volume_by_sku",
    ],
    target_normalizer=GroupNormalizer(
        groups=["agency", "sku"], transformation="softplus"
    ),  # use softplus and normalize by group
    add_relative_time_idx=True,
    add_target_scales=True,
    add_encoder_length=True,
)

# 由每个序列预测最后一个max_prediction_length，创建校验集(predict=True)
validation = TimeSeriesDataSet.from_dataset(training, data, predict=True, stop_randomization=True)

# create dataloaders for model
batch_size = 128  # set this between 32 to 128
train_dataloader = training.to_dataloader(train=True, batch_size=batch_size, num_workers=0)
val_dataloader = validation.to_dataloader(train=False, batch_size=batch_size * 10, num_workers=0)

# 创建基线模型
# 通过简单地重复上次观察到的交易量来评估Baseline预测未来 6 个月的模型，为我们提供了一个我们想要超越的简单基准。

# 计算基线平均绝对误差，即将下一个值预测为历史中的最后一个可用值
actuals = torch.cat([y for x, (y, weight) in iter(val_dataloader)])
baseline_predictions = Baseline().predict(val_dataloader)
(actuals - baseline_predictions).abs().mean().item()
# 293.0088195800781


# 训练时间融合Transformer
# 现在是时候创建我们的TemporalFusionTransformer模型了。我们使用 PyTorch Lightning 训练模型。
#
# 找到最佳学习率
# 在训练之前，您可以使用PyTorch Lightning 学习率查找器确定最佳学习率。

# 配置网络及训练
pl.seed_everything(42)
trainer = pl.Trainer(
    gpus=0,
    # clipping gradients is a hyperparameter and important to prevent divergance
    # of the gradient for recurrent neural networks
    gradient_clip_val=0.1,
)


tft = TemporalFusionTransformer.from_dataset(
    training,
    # not meaningful for finding the learning rate but otherwise very important
    learning_rate=0.03,
    hidden_size=16,  # most important hyperparameter apart from learning rate
    # number of attention heads. Set to up to 4 for large datasets
    attention_head_size=1,
    dropout=0.1,  # between 0.1 and 0.3 are good values
    hidden_continuous_size=8,  # set to <= hidden_size
    output_size=7,  # 7 quantiles by default
    loss=QuantileLoss(),
    # reduce learning rate if no improvement in validation loss after x epochs
    reduce_on_plateau_patience=4,
)
print(f"Number of parameters in network: {tft.size()/1e3:.1f}k")
# 全球种子设定为 42
# GPU 可用：False，已使用：False
# TPU 可用：False，使用：0 TPU 内核
# 网络中的参数数量：29.7k

# 展示模型结构：
tft.summarize()

#    | Name                               | Type                            | Params
# ----------------------------------------------------------------------------------------
# 0  | loss                               | QuantileLoss                    | 0
# 1  | logging_metrics                    | ModuleList                      | 0
# 2  | input_embeddings                   | MultiEmbedding                  | 1.3 K
# 3  | prescalers                         | ModuleDict                      | 256
# 4  | static_variable_selection          | VariableSelectionNetwork        | 3.4 K
# 5  | encoder_variable_selection         | VariableSelectionNetwork        | 8.0 K
# 6  | decoder_variable_selection         | VariableSelectionNetwork        | 2.7 K
# 7  | static_context_variable_selection  | GatedResidualNetwork            | 1.1 K
# 8  | static_context_initial_hidden_lstm | GatedResidualNetwork            | 1.1 K
# 9  | static_context_initial_cell_lstm   | GatedResidualNetwork            | 1.1 K
# 10 | static_context_enrichment          | GatedResidualNetwork            | 1.1 K
# 11 | lstm_encoder                       | LSTM                            | 2.2 K
# 12 | lstm_decoder                       | LSTM                            | 2.2 K
# 13 | post_lstm_gate_encoder             | GatedLinearUnit                 | 544
# 14 | post_lstm_add_norm_encoder         | AddNorm                         | 32
# 15 | static_enrichment                  | GatedResidualNetwork            | 1.4 K
# 16 | multihead_attn                     | InterpretableMultiHeadAttention | 1.1 K
# 17 | post_attn_gate_norm                | GateAddNorm                     | 576
# 18 | pos_wise_ff                        | GatedResidualNetwork            | 1.1 K
# 19 | pre_output_gate_norm               | GateAddNorm                     | 576
# 20 | output_layer                       | Linear                          | 119
# ----------------------------------------------------------------------------------------
# 29.7 K    Trainable params
# 0         Non-trainable params
# 29.7 K    Total params
# 0.119     Total estimated model params size (MB)

# 寻找最佳学习率
res = trainer.tuner.lr_find(
    tft,
    train_dataloaders=train_dataloader,
    val_dataloaders=val_dataloader,
    max_lr=10.0,
    min_lr=1e-6,
)

print(f"建议学习率(learning rate): {res.suggestion()}")
fig = res.plot(show=True, suggest=True)
fig.show()
# 建议学习率(learning rate): 0.01862087136662867


# 对于TemporalFusionTransformer，最佳学习率似乎略低于建议的学习率。
# 此外，我们不想直接使用建议的学习率，因为 PyTorch Lightning 有时会被较低学习率的噪音混淆，并且建议的学习率太低。
# 手动控制是必不可少的。我们决定选择 0.03 作为学习率。

# 训练模型
# 如果您在训练模型时遇到问题并出现错误，请考虑卸载 tensorflow 或先执行AttributeError: module 'tensorflow._api.v2.io.gfile' has no attribute 'get_filesystem'

# import tensorflow as tf
# import tensorboard as tb
# tf.io.gfile = tb.compat.tensorflow_stub.io.gfile

# configure network and trainer
early_stop_callback = EarlyStopping(monitor="val_loss", min_delta=1e-4, patience=10, verbose=False, mode="min")
lr_logger = LearningRateMonitor()  # log the learning rate
logger = TensorBoardLogger("lightning_logs")  # logging results to a tensorboard

trainer = pl.Trainer(
    max_epochs=30,
    gpus=0,
    enable_model_summary=True,
    gradient_clip_val=0.1,
    limit_train_batches=30,  # coment in for training, running valiation every 30 batches
    # fast_dev_run=True,  # comment in to check that networkor dataset has no serious bugs
    callbacks=[lr_logger, early_stop_callback],
    logger=logger,
)


tft = TemporalFusionTransformer.from_dataset(
    training,
    learning_rate=0.03,
    hidden_size=16,
    attention_head_size=1,
    dropout=0.1,
    hidden_continuous_size=8,
    output_size=7,  # 7 quantiles by default
    loss=QuantileLoss(),
    log_interval=10,  # uncomment for learning rate finder and otherwise, e.g. to 10 for logging every 10 batches
    reduce_on_plateau_patience=4,
)
print(f"Number of parameters in network: {tft.size()/1e3:.1f}k")
# GPU 可用：False，已使用：False
# TPU 可用：False，使用：0 TPU 内核
# 网络中的参数数量：29.7k


# 在我的 Macbook 上训练需要几分钟，但对于更大的网络和数据集，可能需要几个小时。
# 这里的训练速度主要由开销决定，选择更大的batch_size或hidden_size（即网络大小）不会减慢线性训练，从而使大型数据集的训练变得可行。
# 在训练期间，我们可以监控可以旋转的张量板。例如，我们可以监控训练和验证集上的示例预测。
# tensorboard --logdir=lightning_logs

# fit network
trainer.fit(
    tft,
    train_dataloaders=train_dataloader,
    val_dataloaders=val_dataloader,
)


# 超参数调优
import pickle

from pytorch_forecasting.models.temporal_fusion_transformer.tuning import optimize_hyperparameters

# create study
study = optimize_hyperparameters(
    train_dataloader,
    val_dataloader,
    model_path="optuna_test",
    n_trials=200,
    max_epochs=50,
    gradient_clip_val_range=(0.01, 1.0),
    hidden_size_range=(8, 128),
    hidden_continuous_size_range=(8, 128),
    attention_head_size_range=(1, 4),
    learning_rate_range=(0.001, 0.1),
    dropout_range=(0.1, 0.3),
    trainer_kwargs=dict(limit_train_batches=30),
    reduce_on_plateau_patience=4,
    use_learning_rate_finder=False,  # use Optuna to find ideal learning rate or use in-built learning rate finder
)

# save study results - also we can resume tuning at a later point in time
with open("test_study.pkl", "wb") as fout:
    pickle.dump(study, fout)

# show best hyperparameters
print(study.best_trial.params)
# {'gradient_clip_val': 0.14918609888209242, 'hidden_size': 97, 'dropout': 0.2842304029635329, 'hidden_continuous_size': 85, 'attention_head_size': 2, 'learning_rate': 0.004385139612980791}

# 评估性能
# PyTorch Lightning 自动检查点训练，因此，我们可以轻松检索最佳模型并加载它。

# load the best model according to the validation loss
# (given that we use early stopping, this is not necessarily the last epoch)
best_model_path = trainer.checkpoint_callback.best_model_path
best_tft = TemporalFusionTransformer.load_from_checkpoint(best_model_path)

# 训练后，我们可以使用 进行预测predict()。
# 该方法允许对其返回的内容进行非常细粒度的控制，例如，您可以轻松地将预测与您的 pandas 数据帧匹配。
# 我们评估验证数据集的指标和几个示例，以了解模型的表现如何。鉴于我们仅使用 21 000 个样本，结果非常令人放心，并且可以与梯度增强器的结果相媲美。
# 我们的表现也优于基线模型。鉴于嘈杂的数据，这并非微不足道。

# calcualte mean absolute error on validation set
actuals = torch.cat([y[0] for x, y in iter(val_dataloader)])
predictions = best_tft.predict(val_dataloader)
(actuals - predictions).abs().mean()
# tensor(265.2021)


# 我们现在还可以直接查看我们用 绘制的样本预测plot_prediction()。
# 从下图中可以看出，预测看起来相当准确。如果您想知道，灰线表示模型在进行预测时对不同时间点的关注程度。这是时间融合变压器的一个特殊功能。

# raw predictions are a dictionary from which all kind of information including quantiles can be extracted
raw_predictions, x = best_tft.predict(val_dataloader, mode="raw", return_x=True)
for idx in range(10):  # plot 10 examples
    best_tft.plot_prediction(x, raw_predictions, idx=idx, add_loss_to_title=True);


# 表现最差
# 查看表现最差的，例如在 方面SMAPE，让我们了解模型在可靠预测方面存在问题的地方。
# 这些示例可以提供有关如何改进模型的重要指示。这种实际与预测图适用于所有模型。
# 当然，使用metrics模块中定义的其他指标（例如 MASE）也是明智的。但是，为了演示，我们只SMAPE在这里使用。

# calcualte metric by which to display
predictions = best_tft.predict(val_dataloader)
mean_losses = SMAPE(reduction="none")(predictions, actuals).mean(1)
indices = mean_losses.argsort(descending=True)  # sort losses
for idx in range(10):  # plot 10 examples
    best_tft.plot_prediction(
        x, raw_predictions, idx=indices[idx], add_loss_to_title=SMAPE(quantiles=best_tft.loss.quantiles)
    );


# 变量的实际值与预测
# 检查模型在不同数据切片上的表现如何让我们发现弱点。下面绘制的是使用 Now 划分为 100 个 bin 的每个变量的预测值与实际值的平均值，
# 我们可以使用calculate_prediction_actual_by_variable()和plot_prediction_actual_by_variable()方法直接预测生成的数据。
# 灰色条表示按 bin 的变量的频率，即是直方图。

predictions, x = best_tft.predict(val_dataloader, return_x=True)
predictions_vs_actuals = best_tft.calculate_prediction_actual_by_variable(x, predictions)
best_tft.plot_prediction_actual_by_variable(predictions_vs_actuals);

# 预测所选数据
# 为了预测数据子集，我们可以使用该filter()方法过滤数据集中的子序列。
# 在这里，我们预测training数据集中映射到组 ID“Agency_01”和“SKU_01”并且其第一个预测值对应于时间索引“15”的子序列。
# 我们输出所有七个分位数。
# 这意味着我们期望一个张量的形状，因为我们预测单个子序列提前六个时间步长和每个时间步长 7 个分位数。
# 1 x n_timesteps x n_quantiles = 1 x 6 x 7

best_tft.predict(
    training.filter(lambda x: (x.agency == "Agency_01") & (x.sku == "SKU_01") & (x.time_idx_first_prediction == 15)),
    mode="quantiles",
)
# tensor([[[ 72.0103, 101.2796, 125.8058, 145.0996, 166.0728, 191.6869, 227.4807],
#          [ 45.2103,  70.3584,  90.0269, 107.9034, 128.3087, 149.9113, 179.7911],
#          [ 45.5753,  67.7294,  84.1701, 100.3818, 120.4252, 140.8588, 169.9457],
#          [ 45.7564,  67.3325,  82.6482,  98.4845, 119.7861, 140.8522, 173.2911],
#          [ 49.6688,  70.4590,  82.5556,  98.6663, 115.5202, 134.6147, 163.6471],
#          [ 36.2067,  58.9810,  72.5047,  89.5287, 107.9294, 128.1242, 157.8855]]])

# 当然，我们也可以很容易地绘制这个预测：

raw_prediction, x = best_tft.predict(
    training.filter(lambda x: (x.agency == "Agency_01") & (x.sku == "SKU_01") & (x.time_idx_first_prediction == 15)),
    mode="raw",
    return_x=True,
)
best_tft.plot_prediction(x, raw_prediction, idx=0);


# 预测新数据
# 因为我们在数据集中有协变量，所以预测新数据需要我们预先定义已知的协变量。

# select last 24 months from data (max_encoder_length is 24)
encoder_data = data[lambda x: x.time_idx > x.time_idx.max() - max_encoder_length]

# select last known data point and create decoder data from it by repeating it and incrementing the month
# in a real world dataset, we should not just forward fill the covariates but specify them to account
# for changes in special days and prices (which you absolutely should do but we are too lazy here)
last_data = data[lambda x: x.time_idx == x.time_idx.max()]
decoder_data = pd.concat(
    [last_data.assign(date=lambda x: x.date + pd.offsets.MonthBegin(i)) for i in range(1, max_prediction_length + 1)],
    ignore_index=True,
)

# add time index consistent with "data"
decoder_data["time_idx"] = decoder_data["date"].dt.year * 12 + decoder_data["date"].dt.month
decoder_data["time_idx"] += encoder_data["time_idx"].max() + 1 - decoder_data["time_idx"].min()

# adjust additional time feature(s)
decoder_data["month"] = decoder_data.date.dt.month.astype(str).astype("category")  # categories have be strings

# combine encoder and decoder data
new_prediction_data = pd.concat([encoder_data, decoder_data], ignore_index=True)

# 现在，我们可以使用该predict()方法直接预测生成的数据。

new_raw_predictions, new_x = best_tft.predict(new_prediction_data, mode="raw", return_x=True)

for idx in range(10):  # plot 10 examples
    best_tft.plot_prediction(new_x, new_raw_predictions, idx=idx, show_future_observed=False);

# 解释模型
# 变量重要性
# 由于其架构的构建方式，该模型具有内置的解释功能。让我们看看它看起来如何。
# 我们首先用 计算解释， interpret_output()然后用 绘制它们plot_interpretation()。

interpretation = best_tft.interpret_output(raw_predictions, reduction="sum")
best_tft.plot_interpretation(interpretation)
# {'attention': <Figure size 640x480 with 1 Axes>,
#  'static_variables': <Figure size 700x375 with 1 Axes>,
#  'encoder_variables': <Figure size 700x525 with 1 Axes>,
#  'decoder_variables': <Figure size 700x350 with 1 Axes>}

# 不出所料，过去观察到的体积特征作为编码器中的顶级变量和价格相关变量是解码器中的顶级预测因子。
#
# 一般的注意力模式似乎是最近的观察更重要和更老。这证实了直觉。平均注意力通常不是很有用 - 通过示例查看注意力更有洞察力，因为模式不是平均的。
#
# 部分依赖
# 部分依赖图通常用于更好地解释模型（假设特征独立）。它们还有助于了解在模拟情况下会发生什么，并使用predict_dependency().

dependency = best_tft.predict_dependency(
    val_dataloader.dataset, "discount_in_percent", np.linspace(0, 30, 30), show_progress_bar=True, mode="dataframe"
)
# plotting median and 25% and 75% percentile
agg_dependency = dependency.groupby("discount_in_percent").normalized_prediction.agg(
    median="median", q25=lambda x: x.quantile(0.25), q75=lambda x: x.quantile(0.75)
)
ax = agg_dependency.plot(y="median")
ax.fill_between(agg_dependency.index, agg_dependency.q25, agg_dependency.q75, alpha=0.3);




def main():
    pass


if __name__ == '__main__':
    main()