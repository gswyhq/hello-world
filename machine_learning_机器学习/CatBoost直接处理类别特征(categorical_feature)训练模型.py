#!/usr/bin/env python3
# -*- coding: utf-8 -*-

# 示例来源：https://catboost.ai/en/docs/concepts/python-usages-examples

from catboost import CatBoostClassifier, Pool

train_data = Pool(
    [
        [[0.1, 0.12, 0.33], [1.0, 0.7], 2, "male"],
        [[0.0, 0.8, 0.2], [1.1, 0.2], 1, "female"],
        [[0.2, 0.31, 0.1], [0.3, 0.11], 2, "female"],
        [[0.01, 0.2, 0.9], [0.62, 0.12], 1, "male"]
    ],
    label = [1, 0, 0, 1],
    cat_features=[3],
    embedding_features=[0, 1]
)

eval_data = Pool(
    [
        [[0.2, 0.1, 0.3], [1.2, 0.3], 1, "female"],
        [[0.33, 0.22, 0.4], [0.98, 0.5], 2, "female"],
        [[0.78, 0.29, 0.67], [0.76, 0.34], 2, "male"],
    ],
    label = [0, 1, 1],
    cat_features=[3],  # 第3列为类别特征
    embedding_features=[0, 1], # 第 0、1列为向量特征
)

model = CatBoostClassifier(iterations=10)

model.fit(train_data, eval_set=eval_data)
preds = model.predict_proba(eval_data)


def main():
    pass


if __name__ == '__main__':
    main()
