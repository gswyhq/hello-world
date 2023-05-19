#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import torch
import os
import math
import pandas as pd
import numpy as np
from glob import glob
from tqdm import tqdm
import re
import json
import functools
import itertools
import datasets
from datasets import ClassLabel, load_dataset
from typing import Callable, List, Optional, Tuple, Type, Union

USERNAME = os.getenv("USERNAME")


# 加载csv格式数据：
raw_datasets = load_dataset('csv', sep='\t', encoding='utf-8', data_files={"train": rf"D:\Users\{USERNAME}\data\数据梳理\经纬度地理位置\20230428\data\train.csv", "validation": rf"D:\Users\{USERNAME}\data\数据梳理\经纬度地理位置\20230428\data\dev.csv"}, )

# 加载json格式数据：
raw_datasets = load_dataset('json', features=datasets.Features(
                {
                    "id": datasets.Value("string"),
                    "tokens": datasets.Sequence(datasets.Value("string")),
                    "ner_tags": datasets.Sequence(datasets.features.ClassLabel(
                            names=['LABEL_0',
                                 'LABEL_1',
                                 'LABEL_2',
                                 'LABEL_3',
                                 'LABEL_4',
                                 'LABEL_5',
                                 'LABEL_6',
                                 'LABEL_7',
                                 'LABEL_8',
                                 'LABEL_9',
                                 'LABEL_10',
                                 'LABEL_11',
                                 'LABEL_12',
                                 'LABEL_13',
                                 'LABEL_14']
                        ))
                }
            ), data_files={"train": rf"D:\Users\{USERNAME}\data\数据梳理\经纬度地理位置\20230428\data\train.json", "validation": rf"D:\Users\{USERNAME}\data\数据梳理\经纬度地理位置\20230428\data\dev.json"}, )


