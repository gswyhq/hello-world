#!/bin/bash

path=`pwd`
cd ${path}

echo "生成所需要的训练和测试数据"
python3 get_ner_loc_train_test_data.py people-daily.txt > log.txt

# 通过下面命令执行训练和测试过程：
echo "开始训练"
crf_learn -f 4 -p 40 -c 3 template train.data model > train.rst  

echo "生成测试数据"
crf_test -m model test.data > test.rst

echo "分类型计算F值"
python3 clc.py test.rst

