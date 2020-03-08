#!/usr/bin/python3
# coding: utf-8


# 第一步，下载数据集；
# Librispeech数据分布在压缩tar文件中，比如 train-clean-100.tar.gz 用于训练，而 dev-clean.tar.gz 用于验证。
# 解压后，每个归档创建一个名为 LibriSpeech的目录
# tar zxvf dev-clean.tar.gz
# LibriSpeech/LICENSE.TXT
# LibriSpeech/README.TXT
# LibriSpeech/CHAPTERS.TXT
# LibriSpeech/SPEAKERS.TXT
# LibriSpeech/BOOKS.TXT
# LibriSpeech/dev-clean/
# LibriSpeech/dev-clean/2277/
# LibriSpeech/dev-clean/2277/149896/
# LibriSpeech/dev-clean/2277/149896/2277-149896-0026.flac
# LibriSpeech/dev-clean/2277/149896/2277-149896-0005.flac
# LibriSpeech/dev-clean/2277/149896/2277-149896-0033.flac
# LibriSpeech/dev-clean/2277/149896/2277-149896-0006.flac
# ...

# 下载训练验证数据并解压
# $ mkdir librispeech && cd librispeech
# $ wget http://www.openslr.org/resources/12/train-clean-100.tar.gz
# $ wget http://www.openslr.org/resources/12/dev-clean.tar.gz
# $ tar xvzf dev-clean.tar.gz LibriSpeech/dev-clean --strip-components=1
# $ tar xvzf train-clean-100.tar.gz LibriSpeech/train-clean-100 --strip-components=1


# 按照上面的方法，将训练数据作为子目录 librispeech/train-clean-100 和子目录 librispeech/dev-clean 中的验证数据。
# 要获取数据，你可以在解压缩干净训练数据的目录上运行 python 脚本，然后按照你希望脚本编写记录和训练mainfests的位置来执行操作：

# https://github.com/NervanaSystems/deepspeech/blob/master/speech/data/ingest_librispeech.py
# /home/gswyhq/github_projects/deepspeech/speech

# $ python3 data/ingest_librispeech.py /home/gswyhq/data/LibriSpeech/dev-clean /home/gswyhq/data/LibriSpeech/dev-transcripts_dir /home/gswyhq/data/LibriSpeech/dev-manifest.csv

# 参数1：/home/gswyhq/data/LibriSpeech/dev-clean
# librispeech目录的路径, 程序会在该目录中递归查找所有.flac文件，
# 然后提取附近的.trans.txt文件，并将每个.flac文件对应的文本写到新目录文件中

# 参数2: /home/gswyhq/data/LibriSpeech/dev-transcripts_dir
# 用于保存每个.flac文件对应的文本文件的目录

# 参数3：/home/gswyhq/data/LibriSpeech/dev-manifest.csv
# flac文件与其对应的文本文件映射


# 训练
root@bd244b3a6a15:/deepspeech/speech# python3 data/ingest_librispeech.py /data/LibriSpeech/dev-clean /data/LibriSpeech/dev-transcripts-dir /data/LibriSpeech/dev-manifest.csv

root@bd244b3a6a15:/deepspeech/speech# python3 data/ingest_librispeech.py /data/LibriSpeech/train-clean-100 /data/LibriSpeech/train-transcripts-dir /data/LibriSpeech/train-manifest.csv

root@bd244b3a6a15:/deepspeech/speech# nohup python3 -u train.py --manifest train:/data/LibriSpeech/train-manifest.csv --manifest val:/data/LibriSpeech/dev-manifest.csv -e 1 -z 16 -s model_output.pkl -b cpu > train.log &

# Epoch 0   [Train |█                   |  115/1784 batches, 500.97 cost, 49597.69s]

# 评估

# $ python evaluate.py --manifest val:/path/to/manifest.csv --model_file/path/to/saved_model.prm


# https://devhub.io/repos/NervanaSystems-deepspeech
# https://www.kutu66.com//GitHub/article_150630

def main():
    pass


if __name__ == '__main__':
    main()