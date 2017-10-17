#!/usr/bin/python3
# coding: utf-8

# http://blog.csdn.net/chunyun0716/article/details/60465806

import codecs
import jieba
import gensim
from gensim.models.word2vec import LineSentence

stopword_zh_file = '/home/gswyhq/data/jieba_dict/stopwords.txt'

def read_source_file(source_file_name):
    try:
        file_reader = codecs.open(source_file_name, 'r', 'utf-8',errors="ignore")
        lines = file_reader.readlines()
        print("Read complete!")
        file_reader.close()
        return lines
    except:
        print("There are some errors while reading.")

def write_file(target_file_name, content):

    file_write = codecs.open(target_file_name, 'w+', 'utf-8')
    file_write.writelines(content)
    print("Write sussfully!")
    file_write.close()

def separate_word(filename,user_dic_file, separated_file):
    print("separate_word")
    lines = read_source_file(filename)
    #jieba.load_userdict(user_dic_file)
    stopkey=[line.strip() for line in codecs.open(stopword_zh_file,'r','utf-8').readlines()]

    output = codecs.open(separated_file, 'w', 'utf-8')
    num = 0
    for line in lines:
        num = num + 1
        if num% 10000 == 0:
            print("Processing line number: " + str(num))
        seg_word_line = jieba.cut(line, cut_all = True)
        wordls = list(set(seg_word_line)-set(stopkey))
        if len(wordls)>0:
            word_line = ' '.join(wordls) + '\n'
            output.write(word_line)
    output.close()
    return separated_file


def build_model(source_separated_words_file,model_path):

    print("start building...",source_separated_words_file)
    # 将分词的结果作为模型的输入
    model = gensim.models.Word2Vec(LineSentence(source_separated_words_file), size=200, window=5, min_count=5, alpha=0.02, workers=4)
    model.save(model_path)
    print("build successful!", model_path)
    return model

def get_similar_words_str(w, model, topn = 10):
    result_words = get_similar_words_list(w, model)
    return str(result_words)


def get_similar_words_list(w, model, topn = 10):
    result_words = []
    try:
        # 保存模型，方便以后调用，获得目标词的同义词
        similary_words = model.most_similar(w, topn=10)
        print(similary_words)
        for (word, similarity) in similary_words:
            result_words.append(word)
        print(result_words)
    except:
        print("There are some errors!" + w)

    return result_words

def load_models(model_path):
    return gensim.models.Word2Vec.load(model_path)


def main():
    # filename = "d:\\data\\dk_mainsuit_800w.txt" #source file
    # user_dic_file = "new_dict.txt" # user dic file
    # separated_file = "d:\\data\\dk_spe_file_20170216.txt" # separeted words file
    # model_path = "information_model0830" # model file

    kefu_file = '/home/gswyhq/data/log/kefu/result.txt'
    filename = kefu_file
    user_dic_file = ''
    separated_file = '/home/gswyhq/data/log/kefu/result_jieba.txt'
    model_path = '/home/gswyhq/data/log/kefu/information_model'
    source_separated_words_file = separate_word(filename, user_dic_file, separated_file)
    # source_separated_words_file = separated_file    # if separated word file exist, don't separate_word again
    build_model(source_separated_words_file, model_path)# if model file is exist, don't buile modl

    model = load_models(model_path)
    words = get_similar_words_str('头痛', model)
    print(words)

    # 判断词是否在训练的模型中
    # In[23]: '头痛' in model
    # Out[23]: False

    # 通过模型计算两句话的WMD距离
    distance = model.wmdistance(['a', '卡能', '打电话', '吗'], ['a', '卡', '在', '哪里', '能', '打电话'])

    # In[28]: distance
    # Out[28]: 7.949923043079514


if __name__ == '__main__':
    main()
