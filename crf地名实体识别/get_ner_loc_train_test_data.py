#coding=utf8

import sys

#home_dir = "D:/source/NLP/people_daily//"

home_dir = "./"

def saveDataFile(trainobj,testobj,isTest,word,handle,tag):
    if isTest:
        saveTrainFile(testobj,word,handle,tag)
    else:
        saveTrainFile(trainobj,word,handle,tag)

def saveTrainFile(fiobj,word,handle,tag): 
    if len(word) > 0 and  word != "。" and word != "，":
        fiobj.write(word + '\t' + handle  + '\t' +tag +'\n')
    else:
        fiobj.write('\n')

#填充地点标注，非地点的不添加
def fill_local_tag(words, tags):
    pos = 0
    while True:
        print ("pos:", pos, " len:", len(words))

        if pos == len(words):
            print ("添加地点tag执行结束")
            print (tags)
            break
        word = words[pos]
        left = word.find("[")
        if left == -1 :
            print ("单个词", word)
            w,h = word.split("/")
            print (w,h)
            if h == "ns": #单个词是地点
                tags[pos] = "LOC_S"
            print("本轮tag", tags[pos])
            pos += 1
        elif left >= 0:
            print("发现词组", word)
            search_pos = pos
            for word in words[pos+1:]:
                print(word)
                search_pos += 1
                if word.find("[") >=0: 
                    print("括号配对异常")
                    sys.exit(255)
                if word.find("]") >=0:
                    break
            if words[search_pos].find("]")  == -1:
                print("括号配对异常，搜索到句尾没有找都另一半括号")
                sys.exit(255)
            else:
                #找到另一半，判断原始标注是不是ns，如果是就进行tag标注
                print("match到一个组", words[pos:search_pos + 1])
                h = words[search_pos].split("]")[-1] #最后一个词性
                if h == "ns":
                    tags[pos] = "LOC_B" #添加首个词
                    for p in range(pos + 1,search_pos + 1):
                      tags[p] = "LOC_I" #中间词
                    tags[search_pos] = "LOC_E" #找到最后一个词
                else:
                    p = pos
                    for word in words[pos:search_pos+1]:
                        print("hhhhhhh", word)
                        w,h = word.strip("[").split("]")[0].split("/")
                        if h == "ns":
                            tags[p] = "LOC_S"
                        p += 1       

            #移动pos
            print("本轮添加的tag", tags[pos:search_pos + 1])
            pos = search_pos + 1



def convertTag():
    if len(sys.argv) > 1:
        data_file = sys.argv[1]
    else:
        data_file = 'people-daily.txt'  # 数据在：https://pan.baidu.com/s/1gemKdoR#list/path=%2F
    fiobj    = open( home_dir + data_file,'r')
    trainobj = open( home_dir +'train.data','w' )
    testobj  = open( home_dir  +'test.data','w')

    arr = fiobj.readlines()
    i = 0
    # for a in sys.stdin:
    for a in arr:
        i += 1
        a = a.strip('\r\n\t ')
        if a=="":continue
        words = a.split(" ")
        test = False
        if i % 5 == 0:
            test = True
        words = words[1:]
        if len(words) == 0: continue

        tags = ["O"] * len(words)
        fill_local_tag(words, tags)

        pos = -1
        for word in words:
            pos += 1
            print("---->", word)
            word = word.strip('\t ')
            if len(word) == 0:
                print("Warning 发现空词")
                continue

            l1 = word.find('[')
            if l1 >=0:
                word = word[l1+1:]

            l2 = word.find(']')
            if l2 >= 0:
                word = word[:l2]

            w,h = word.split('/')
            
            saveDataFile(trainobj,testobj,test,w,h,tags[pos])
        saveDataFile(trainobj, testobj, test,"","","")
            
    trainobj.flush()
    testobj.flush()

if __name__ == '__main__':    
    convertTag()

