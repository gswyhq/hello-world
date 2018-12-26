
1、py 文件运行正常，但是编译成so文件出错
    q = ['123']
    ds = [q for q in [['abcd']]]
    print("q: {}".format(q))

在py程序中最后输出的会是"q: ['123']"， 但编译成so文件后会输出的是“q: ['abcd']”;

2、py文件编译成so文件后，运行耗时增加：
    datas = ''
    for value in dict_data:
        datas += json.dumps(value, ensure_ascii=False) + '\n'

上面代码，当数据量`dict_data`比较大的时候耗时会增加，建议改成下面这样：
    datas = []
    for value in dict_data:
        datas.append(json.dumps(value, ensure_ascii=False) + '\n')
    datas = ''.join(datas)

当然，当数据量`dict_data`比较大的时候，若改成下面这样，py文件即使不编译耗时也会增加：
    datas = ''
    for value in dict_data:
        datas = datas + json.dumps(value, ensure_ascii=False) + '\n'


