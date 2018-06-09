import pygame

# 来源：https://zhuanlan.zhihu.com/p/26726297

def generate_dict():
    dic = {}
    # 文件`unicode_pinyin.txt`的链接
    # https://pan.baidu.com/share/link?shareid=1866601421&uk=4013012087；密码：fge1
    with open("/home/gswewf/data/unicode_pinyin.txt") as f:
        for i in f.readlines():
            dic[i.split()[0]] = i.split()[1]
    return dic

dic = generate_dict()

def chinese_to_pinyin(x):
    y = ''

    for i in x:
        i = str(i.encode('unicode_escape'))[-5:-1].upper()
        try:
            y += dic[i] + ' '
        except:
            y += 'XXXX '
    return y

vioce_file='/home/gswewf/ekho-7.5.1/ekho-data/pinyin/{}.wav'

def make_vioce(x):
    pygame.mixer.init()
    voi = chinese_to_pinyin(x).split()
    for i in voi:
        if i == 'XXXX':
            continue
        print(i)
        # pygame.mixer.music.load("voice/" + i + ".mp3")
        pygame.mixer.music.load(vioce_file.format(i.lower()))
        pygame.mixer.music.play()  # 播放音乐
        while pygame.mixer.music.get_busy() == True:
            # pygame.mixer.music.get_busy
            # 说明：判断当前是否有音乐在播放
            pass
    return None

def main():
    p = input("请输入文字：")
    make_vioce(p)

if __name__ == '__main__':
    main()
