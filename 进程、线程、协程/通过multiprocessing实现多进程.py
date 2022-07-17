#/usr/lib/python3.5
# -*- coding: utf-8 -*-

#https://segmentfault.com/a/1190000000414339
#这段代码的主要工作就是将遍历传入的文件夹中的图片文件，一一生成缩略图，并将这些缩略图保存到特定文件夹中。


import os 
import PIL 

from multiprocessing import Pool 
from PIL import Image

SIZE = (75,75)
SAVE_DIRECTORY = 'thumbs'

def get_image_paths(folder):
    return (os.path.join(folder, f) 
            for f in os.listdir(folder) 
            if 'jpeg' in f)

def create_thumbnail(filename): 
    im = Image.open(filename)
    im.thumbnail(SIZE, Image.ANTIALIAS)
    base, fname = os.path.split(filename) 
    save_path = os.path.join(base, SAVE_DIRECTORY, fname)
    im.save(save_path)

if __name__ == '__main__':
    folder = os.path.abspath(
        '11_18_2013_R000_IQM_Big_Sur_Mon__e10d1958e7b766c3e840')
    os.mkdir(os.path.join(folder, SAVE_DIRECTORY))

    images = get_image_paths(folder)

    pool = Pool()
    pool.map(creat_thumbnail, images)
    pool.close()
    pool.join()
