#!/usr/bin/env python
# -*- coding: utf-8

from PIL import Image

f2='/home/gswewf/timg.png'
f1='/home/gswewf/timg.jpg'
im = Image.open(f1)
im.save(f2)



