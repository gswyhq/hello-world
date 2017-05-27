#!/usr/bin/env python
# -*- coding: utf-8

from PIL import Image

img = Image.open("imageToSave.png")
r, g, b, a = img.split()
img = Image.merge("RGB", (r, g, b))
img.save('imagetoadd.bmp')
xlwt.insert_bitmap('imagetoadd.bmp', row, col)


