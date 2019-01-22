# /usr/lib/python3.5
# -*- coding: utf-8 -*-

from PIL import Image

def gif_to_png(gif_file):
    #将gif图片转成PNG图片
    assert gif_file.lower().endswith('.gif'), '输入应该为gif文件'
    save_file = gif_file.lower().replace('.gif', '.png')
    im = Image.open(gif_file)
    def iter_frames(im):
        try:
            i= 0
            while 1:
                im.seek(i)
                imframe = im.copy()
                if i == 0:
                    palette = imframe.getpalette()
                else:
                    imframe.putpalette(palette)
                yield imframe
                i += 1
        except EOFError:
            pass
    for i, frame in enumerate(iter_frames(im)):
        frame.save(save_file,**frame.info)

def jpeg_to_png(jpeg_file):
    
    assert jpeg_file.lower().endswith('.jpeg'), '输入应该为jepg格式文件'
    save_file = jpeg_file.replace('.jpeg', '.png')
    im = Image.open(jpeg_file)

    im.save(save_file, "png")  # 保存图像为png格式

def main():
    # gif_to_png('/home/gswyhq/five_in_a_row/black.gif')
    # gif_to_png('/home/gswyhq/five_in_a_row/white.gif')
    jpeg_to_png('/home/gswyhq/five_in_a_row/bg.jpeg')

if __name__ == "__main__":
    main()
