#!/usr/bin/env python
# coding=utf-8

'''
不是通过pip3 install fontforge 安装
而是通过：
root@991d5a285c99:~# apt-get install fontforge
安装之后，可以通过如下命令查询对应的python环境：
fontforge -script - <<< 'import sys; print(sys.path)'
fontforge -script - <<< 'import sys; import fontforge; print(fontforge); print(sys.modules["fontforge"])'
输出为：
<module 'fontforge' (built-in)>
<module 'fontforge' (built-in)>
这说明：
fontforge 是一个 内置模块（built-in），不是普通的 Python 模块。
它是 FontForge 自带的 Python 接口，只能在 FontForge 自带的 Python 环境中使用。
要使用 FontForge 自带的 Python 环境运行脚本，如：fontforge -script your_script.py

如果你必须在系统 Python 中使用 FontForge
由于 FontForge 的 Python 模块是内置的，无法直接在系统 Python 中使用，除非你：
从源码编译 FontForge，并启用 Python 模块的共享库支持（--enable-python）。
将 fontforge.so 编译为共享库，并手动安装到 Python 的 site-packages 中。
'''

import fontforge

# 打开字体文件
font = fontforge.open("方正粗圆简体.ttf")

# 遍历所有字符,查看字体文件中包含哪些字符（输出所有字体文字）
print("字体中包含的字符：")
for glyph in font.glyphs():
    if glyph.unicode != -1:
        try:
            char = chr(glyph.unicode)
            print(f"Unicode: {glyph.unicode:04X} -> 字符: {char}")
        except:
            print(f"Unicode: {glyph.unicode:04X} -> 无法解码")




##########################################################################################################################
# 修改字体名称和版权信息
# 打开字体文件
import fontforge
font = fontforge.open("方正粗圆简体.ttf")

# 修改字体信息
font.fontname = "MyCustomFont"
font.familyname = "MyCustomFont"
font.fullname = "MyCustomFont"
font.copyright = "Copyright © 2025 My Company"

# 保存为新字体
font.generate("MyCustomFont.ttf")

##########################################################################################################################
# FontForge 提取某个字符的字形轮廓，并保存为 SVG 文件。

# ✅ 示例代码：提取“中”字为 SVG

import fontforge

# 打开字体文件
font = fontforge.open("宋体.ttc")

for word in '中国人':
    # 选择“中”字的 Unicode 编码
    unicode_code = ord(word)  # 20013

    # 获取对应的字形
    glyph = font[unicode_code]

    # 导出为 SVG 文件
    glyph.export(f"{word}字轮廓.svg")

##########################################################################################################################

# 4. 生成只包含特定字符的新字体（如“中”字）

import fontforge

# 打开字体
font = fontforge.open("宋体.ttc")

# 清除所有字符
for glyph in font.glyphs():
    font.removeGlyph(glyph)

# 添加“中”字
glyph = font.createChar(ord("中"), "uni4E2D")
glyph.importOutlines("中字轮廓.svg")  # 你需要提供 SVG 字形文件

# 保存为新字体
font.generate("中字字体.ttf")

##########################################################################################################################
# 保留所有字符，只修改某个字符（比如“中”字），可以这样写：
# ✅ 示例代码：保留所有字符，只修改“中”字

import fontforge

# 打开字体文件
font = fontforge.open("方正粗圆简体.ttf")
for word in '中国人':
    # 选择“中”字的 Unicode 编码
    unicode_code = ord(word)  # 20013

    # 创建新字形
    glyph = font.createChar(unicode_code, f"uni{unicode_code:04X}")
    glyph.clear()
    # 导入新的 SVG 字形（替换原字形）
    glyph.importOutlines(f"{word}字轮廓.svg")

# 保存修改后的字体
font.generate("修改后的方正粗圆简体.ttf")

##########################################################################################################################
# 使用修改后的字体画图，对比效果
from PIL import Image,ImageDraw,ImageFont
for font_name in [ '修改后的方正粗圆简体.ttf', '方正粗圆简体.ttf', 'simsun.ttc',]:
    font = ImageFont.truetype(rf'fonts\{font_name}', size=45,encoding='utf-8')
    img=Image.new('RGB',(800,200),(255,255,255))#200*200的白色背景
    draw=ImageDraw.Draw(img)
    draw.line((20,50,150,80),fill=(255,0,0))#红色线条
    draw.line((150,150,20,200),fill=(0,255,0))#绿色线条
    draw.text((40,80),'您好,我是中国人，中国人你好！',(0,0,0),font=font)#黑色文本
    img.save(rf'fonts\{font_name[:-4]}.jpg','JPEG')
    
def main():
    pass


if __name__ == "__main__":
    main()

# fontforge -script your_script.py
