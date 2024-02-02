
# 我们可以通过Tesseract来识别图片中的文字，在Python中实现起来非常简单，但是前期下载文件、配置环境变量等稍微有些繁琐，所以本文只展示代码：

import pytesseract
from PIL import Image
img = Image.open('text.jpg')
text = pytesseract.image_to_string(img)
print(text)
# 其中text就是识别出来的文本。

