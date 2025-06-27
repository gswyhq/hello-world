
png to svg
方法一：
将 PNG 转为 Base64 并嵌入 SVG：
1、生成一个 SVG 文件，内容如下：
<svg xmlns="http://www.w3.org/2000/svg" width="800" height="600" viewBox="0 0 800 600">
  <image href="data:image/png;base64,[你的PNG Base64编码]" width="800" height="600"/>
</svg>
2、替换 [你的PNG Base64编码]：
在终端中运行：
base64 output_resized.png | tr -d '\n' > base64.txt
3、将生成的文本粘贴到 SVG 文件中。


方法二：
from PIL import Image
import svgwrite
import numpy as np
def convert_image_to_svg(input_image_path, output_svg_path, max_size=64, merge_threshold=10):
    # 打开图像并缩小尺寸
    img = Image.open(input_image_path)
    img = img.convert("RGBA")
    # 计算缩放比例
    scale = min(max_size / img.width, max_size / img.height)
    new_width = int(img.width * scale)
    new_height = int(img.height * scale)
    # 缩小图像尺寸
    img = img.resize((new_width, new_height), Image.LANCZOS)
    # 转换为numpy数组
    data = np.array(img)
    # 创建SVG
    dwg = svgwrite.Drawing(output_svg_path, size=(new_width, new_height))
    # 合并相邻同色像素块
    for y in range(new_height):
        x = 0
        while x < new_width:
            r, g, b, a = data[y, x]
            if a == 0:
                x += 1
                continue
            # 寻找相同颜色的连续区域
            w = 1
            while x + w < new_width and np.all(data[y, x + w] == [r, g, b, a]) and w < merge_threshold:
                w += 1
            dwg.add(dwg.rect(
                insert=(x, y),
                size=(w, 1),
                fill=svgwrite.rgb(r, g, b)
            ))
            x += w
    dwg.save()
    
convert_image_to_svg("input_image.png", "output.svg")


from PIL import Image
import svgwrite
import numpy as np

def convert_image_to_svg(input_image_path, output_svg_path, max_size=64, merge_threshold=10):
    # 打开图像并缩小尺寸
    img = Image.open(input_image_path)
    img = img.convert("RGB")  # 转换为RGB模式，因为JPG通常是不透明的
    # 计算缩放比例
    scale = min(max_size / img.width, max_size / img.height)
    new_width = int(img.width * scale)
    new_height = int(img.height * scale)
    # 缩小图像尺寸
    img = img.resize((new_width, new_height), Image.LANCZOS)
    # 转换为numpy数组
    data = np.array(img)
    # 创建SVG
    dwg = svgwrite.Drawing(output_svg_path, size=(new_width, new_height))
    # 合并相邻同色像素块
    for y in range(new_height):
        x = 0
        while x < new_width:
            r, g, b = data[y, x]
            # 寻找相同颜色的连续区域
            w = 1
            while x + w < new_width and np.all(data[y, x + w] == [r, g, b]) and w < merge_threshold:
                w += 1
            dwg.add(dwg.rect(
                insert=(x, y),
                size=(w, 1),
                fill=svgwrite.rgb(r, g, b)
            ))
            x += w
    dwg.save()

# 调用函数，输入JPG文件
convert_image_to_svg("input_image.jpg", "output.svg")


