
# 使用Python将图片转为ICO图标
# pip install pillow

# 使用以下Python代码将JPEG或PNG图片转换为ICO图标：

from PIL import Image


def png_to_ico(png_path, ico_path, sizes=[(64, 64), (128, 128), (256, 256)]):
    """
    将PNG图片转换为ICO格式
    :param png_path: 图片路径
    :param ico_path: 输出ICO图片路径
    :param sizes: ICO文件中包含的图标尺寸列表
    """
    # 打开PNG图片
    img = Image.open(png_path)

    # 保存为ICO文件
    img.save(ico_path, format='ICO', sizes=sizes)


png_path = 'xxx.jpg'  # 图片路径
ico_path = 'xxx.ico'  # 保存的ICO图片路径

png_to_ico(png_path, ico_path)

# 这段代码会读取指定路径下的图片文件，并将其保存为ICO格式的图标。注意，图片应该是一个正方形的图像，因为ICO图标的大小通常是正方形的。如果图片不是正方形，PIL会自动裁剪或填充图片来创建正方形的图标

