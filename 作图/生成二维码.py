
# 生成二维码
# 二维码又称二维条码，常见的二维码为QR Code，QR全称Quick Response，是一个近几年来移动设备上超流行的一种编码方式，而生成一个二维码也非常简单，在Python中我们可以通过MyQR模块了生成二维码，而生成一个二维码我们只需要2行代码，我们先安装MyQR模块，这里选用国内的源下载：

# pip install qrcode
# 安装完成后我们就可以开始写代码了：

import qrcode

text = input(输入文字或URL：)
# 设置URL必须添加http://
img =qrcode.make(text)
img.save()
#保存图片至本地目录，可以设定路径
img.show()


# 我们执行代码后会在项目下生成一张二维码。当然我们还可以丰富二维码：
# 我们先安装MyQR模块

# pip install  myqr
def gakki_code():
    version, level, qr_name = myqr.run(
        words=https://520mg.com/it/#/main/2,
        # 可以是字符串，也可以是网址(前面要加http(s)://)
        version=1,  # 设置容错率为最高
        level='H',
        # 控制纠错水平，范围是L、M、Q、H，从左到右依次升高
        picture=gakki.gif,
        # 将二维码和图片合成
        colorized=True,  # 彩色二维码
        contrast=1.0,
         # 用以调节图片的对比度，1.0 表示原始图片，更小的值表示更低对比度，更大反之。默认为1.0
        brightness=1.0,
        # 用来调节图片的亮度，其余用法和取值同上
        save_name=gakki_code.gif,
        # 保存文件的名字，格式可以是jpg,png,bmp,gif
        save_dir=os.getcwd()  # 控制位置

    )
 gakki_code()

