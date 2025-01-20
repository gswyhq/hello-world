

from PIL import Image

def replace_color(input_image_path, output_image_path, target_color, new_color):
    # 打开输入图片
    img = Image.open(input_image_path).convert("RGBA")
    
    # 创建一个新的图片
    data = img.getdata()
    
    new_data = []
    for item in data:
        # 如果当前像素是目标颜色，替换为新颜色
        if item[0:3] == target_color:  # RGB值
            new_data.append(new_color + (item[3],))  # 保持透明度
        else:
            new_data.append(item)  # 不改变其他颜色
            
    img.putdata(new_data)
    img.save(output_image_path)

# 示例使用
target_color = (255, 255, 255)  # 白色
new_color = (255, 0, 0)  # 红色 
# (0, 0, 255) # 蓝色；
# (29, 109, 243) # color="#296DF3"
replace_color("transparent_image.png", "output_image.png", target_color, new_color)




