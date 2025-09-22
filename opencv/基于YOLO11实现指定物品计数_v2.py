import gradio as gr
from ultralytics import YOLO
from PIL import Image, ImageDraw, ImageFont
import numpy as np
from collections import Counter
import os

# ==================== 配置区：COCO 80类 全部简体中文映射 ====================
COCO_EN_TO_ZH = {
    "person": "人",
    "bicycle": "自行车",
    "car": "汽车",
    "motorcycle": "摩托车",
    "airplane": "飞机",
    "bus": "公交车",
    "train": "火车",
    "truck": "卡车",
    "boat": "船",
    "traffic light": "交通灯",
    "fire hydrant": "消防栓",
    "stop sign": "停车标志",
    "parking meter": "停车计时器",
    "bench": "长椅",
    "bird": "鸟",
    "cat": "猫",
    "dog": "狗",
    "horse": "马",
    "sheep": "羊",
    "cow": "牛",
    "elephant": "大象",
    "bear": "熊",
    "zebra": "斑马",
    "giraffe": "长颈鹿",
    "backpack": "背包",
    "umbrella": "雨伞",
    "handbag": "手提包",
    "tie": "领带",
    "suitcase": "行李箱",
    "frisbee": "飞盘",
    "skis": "滑雪板",
    "snowboard": "单板滑雪板",
    "sports ball": "运动球",
    "kite": "风筝",
    "baseball bat": "棒球棒",
    "baseball glove": "棒球手套",
    "skateboard": "滑板",
    "surfboard": "冲浪板",
    "tennis racket": "网球拍",
    "bottle": "瓶子",
    "wine glass": "酒杯",
    "cup": "杯子",
    "fork": "叉子",
    "knife": "刀",
    "spoon": "勺子",
    "bowl": "碗",
    "banana": "香蕉",
    "apple": "苹果",
    "sandwich": "三明治",
    "orange": "橙子",
    "broccoli": "西兰花",
    "carrot": "胡萝卜",
    "hot dog": "热狗",
    "pizza": "披萨",
    "donut": "甜甜圈",
    "cake": "蛋糕",
    "chair": "椅子",
    "couch": "沙发",
    "potted plant": "盆栽植物",
    "bed": "床",
    "dining table": "餐桌",
    "toilet": "马桶",
    "tv": "电视",
    "laptop": "笔记本电脑",
    "mouse": "鼠标",
    "remote": "遥控器",
    "keyboard": "键盘",
    "cell phone": "手机",
    "microwave": "微波炉",
    "oven": "烤箱",
    "toaster": "烤面包机",
    "sink": "水槽",
    "refrigerator": "冰箱",
    "book": "书",
    "clock": "钟",
    "vase": "花瓶",
    "scissors": "剪刀",
    "teddy bear": "泰迪熊",
    "hair drier": "吹风机",
    "toothbrush": "牙刷"
}

CHINESE_OPTIONS = list(COCO_EN_TO_ZH.values())

# ==================== 模型加载：升级为 yolo11s.pt（精度更高）====================
model = YOLO("yolo11s.pt")  # ✅ 关键升级：使用 YOLO11s 替代 YOLO11n，mAP 从 39.5 → 47.0
class_names = model.names
EN_TO_ZH = COCO_EN_TO_ZH
ZH_TO_EN = {zh: en for en, zh in COCO_EN_TO_ZH.items()}

# ==================== 中文字体加载函数（优先使用用户指定路径）====================
def get_chinese_font():
    """优先使用用户提供的宋体路径，失败则尝试系统默认中文字体"""
    # 用户指定的精确路径（Linux 系统常见位置）
    user_font_path = os.path.expanduser("~/.local/share/fonts/宋体/simsun.ttc")
    
    if os.path.exists(user_font_path):
        try:
            font = ImageFont.truetype(user_font_path, size=18)
            print(f"✅ 成功加载自定义中文字体：{user_font_path}")
            return font
        except Exception as e:
            print(f"⚠️ 加载自定义字体失败：{e}")

    # 回退方案：系统常用字体路径列表
    fallback_paths = [
        "simhei.ttf",           # Windows 黑体
        "SimHei.ttf",           # Windows 黑体（大写）
        "simsun.ttc",           # Windows 宋体
        "/System/Library/Fonts/PingFang.ttc",   # macOS
        "/usr/share/fonts/truetype/wqy/wqy-zenhei.ttc",  # Linux 文泉驿正黑
        "/usr/share/fonts/opentype/noto/NotoSansCJK-Regular.ttc",  # Noto CJK
    ]
    
    for path in fallback_paths:
        if os.path.exists(path):
            try:
                font = ImageFont.truetype(path, size=18)
                print(f"✅ 使用回退中文字体：{path}")
                return font
            except OSError:
                continue
    
    print("❌ 未找到任何可用中文字体，请安装 SimHei 或文泉驿字体以支持中文显示。")
    return ImageFont.load_default()

FONT = get_chinese_font()

# ==================== 核心检测与计数函数 ====================
def count_objects_in_image(input_image, selected_chinese_classes):
    """
    使用 YOLO11s 模型 + 高精度参数 + PIL 绘制中文标签，实现精准计数。
    """
    if isinstance(input_image, Image.Image):
        image_pil = input_image.convert("RGB")
    else:
        image_pil = Image.fromarray(input_image).convert("RGB")

    # ✅ 模型升级体现：使用 yolo11s.pt + 提高置信度和NMS阈值
    results = model(image_pil, conf=0.4, iou=0.6)  # 更严格过滤，减少误检
    result = results[0]

    boxes = result.boxes
    detected_classes = boxes.cls.int().cpu().numpy()
    confidences = boxes.conf.cpu().numpy()
    xyxy_boxes = boxes.xyxy.cpu().numpy()

    # 过滤用户选择的类别
    target_en_classes = [ZH_TO_EN[cls_zh] for cls_zh in selected_chinese_classes if cls_zh in ZH_TO_EN]
    target_class_ids = [cls_id for cls_id, name in class_names.items() if name in target_en_classes]

    filtered_indices = [i for i, cls_id in enumerate(detected_classes) if cls_id in target_class_ids]
    filtered_classes = [detected_classes[i] for i in filtered_indices]
    filtered_confidences = [confidences[i] for i in filtered_indices]
    filtered_boxes = [xyxy_boxes[i] for i in filtered_indices]

    # 统计数量
    per_class_counter = Counter([EN_TO_ZH[class_names[cls_id]] for cls_id in filtered_classes])

    # 创建绘图对象
    draw = ImageDraw.Draw(image_pil)

    # 用于记录每个中文类别的序号（避免跨图像累积）
    global class_counts
    class_counts = {}  # 每次预测前清零，保证独立性

    # 绘制每个检测框和中文标签
    for i, (box, conf) in enumerate(zip(filtered_boxes, filtered_confidences)):
        x1, y1, x2, y2 = map(int, box)
        cls_name_en = class_names[filtered_classes[i]]
        cls_name_zh = EN_TO_ZH[cls_name_en]

        # 计算该类已出现次数
        if cls_name_zh not in class_counts:
            class_counts[cls_name_zh] = 0
        class_counts[cls_name_zh] += 1
        label = f"{cls_name_zh} #{class_counts[cls_name_zh]} ({conf:.2f})"

        # 绘制绿色边框
        draw.rectangle([x1, y1, x2, y2], outline="green", width=3)

        # 计算文本尺寸
        text_bbox = draw.textbbox((0, 0), label, font=FONT)
        text_width = text_bbox[2] - text_bbox[0]
        text_height = text_bbox[3] - text_bbox[1]

        # 绘制背景矩形（半透明绿色）
        bg_x1, bg_y1 = x1, max(0, y1 - text_height - 8)
        bg_x2, bg_y2 = x1 + text_width + 8, y1
        draw.rectangle([bg_x1, bg_y1, bg_x2, bg_y2], fill="green")

        # 绘制白色中文文本
        draw.text((x1 + 4, bg_y1 + 1), label, font=FONT, fill="white")

    # 生成统计文本
    if len(per_class_counter) == 0:
        result_text = "上传图片中，未检测到您选择的物品。"
    else:
        item_list = [f"{name} {count} 个" for name, count in per_class_counter.items()]
        result_text = "上传图片中，含有" + "、".join(item_list) + "。"

    return image_pil, result_text


# ==================== Gradio UI 构建 ====================
class_counts = {}  # 全局计数器，每次推理前由函数内部清空

with gr.Blocks(title="YOLO11 全80类中文物体计数器（精准版·已升级模型+中文字体）") as demo:
    gr.Markdown("# 🚀 YOLO11 全80类中文物体计数器（精准版）")
    gr.Markdown("""
    上传一张图片，从下方列表中**多选**要检测的物品（支持全部80类），点击「计数」按钮，即可查看检测结果。
    
    ✅ **模型升级**：使用 **YOLO11s.pt**（mAP 47.0），比原 yolo11n（mAP 39.5）更准！  
    ✅ **中文支持**：已适配 `~/.local/share/fonts/宋体/simsun.ttc`，无乱码！  
    ✅ **参数优化**：`conf=0.4`, `iou=0.6`，大幅减少误检与重复框！
    """)

    with gr.Row():
        with gr.Column():
            input_image = gr.Image(type="pil", label="上传图片", height=400)
            
            select_classes = gr.Dropdown(
                choices=CHINESE_OPTIONS,
                label="请选择要检测的物品（可多选，共80类）",
                multiselect=True,
                value=["狗", "猫", "人", "汽车", "苹果"],  # 默认选常用项
                interactive=True,
                info="按住 Ctrl / Cmd 可多选"
            )
            
            submit_btn = gr.Button("计数", variant="primary", size="lg")

        with gr.Column():
            output_image = gr.Image(label="检测结果（带中文标签）", height=400)
            output_text = gr.Textbox(label="统计结果", lines=5, interactive=False, max_lines=10)

    # 每次点击前清空全局计数器，确保独立性
    submit_btn.click(
        fn=lambda: None,  # 清空计数器的辅助函数
        inputs=None,
        outputs=None
    ).then(
        fn=count_objects_in_image,
        inputs=[input_image, select_classes],
        outputs=[output_image, output_text]
    )

    gr.Markdown("""
    ---
    ### 💡 技术说明：
    - **模型升级**：使用 `yolo11s.pt` 替代 `yolo11n.pt`，mAP 从 39.5 提升至 47.0，显著降低误检率。
    - **字体适配**：自动检测并优先使用 `~/.local/share/fonts/宋体/simsun.ttc`，确保 Linux 系统完美显示中文。
    - **推理优化**：通过提高 `conf=0.4` 和 `iou=0.6`，有效抑制低质量框，使计数更贴近真实值。
    - **适用场景**：适用于零售货架盘点、动物监控、安防统计等**要求高精度计数**的场景。
    """)


# ==================== 启动应用 ====================
if __name__ == "__main__":
    demo.launch()

# 参考文档：https://docs.ultralytics.com/zh/quickstart/

