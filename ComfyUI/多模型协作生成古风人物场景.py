#!/usr/bin/env python
# coding=utf-8

'''
输入：妈妈和小女孩的肖像图（两张图片）。
输出：一张新的图像，内容为：
中国古代农家小院。
妈妈站在厨房门口，对小女孩说话。
小女孩从厨房方向跑向院子的篱笆大门。

多模型协作流程，包括：
使用 IP-Adapter / InstantID 注入人物身份特征（保留输入人物的面部特征）。
使用 Stable Diffusion 生成图像。
使用 ControlNet 控制人物姿势。
使用 FaceSwap 替换人脸。

完整工作流
[Load Image (mom.jpg)] -> [IP-Adapter / InstantID] -> [CLIP Text Encode] -> [KSampler] -> [VAE Decode] -> [FaceSwap] -> [Save Image]
[Load Image (girl.jpg)] -> [IP-Adapter / InstantID] -----------------------------^
[Load Pose (mom_pose.jpg)] -> [ControlNet (pose)] ----------------------------------^
[Load Pose (girl_pose.jpg)] -> [ControlNet (pose)] ---------------------------------^

############################################################################################################################
# 使用 ControlNet 生成 pose 图像
from diffusers import StableDiffusionControlNetPipeline, ControlNetModel

controlnet = ControlNetModel.from_pretrained("lllyasviel/control_v11p_sd15_openpose")
pipe = StableDiffusionControlNetPipeline.from_pretrained("runwayml/stable-diffusion-v1-4", controlnet=controlnet)

prompt = "A woman standing at the kitchen door, looking at a girl."
image = pipe(prompt=prompt, num_inference_steps=50).images[0]
image.save("mom_pose.jpg")

'''

import torch
from diffusers import (
    StableDiffusionControlNetPipeline,
    ControlNetModel,
    IPAdapterPlus,
    StableDiffusionPipeline
)
from diffusers.utils import load_image, make_image_grid
from facexlib.utils.face_restoration_helper import FaceRestoreHelper
from facexlib.utils import img2tensor, tensor2img
from torchvision.transforms import ToTensor, ToPILImage
import os

# 设置设备
device = "cuda" if torch.cuda.is_available() else "cpu"
torch_dtype = torch.float16 if torch.cuda.is_available() else torch.float32

# 如果你需要的是从头生成图像，使用 FLUX.1-dev、Qwen/Qwen-Image；
# 如果你需要的是在已有图像上进行编辑，使用 FLUX.1-Kontext-dev、Qwen/Qwen-Image-Edit-2509。
# 图像理解，生成图像描述：Qwen/Qwen2.5-VL-7B-Instruct

# 加载模型
base_model_path = "black-forest-labs/FLUX.1-dev" # "runwayml/stable-diffusion-v1-4"
controlnet_path = "Shakker-Labs/FLUX.1-dev-ControlNet-Union-Pro-2.0" # "lllyasviel/control_v11p_sd15_openpose"
ip_adapter_path = "huggingface.co/h94/IP-Adapter/sdxl_models"

# 加载 ControlNet 模型
controlnet = ControlNetModel.from_pretrained(controlnet_path, torch_dtype=torch_dtype)
controlnet = controlnet.to(device)

# 加载 IP-Adapter
ip_adapter = IPAdapterPlus.from_pretrained(ip_adapter_path, torch_dtype=torch_dtype).to(device)

# 加载 Stable Diffusion Pipeline
pipe = StableDiffusionControlNetPipeline.from_pretrained(
    base_model_path,
    controlnet=controlnet,
    torch_dtype=torch_dtype,
    safety_checker=None
).to(device)

# 加载 IP-Adapter 到 Pipeline
pipe.load_ip_adapter(ip_adapter_path, subfolder="ipadapter", weight_name="ip-adapter-plus_sd15.bin")
pipe.set_ip_adapter_scale(0.8)

# 加载 FaceSwap 模型
face_restore = FaceRestoreHelper(
    upscale_factor=1,
    face_size=512,
    crop_ratio=(1, 1),
    det_model='retinaface_resnet50',
    save_ext='png'
)

# 加载图像
mom_image = load_image("mom.jpg").convert("RGB").resize((768, 768))
girl_image = load_image("girl.jpg").convert("RGB").resize((768, 768))

# 加载 pose 图像（从网络下载或使用 ControlNet 生成）
mom_pose = load_image("mom_pose.jpg").convert("RGB").resize((768, 768))
girl_pose = load_image("girl_pose.jpg").convert("RGB").resize((768, 768))

# 构建提示词
prompt = "A traditional Chinese rural courtyard in ancient times. A middle-aged woman stands at the kitchen door, looking at a young girl running from the kitchen towards the fence gate. The courtyard has a wooden fence, a small garden, and a thatched roof house."
negative_prompt = "low quality, bad anatomy, extra limbs, bad proportions"

# 生成图像
image = pipe(
    prompt=prompt,
    negative_prompt=negative_prompt,
    image=[mom_image, girl_image],
    control_image=[mom_pose, girl_pose],
    num_inference_steps=50,
    guidance_scale=7.5,
    ip_adapter_image=[mom_image, girl_image],
    ip_adapter_scale=0.8,
    width=768,
    height=768,
    generator=torch.Generator(device=device).manual_seed(42)
).images[0]

# 保存生成图像
image.save("generated.png")
print("生成图像已保存为 generated.png")

# 使用 FaceSwap 替换人脸
face_restore.get_face_info(image)
face_restore.restore_faces()
restored_image = face_restore.get_restored_face()

# 保存最终图像
restored_image.save("final_output.png")
print("最终图像已保存为 final_output.png")

def main():
    pass


if __name__ == "__main__":
    main()
