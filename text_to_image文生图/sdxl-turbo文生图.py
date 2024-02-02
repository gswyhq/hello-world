
# 文生图
from diffusers import AutoPipelineForText2Image
import torch
from modelscope import snapshot_download

model_dir = snapshot_download("AI-ModelScope/sdxl-turbo")

pipe = AutoPipelineForText2Image.from_pretrained(model_dir, torch_dtype=torch.float16, variant="fp16")
pipe.to("cuda")

prompt = "Beautiful and cute girl, 16 years old, denim jacket, gradient background, soft colors, soft lighting, cinematic edge lighting, light and dark contrast, anime, art station Seraflur, blind box, super detail, 8k"

image = pipe(prompt=prompt, num_inference_steps=1, guidance_scale=0.0).images[0]
image.save("image.png")

# 图生图

from diffusers import AutoPipelineForImage2Image
from diffusers.utils import load_image
from modelscope import snapshot_download
model_dir = snapshot_download("AI-ModelScope/sdxl-turbo")
pipe = AutoPipelineForImage2Image.from_pretrained(model_dir,  variant="fp16")

init_image = load_image("image.png").resize((512, 512))

prompt = "grey image"

image = pipe(prompt, image=init_image, num_inference_steps=2, strength=0.5, guidance_scale=0.0).images[0]
image.save("image1.png")


LCM-SDXL最佳实践

文生图推理：

from diffusers import UNet2DConditionModel, DiffusionPipeline, LCMScheduler
import torch
from modelscope import snapshot_download

model_dir_lcm = snapshot_download("AI-ModelScope/lcm-sdxl",revision = "master")
model_dir_sdxl = snapshot_download("AI-ModelScope/stable-diffusion-xl-base-1.0",revision = "v1.0.9")

unet = UNet2DConditionModel.from_pretrained(model_dir_lcm, torch_dtype=torch.float16, variant="fp16")
pipe = DiffusionPipeline.from_pretrained(model_dir_sdxl, unet=unet, torch_dtype=torch.float16, variant="fp16")

pipe.scheduler = LCMScheduler.from_config(pipe.scheduler.config)
pipe.to("cuda")

prompt = "Beautiful and cute girl, 16 years old, denim jacket, gradient background, soft colors, soft lighting, cinematic edge lighting, light and dark contrast, anime, art station Seraflur, blind box, super detail, 8k"


image = pipe(prompt, num_inference_steps=4, guidance_scale=8.0).images[0]
image.save("image.png")

本文使用的模型链接：

SDXL-Turbo：

https://modelscope.cn/models/AI-ModelScope/sdxl-turbo



LCM-SDXL：

https://modelscope.cn/models/AI-ModelScope/lcm-sdxl


LCM-LoRA-SDXL：

https://modelscope.cn/models/AI-ModelScope/lcm-lora-sdxl


相关模型链接：

SDXL 1.0：

https://modelscope.cn/models/AI-ModelScope/stable-diffusion-xl-base-1.0

