
from openai import OpenAI
import httpx
# 初始化OpenAI客户端，设置基础URL和API密钥
client = OpenAI(
    base_url="https://localhost:8080/v1/",
    api_key="OPENAI_API_KEY",  # 从环境变量中获取API密钥
    http_client=httpx.Client(verify=False)
)
# 使用模型生成模型响应
response = client.chat.completions.create(
    model="model",
    temperature=0,
    messages=[
        {"role": "system", "content": """你的角色是AI图片生成机器人，接下来我会给你一些中文关键词描述，请你在不影响我关键词描述的情况下，先根据我的描述进行文本润色、丰富描述细节，之后转换成英文，并将英文文本填充到下面URL链接的占位符prompt中:

![image]

(https://image.pollinations.ai/prompt/{prompt}?width=1024&height=1024&enhance=true&private=true&nologo=true&safe=true&model=flux)

生成后请你给出你润色后的中文提示语。"""},
        {"role": "user", "content": "一只小鱼在地铁上面荡秋千"}
    ]
)
# 返回模型的响应答案；
print(response.choices[0].message.content)


