
MCP 实际上就是调用工具的过程；
tools=[{'type': 'function', 'function': {'name': 'add', 'description': '这是个加法器，用于计算两个数字和', 'parameters': {'properties': {'a': {'type': 'integer'}, 'b': {'type': 'integer'}}, 'required': ['a', 'b'], 'type': 'object'}}}] 

工具调用的流程：
用户问题 + tools 参数传入工具的描述信息给到大模型 -> 大模型输出工具名称解析工具入参 或给出回复 -> 若模型返回结果是要调用工具，程序根据工具名称、入参调用工具 -> 根据工具结果、用户问题再次调用大模型
# 来源：https://help.aliyun.com/zh/model-studio/qwen-function-calling

虽然深度思考模型拥有强大的推理能力，但无法与外部世界进行交互。Function Calling 通过引入外部工具，可以帮助深度思考模型实现天气查询、数据库查询、发送邮件等功能。
DeepSeek-R1 系列模型暂不支持 Function Calling。
Qwen3、QwQ 模型会在思考完成后进行工具调用信息的输出，tool_choice参数只支持设置为"auto"（默认值，表示由模型自主选择工具）或"none"（强制模型不选择工具）。

import os
from openai import OpenAI

# 初始化OpenAI客户端，配置阿里云DashScope服务
client = OpenAI(
    # 若没有配置环境变量，请用阿里云百炼API Key将下行替换为：api_key="sk-xxx",
    api_key=os.getenv("DASHSCOPE_API_KEY"),  # 从环境变量读取API密钥
    base_url="https://dashscope.aliyuncs.com/compatible-mode/v1",
)

# 定义可用工具列表
tools = [
    # 工具1 获取当前时刻的时间
    {
        "type": "function",
        "function": {
            "name": "get_current_time",
            "description": "当你想知道现在的时间时非常有用。",
            "parameters": {}  # 无需参数
        }
    },  
    # 工具2 获取指定城市的天气
    {
        "type": "function",
        "function": {
            "name": "get_current_weather",
            "description": "当你想查询指定城市的天气时非常有用。",
            "parameters": {  
                "type": "object",
                "properties": {
                    "location": {
                        "type": "string",
                        "description": "城市或县区，比如北京市、杭州市、余杭区等。"
                    }
                },
                "required": ["location"]  # 必填参数
            }
        }
    }
]

messages = [{"role": "user", "content": input("请输入问题：")}]
completion = client.chat.completions.create(
    # 此处以qwen-plus-2025-04-28为例，可更换为其它深度思考模型
    model="qwen-plus-2025-04-28",
    messages=messages,
    extra_body={
        # 开启深度思考，该参数对 QwQ 模型无效
        "enable_thinking": True
    },
    tools=tools,
    parallel_tool_calls=True,
    stream=True,
    # 解除注释后，可以获取到token消耗信息
    # stream_options={
    #     "include_usage": True
    # }
)

reasoning_content = ""  # 定义完整思考过程
answer_content = ""     # 定义完整回复
tool_info = []          # 存储工具调用信息
is_answering = False   # 判断是否结束思考过程并开始回复
print("="*20+"思考过程"+"="*20)
for chunk in completion:
    if not chunk.choices:
        # 处理用量统计信息
        print("\n"+"="*20+"Usage"+"="*20)
        print(chunk.usage)
    else:
        delta = chunk.choices[0].delta
        # 处理AI的思考过程（链式推理）
        if hasattr(delta, 'reasoning_content') and delta.reasoning_content is not None:
            reasoning_content += delta.reasoning_content
            print(delta.reasoning_content,end="",flush=True)  # 实时输出思考过程
            
        # 处理最终回复内容
        else:
            if not is_answering:  # 首次进入回复阶段时打印标题
                is_answering = True
                print("\n"+"="*20+"回复内容"+"="*20)
            if delta.content is not None:
                answer_content += delta.content
                print(delta.content,end="",flush=True)  # 流式输出回复内容
            
            # 处理工具调用信息（支持并行工具调用）
            if delta.tool_calls is not None:
                for tool_call in delta.tool_calls:
                    index = tool_call.index  # 工具调用索引，用于并行调用
                    
                    # 动态扩展工具信息存储列表
                    while len(tool_info) <= index:
                        tool_info.append({})
                    
                    # 收集工具调用ID（用于后续函数调用）
                    if tool_call.id:
                        tool_info[index]['id'] = tool_info[index].get('id', '') + tool_call.id
                    
                    # 收集函数名称（用于后续路由到具体函数）
                    if tool_call.function and tool_call.function.name:
                        tool_info[index]['name'] = tool_info[index].get('name', '') + tool_call.function.name
                    
                    # 收集函数参数（JSON字符串格式，需要后续解析）
                    if tool_call.function and tool_call.function.arguments:
                        tool_info[index]['arguments'] = tool_info[index].get('arguments', '') + tool_call.function.arguments
            
print(f"\n"+"="*19+"工具调用信息"+"="*19)
if not tool_info:
    print("没有工具调用")
else:
    print(tool_info)

# 资料来源：https://help.aliyun.com/zh/model-studio/deep-thinking


