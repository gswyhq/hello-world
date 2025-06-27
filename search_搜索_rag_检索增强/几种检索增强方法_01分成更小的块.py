
# 检索增强生成 （RAG） 是一种将信息检索与生成模型相结合的混合方法。它通过整合外部知识来增强语言模型的性能，从而提高准确性和事实正确性。

####################################################################################################################################
# 方法1:
# 分块：将数据分成更小的块以提高检索性能。
# 嵌入创建：使用嵌入模型将文本块转换为数字表示形式。
# 语义搜索：根据用户查询检索相关块。
# 响应生成：使用语言模型根据检索到的文本生成响应。
import fitz # pip3 install PyMuPDFb
import os
import numpy as np
import json
from openai import OpenAI

# 初始化OpenAI客户端，设置基础URL和API密钥
client = OpenAI(
    base_url="https://api.studio.nebius.com/v1/",
    api_key=os.getenv("OPENAI_API_KEY")  # 从环境变量中获取API密钥
)

def extract_text_from_pdf(pdf_path):
    """
    从PDF文件中提取文本并打印前num_chars个字符。

    Args:
        pdf_path (str): PDF文件的路径。

    Returns:
        str: 从PDF中提取的文本。
    """
    # 打开PDF文件
    mypdf = fitz.open(pdf_path)
    all_text = ""  # 初始化一个空字符串以存储提取的文本

    # 遍历PDF中的每一页
    for page_num in range(mypdf.page_count):
        page = mypdf[page_num]  # 获取页面
        text = page.get_text("text")  # 从页面中提取文本
        all_text += text  # 将提取的文本追加到all_text字符串中

    return all_text  # 返回提取的文本

# 将提取的文本分块成 1000 个字符的片段,重叠 200 个字符
def chunk_text(text, n, overlap):
    """
    将给定文本分块为 n 个字符的重叠段。
    参数:
    text (str):要分块的文本。
    n (int):每个块中的字符数。
    overlap (int):块之间的重叠字符数。

    返回:
    List[str]:文本块的列表。
    """
    chunks = [] # 初始化一个空列表来存储 chunk

    # 以 (n - overlap) 的步长遍历文本
    for i in range(0, len(text), n - overlap):
        # 将索引 i 到 i n 中的文本块附加到块列表中
        chunks.append(文本[i:i+n])

    return chunks # 返回文本块列表

text_chunks = chunk_text(extracted_text, 1000, 200)

system_prompt = "您是一个 AI 助手，严格根据给定的上下文进行回答。如果无法直接从提供的上下文中得出答案，请回答：'我没有足够的信息来回答这个问题。'"

# 定义评估系统的系统提示符
evaluate_system_prompt = "您是一个智能评估系统，负责评估 AI 助手的响应。如果 AI 助手的响应非常接近真实响应，则分配分数 1。如果响应相对于真实响应不正确或不满意，则分配分数 0。如果响应与真实响应部分一致，则分配 0.5 分。"
# 结合用户查询、AI 响应、真实响应和评估系统提示来创建评估提示
evaluation_prompt = f"用户查询：{query}\nAI 响应：n{ai_response.choices[0].message.content}n真实响应：{data[0]['ideal_answer']}\n{evaluate_system_prompt}"

####################################################################################################################################




# 资料来源：https://github.com/FareedKhan-dev/all-rag-techniques.git


