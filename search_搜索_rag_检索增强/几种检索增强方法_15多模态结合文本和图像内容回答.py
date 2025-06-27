
# 检索增强生成 （RAG） 是一种将信息检索与生成模型相结合的混合方法。它通过整合外部知识来增强语言模型的性能，从而提高准确性和事实正确性。


####################################################################################################################################
# 方法15：
# 一个多模态RAG系统，该系统可以从文档中提取文本和图像，为图像生成描述，并结合文本和图像内容来回答查询。这种方法通过将视觉信息纳入知识库，增强了传统的RAG系统。
#
# 传统的RAG系统仅处理文本，但许多文档中包含关键信息的图像、图表和表格。通过为这些视觉元素生成描述并将其纳入检索系统，我们可以：
#
# - 访问图表和示意图中的信息
# - 理解补充文本的表格和图表
# - 创建更全面的知识库
# - 回答依赖视觉数据的问题

import os
import io
import numpy as np
import json
import fitz
from PIL import Image
from openai import OpenAI
import base64
import re
import tempfile
import shutil

# 文档处理函数
def extract_content_from_pdf(pdf_path, output_dir=None):
    """
    从PDF文件中提取文本和图像。

    参数:
        pdf_path (str): PDF文件的路径
        output_dir (str, 可选): 保存提取图像的目录

    返回:
        Tuple[List[Dict], List[Dict]]: 文本数据和图像数据
    """
    # 创建临时目录以保存图像（如果未提供）
    temp_dir = None
    if output_dir is None:
        temp_dir = tempfile.mkdtemp()
        output_dir = temp_dir
    else:
        os.makedirs(output_dir, exist_ok=True)

    text_data = []  # 存储提取文本数据的列表
    image_paths = []  # 存储图像路径的列表

    print(f"从{pdf_path}提取内容...")

    try:
        with fitz.open(pdf_path) as pdf_file:
            # 遍历PDF中的每一页
            for page_number in range(len(pdf_file)):
                page = pdf_file[page_number]

                # 从页面提取文本
                text = page.get_text().strip()
                if text:
                    text_data.append({
                        "content": text,
                        "metadata": {
                            "source": pdf_path,
                            "page": page_number + 1,
                            "type": "text"
                        }
                    })

                # 从页面提取图像
                image_list = page.get_images(full=True)
                for img_index, img in enumerate(image_list):
                    xref = img[0]  # 图像的XREF
                    base_image = pdf_file.extract_image(xref)

                    if base_image:
                        image_bytes = base_image["image"]
                        image_ext = base_image["ext"]

                        # 将图像保存到输出目录
                        img_filename = f"page_{page_number+1}_img_{img_index+1}.{image_ext}"
                        img_path = os.path.join(output_dir, img_filename)

                        with open(img_path, "wb") as img_file:
                            img_file.write(image_bytes)

                        image_paths.append({
                            "path": img_path,
                            "metadata": {
                                "source": pdf_path,
                                "page": page_number + 1,
                                "image_index": img_index + 1,
                                "type": "image"
                            }
                        })
        print(f"提取了{len(text_data)}个文本片段和{len(image_paths)}个图像")
        return text_data, image_paths

    except Exception as e:
        print(f"提取内容时出错: {e}")
        if temp_dir and os.path.exists(temp_dir):
            shutil.rmtree(temp_dir)
        raise

# 文本分块
def chunk_text(text_data, chunk_size=1000, overlap=200):
    """
    将文本数据分割为重叠的块。

    参数:
        text_data (List[Dict]): 从PDF中提取的文本
        chunk_size (int): 每个块的大小（以字符为单位）
        overlap (int): 块之间的重叠（以字符为单位）

    返回:
        List[Dict]: 分割后的文本数据
    """
    chunked_data = []  # 初始化一个空列表以存储分块数据

    for item in text_data:
        text = item["content"]  # 提取文本内容
        metadata = item["metadata"]  # 提取元数据

        # 如果文本太短则跳过
        if len(text) < chunk_size / 2:
            chunked_data.append({
                "content": text,
                "metadata": metadata
            })
            continue

        # 创建具有重叠的块
        chunks = []
        for i in range(0, len(text), chunk_size - overlap):
            chunk = text[i:i + chunk_size]  # 提取指定大小的块
            if chunk:  # 确保不添加空块
                chunks.append(chunk)

        # 添加每个块并更新元数据
        for i, chunk in enumerate(chunks):
            chunk_metadata = metadata.copy()  # 复制原始元数据
            chunk_metadata["chunk_index"] = i  # 添加块索引到元数据
            chunk_metadata["chunk_count"] = len(chunks)  # 添加块总数到元数据

            chunked_data.append({
                "content": chunk,  # 块文本
                "metadata": chunk_metadata  # 更新后的元数据
            })

    print(f"创建了{len(chunked_data)}个文本块")  # 打印创建的块数
    return chunked_data  # 返回分块数据列表

# 使用OpenAI Vision进行图像描述
def encode_image(image_path):
    """
    将图像文件编码为base64。

    参数:
        image_path (str): 图像文件的路径

    返回:
        str: base64编码的图像
    """
    # 以二进制读取模式打开图像文件
    with open(image_path, "rb") as image_file:
        # 读取图像文件并编码为base64
        encoded_image = base64.b64encode(image_file.read())
        # 将base64字节解码为字符串并返回
        return encoded_image.decode('utf-8')

def generate_image_caption(image_path):
    """
    使用OpenAI的视觉功能为图像生成描述。

    参数:
        image_path (str): 图像文件的路径

    返回:
        str: 生成的描述
    """
    # 检查文件是否存在且是图像
    if not os.path.exists(image_path):
        return "错误: 图像文件未找到"

    try:
        # 打开并验证图像
        Image.open(image_path)

        # 将图像编码为base64
        base64_image = encode_image(image_path)

        # 创建API请求以生成描述
        response = client.chat.completions.create(
            model="llava-hf/llava-1.5-7b-hf",  # 使用llava-1.5-7b模型
            messages=[
                {
                    "role": "system",
                    "content": "你是一个专门描述学术论文图像的助手。"
                    "为图像提供详细描述，捕捉关键信息。"
                    "如果图像包含图表、表格或示意图，请清晰描述其内容和目的。"
                    "你的描述应优化为未来检索时，当人们询问此内容时使用。"
                },
                {
                    "role": "user",
                    "content": [
                        {"type": "text", "text": "详细描述这张图像，重点放在其学术内容上:"},
                        {
                            "type": "image_url",
                            "image_url": {
                                "url": f"data:image/jpeg;base64,{base64_image}"
                            }
                        }
                    ]
                }
            ],
            max_tokens=300
        )

        # 从响应中提取描述
        caption = response.choices[0].message.content
        return caption

    except Exception as e:
        # 如果发生异常返回错误消息
        return f"生成描述时出错: {str(e)}"

def process_images(image_paths):
    """
    处理所有图像并生成描述。

    参数:
        image_paths (List[Dict]): 提取图像的路径

    返回:
        List[Dict]: 带有描述的图像数据
    """
    image_data = []  # 初始化一个空列表以存储带有描述的图像数据

    print(f"为{len(image_paths)}个图像生成描述...")  # 打印要处理的图像数量
    for i, img_item in enumerate(image_paths):
        print(f"处理图像 {i+1}/{len(image_paths)}...")  # 打印当前正在处理的图像
        img_path = img_item["path"]  # 获取图像路径
        metadata = img_item["metadata"]  # 获取图像元数据

        # 为图像生成描述
        caption = generate_image_caption(img_path)

        # 将图像数据与描述添加到列表中
        image_data.append({
            "content": caption,  # 生成的描述
            "metadata": metadata,  # 图像元数据
            "image_path": img_path  # 图像路径
        })

    return image_data  # 返回带有描述的图像数据列表

# 简单向量存储实现

class MultiModalVectorStore:
    """
    一个多模态内容的简单向量存储实现。
    """
    def __init__(self):
        # 初始化存储向量、内容和元数据的列表
        self.vectors = []
        self.contents = []
        self.metadata = []

    def add_item(self, content, embedding, metadata=None):
        """
        将一个项目添加到向量存储中。

        参数:
            content (str): 内容（文本或图像描述）
            embedding (List[float]): 嵌入向量
            metadata (Dict, 可选): 额外的元数据
        """
        # 将嵌入向量、内容和元数据添加到各自的列表中
        self.vectors.append(np.array(embedding))
        self.contents.append(content)
        self.metadata.append(metadata or {})

    def add_items(self, items, embeddings):
        """
        将多个项目添加到向量存储中。

        参数:
            items (List[Dict]): 内容项目列表
            embeddings (List[List[float]]): 嵌入向量列表
        """
        # 遍历项目和嵌入并将每个添加到向量存储中
        for item, embedding in zip(items, embeddings):
            self.add_item(
                content=item["content"],
                embedding=embedding,
                metadata=item.get("metadata", {})
            )

    def similarity_search(self, query_embedding, k=5):
        """
        找到与查询嵌入最相似的项目。

        参数:
            query_embedding (List[float]): 查询嵌入向量
            k (int): 要返回的结果数量

        返回:
            List[Dict]: 最相似的前k个项目
        """
        # 如果存储中没有向量则返回空列表
        if not self.vectors:
            return []

        # 将查询嵌入转换为numpy数组
        query_vector = np.array(query_embedding)

        # 使用余弦相似度计算相似性
        similarities = []
        for i, vector in enumerate(self.vectors):
            similarity = np.dot(query_vector, vector) / (np.linalg.norm(query_vector) * np.linalg.norm(vector))
            similarities.append((i, similarity))

        # 按相似度（降序）排序
        similarities.sort(key=lambda x: x[1], reverse=True)

        # 返回前k个结果
        results = []
        for i in range(min(k, len(similarities))):
            idx, score = similarities[i]
            results.append({
                "content": self.contents[idx],
                "metadata": self.metadata[idx],
                "similarity": float(score)  # 转换为float以进行JSON序列化
            })

        return results

# 创建嵌入
def create_embeddings(texts, model="BAAI/bge-en-icl"):
    """
    为给定的文本创建嵌入。

    参数:
        texts (List[str]): 输入文本
        model (str): 嵌入模型名称

    返回:
        List[List[float]]: 嵌入向量
    """
    # 处理空输入
    if not texts:
        return []

    # 分批处理（OpenAI API限制）
    batch_size = 100
    all_embeddings = []

    # 遍历输入文本的批次
    for i in range(0, len(texts), batch_size):
        batch = texts[i:i + batch_size]  # 获取当前批次的文本

        # 为当前批次创建嵌入
        response = client.embeddings.create(
            model=model,
            input=batch
        )

        # 从响应中提取嵌入
        batch_embeddings = [item.embedding for item in response.data]
        all_embeddings.extend(batch_embeddings)  # 将批次嵌入添加到列表中

    return all_embeddings  # 返回所有嵌入

# 完整处理管道
def process_document(pdf_path, chunk_size=1000, chunk_overlap=200):
    """
    为多模态RAG处理文档。

    参数:
        pdf_path (str): PDF文件的路径
        chunk_size (int): 每个块的大小（以字符为单位）
        chunk_overlap (int): 块之间的重叠（以字符为单位）

    返回:
        Tuple[MultiModalVectorStore, Dict]: 向量存储和文档信息
    """
    # 创建用于提取图像的目录
    image_dir = "extracted_images"
    os.makedirs(image_dir, exist_ok=True)

    # 从PDF中提取文本和图像
    text_data, image_paths = extract_content_from_pdf(pdf_path, image_dir)

    # 分割提取的文本
    chunked_text = chunk_text(text_data, chunk_size, chunk_overlap)

    # 处理提取的图像以生成描述
    image_data = process_images(image_paths)

    # 结合所有内容项（文本块和图像描述）
    all_items = chunked_text + image_data

    # 提取内容进行嵌入
    contents = [item["content"] for item in all_items]

    # 为所有内容创建嵌入
    print("为所有内容创建嵌入...")
    embeddings = create_embeddings(contents)

    # 构建向量存储并将项目及其嵌入添加进去
    vector_store = MultiModalVectorStore()
    vector_store.add_items(all_items, embeddings)

    # 准备带有文本块和图像描述数量的文档信息
    doc_info = {
        "text_count": len(chunked_text),
        "image_count": len(image_data),
        "total_items": len(all_items),
    }

    # 打印添加到存储中的项目摘要
    print(f"将{len(all_items)}个项目添加到向量存储（{len(chunked_text)}个文本块，{len(image_data)}个图像描述）")

    # 返回向量存储和文档信息
    return vector_store, doc_info

# 查询处理和响应生成

def query_multimodal_rag(query, vector_store, k=5):
    """
    查询多模态RAG系统。

    参数:
        query (str): 用户查询
        vector_store (MultiModalVectorStore): 包含文档内容的向量存储
        k (int): 要检索的结果数量

    返回:
        Dict: 查询结果和生成的响应
    """
    print(f"\n=== 处理查询: {query} ===\n")

    # 为查询生成嵌入
    query_embedding = create_embeddings(query)

    # 从向量存储中检索相关内容
    results = vector_store.similarity_search(query_embedding, k=k)

    # 分离文本和图像结果
    text_results = [r for r in results if r["metadata"].get("type") == "text"]
    image_results = [r for r in results if r["metadata"].get("type") == "image"]

    print(f"检索到{len(results)}个相关项目（{len(text_results)}个文本，{len(image_results)}个图像描述）")

    # 生成响应
    response = generate_response(query, results)

    return {
        "query": query,
        "results": results,
        "response": response,
        "text_results_count": len(text_results),
        "image_results_count": len(image_results)
    }

def generate_response(query, results):
    """
    基于查询和检索结果生成响应。

    参数:
        query (str): 用户查询
        results (List[Dict]): 检索到的内容

    返回:
        str: 生成的响应
    """
    # 格式化检索到的内容
    context = ""

    for i, result in enumerate(results):
        # 确定内容类型（文本或图像描述）
        content_type = "文本" if result["metadata"].get("type") == "text" else "图像描述"
        # 从元数据中获取页码
        page_num = result["metadata"].get("page", "未知")

        # 将内容类型和页码添加到上下文中
        context += f"[{content_type}来自第{page_num}页]\n"
        # 将实际内容添加到上下文中
        context += result["content"]
        context += "\n\n"

    # 指导AI助手的系统消息
    system_message = """你是一个专门回答包含文本和图像的文档问题的AI助手。
    你已经从文档中获得了相关文本段落和图像描述。请根据这些信息提供一个全面、准确的回答。
    如果信息来自图像或图表，请在回答中提及。
    如果检索到的信息无法完全回答查询，请承认其局限性。"""

    # 包含查询和格式化上下文的用户消息
    user_message = f"""查询: {query}

    检索到的内容:
    {context}

    请根据检索到的内容回答查询。
    """

    # 使用OpenAI API生成响应
    response = client.chat.completions.create(
        model="meta-llama/Llama-3.2-3B-Instruct",
        messages=[
            {"role": "system", "content": system_message},
            {"role": "user", "content": user_message}
        ],
        temperature=0.1
    )

    # 返回生成的响应
    return response.choices[0].message.content

# 与仅文本RAG的评估

def build_text_only_store(pdf_path, chunk_size=1000, chunk_overlap=200):
    """
    为比较构建仅文本向量存储。

    参数:
        pdf_path (str): PDF文件的路径
        chunk_size (int): 每个块的大小（以字符为单位）
        chunk_overlap (int): 块之间的重叠（以字符为单位）

    返回:
        MultiModalVectorStore: 仅文本向量存储
    """
    # 从PDF中提取文本（重用函数但忽略图像）
    text_data, _ = extract_content_from_pdf(pdf_path, None)

    # 分割文本
    chunked_text = chunk_text(text_data, chunk_size, chunk_overlap)

    # 提取内容进行嵌入
    contents = [item["content"] for item in chunked_text]

    # 创建嵌入
    print("为仅文本内容创建嵌入...")
    embeddings = create_embeddings(contents)

    # 构建向量存储
    vector_store = MultiModalVectorStore()
    vector_store.add_items(chunked_text, embeddings)

    print(f"将{len(chunked_text)}个文本项添加到仅文本向量存储")
    return vector_store

def evaluate_multimodal_vs_textonly(pdf_path, test_queries, reference_answers=None):
    """
    比较多模态RAG与仅文本RAG。

    参数:
        pdf_path (str): PDF文件的路径
        test_queries (List[str]): 测试查询
        reference_answers (List[str], 可选): 参考答案

    返回:
        Dict: 评估结果
    """
    print("=== 评估多模态RAG与仅文本RAG ===\n")

    # 为多模态RAG处理文档
    print("\n为多模态RAG处理文档...")
    mm_vector_store, mm_doc_info = process_document(pdf_path)

    # 构建仅文本存储
    print("\n为仅文本RAG处理文档...")
    text_vector_store = build_text_only_store(pdf_path)

    # 为每个查询运行评估
    results = []

    for i, query in enumerate(test_queries):
        print(f"\n\n=== 评估查询 {i+1}: {query} ===")

        # 获取参考答案（如果可用）
        reference = None
        if reference_answers and i < len(reference_answers):
            reference = reference_answers[i]

        # 运行多模态RAG
        print("\n运行多模态RAG...")
        mm_result = query_multimodal_rag(query, mm_vector_store)

        # 运行仅文本RAG
        print("\n运行仅文本RAG...")
        text_result = query_multimodal_rag(query, text_vector_store)

        # 比较响应
        comparison = compare_responses(query, mm_result["response"], text_result["response"], reference)

        # 添加到结果中
        results.append({
            "query": query,
            "multimodal_response": mm_result["response"],
            "textonly_response": text_result["response"],
            "multimodal_results": {
                "text_count": mm_result["text_results_count"],
                "image_count": mm_result["image_results_count"]
            },
            "reference_answer": reference,
            "comparison": comparison
        })

    # 生成整体分析
    overall_analysis = generate_overall_analysis(results)

    return {
        "results": results,
        "overall_analysis": overall_analysis,
        "multimodal_doc_info": mm_doc_info
    }

def compare_responses(query, mm_response, text_response, reference=None):
    """
    比较多模态和仅文本响应。

    参数:
        query (str): 用户查询
        mm_response (str): 多模态响应
        text_response (str): 仅文本响应
        reference (str, 可选): 参考答案

    返回:
        str: 比较分析
    """
    # 评估器的系统提示
    system_prompt = """你是一个比较两个RAG系统的专家:
    1. 多模态RAG: 从文本和图像描述中检索
    2. 仅文本RAG: 仅从文本中检索

    根据以下标准评估哪个响应更好地回答查询:
    - 准确性和正确性
    - 信息的完整性
    - 与查询的相关性
    - 视觉元素的独特信息（对于多模态）"""

    # 包含查询和响应的用户提示
    user_prompt = f"""查询: {query}

    多模态RAG响应:
    {mm_response}

    仅文本RAG响应:
    {text_response}
    """

    if reference:
        user_prompt += f"""参考答案:
    {reference}
    """

        user_prompt += """比较这些响应并解释哪个更好地回答查询以及原因。
    注意多模态响应中来自图像的任何特定信息。
    """

    # 使用meta-llama/Llama-3.2-3B-Instruct生成比较
    response = client.chat.completions.create(
        model="meta-llama/Llama-3.2-3B-Instruct",
        messages=[
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt}
        ],
        temperature=0
    )

    return response.choices[0].message.content

def generate_overall_analysis(results):
    """
    生成多模态与仅文本RAG的整体分析。

    参数:
        results (List[Dict]): 每个查询的评估结果

    返回:
        str: 整体分析
    """
    # 评估器的系统提示
    system_prompt = """你是一个RAG系统的专家评估者。根据多个测试查询，提供多模态RAG（文本+图像）
    与仅文本RAG的比较分析。

    重点:
    1. 多模态RAG优于仅文本RAG的查询类型
    2. 图像信息带来的具体优势
    3. 多模态方法的任何缺点或限制
    4. 推荐每种方法的使用场景"""

    # 创建评估摘要
    evaluations_summary = ""
    for i, result in enumerate(results):
        evaluations_summary += f"查询 {i+1}: {result['query']}\n"
        evaluations_summary += f"多模态检索到 {result['multimodal_results']['text_count']} 个文本块和 {result['multimodal_results']['image_count']} 个图像描述\n"
        evaluations_summary += f"比较摘要: {result['comparison'][:200]}...\n\n"

    # 包含评估摘要的用户提示
    user_prompt = f"""根据以下多模态与仅文本RAG的评估结果，提供这两种方法的比较分析:
    {evaluations_summary}

    请根据检索到的内容全面分析多模态RAG与仅文本RAG的相对优缺点，
    特别注意图像信息对响应质量的贡献（或未贡献）。"""

    # 使用meta-llama/Llama-3.2-3B-Instruct生成整体分析
    response = client.chat.completions.create(
        model="meta-llama/Llama-3.2-3B-Instruct",
        messages=[
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt}
        ],
        temperature=0
    )

    return response.choices[0].message.content



####################################################################################################################################




# 资料来源：https://github.com/FareedKhan-dev/all-rag-techniques.git


