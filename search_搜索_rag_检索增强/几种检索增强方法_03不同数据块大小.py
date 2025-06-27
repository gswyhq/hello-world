
# 检索增强生成 （RAG） 是一种将信息检索与生成模型相结合的混合方法。它通过整合外部知识来增强语言模型的性能，从而提高准确性和事实正确性。

####################################################################################################################################
# 方法3：
# 选择正确的数据块大小对于提高检索增强生成 （RAG）的检索准确性至关重要。目标是平衡检索性能与响应质量。
#
# 通过以下方式评估不同的数据块大小：
#
# 从 PDF 中提取文本。
# 将文本拆分为大小不同的块。
# 为每个 chunk 创建 embedding。
# 检索查询的相关块。
# 使用检索到的块生成响应。
# 评估忠实度和相关性。
# 比较不同块大小的结果。

def chunk_text(text, n, overlap):
    """
    将文本分割为重叠的块

    Args:
    text (str): 要分块的文本.
    n (int): 每块的文本字符数
    overlap (int): 块直接重叠的字符数

    Returns:
    List[str]: 文本块列表
    """
    chunks = []
    for i in range(0, len(text), n - overlap):
        # 将当前索引对应的分块文本添加到文本块列表
        chunks.append(text[i:i + n])

    return chunks


# 定义不同的分块文本大小来进行评估；
chunk_sizes = [128, 256, 512]

# 创建一个字典来存储每个分块尺寸对应的分块文本；
text_chunks_dict = {size: chunk_text(extracted_text, size, size // 5) for size in chunk_sizes}

# 为每个文本块生成对应的向量；
chunk_embeddings_dict = {size: [get_embedding(chunk) for chunk in chunks] for size, chunks in tqdm(text_chunks_dict.items(), desc="Generating Embeddings")}

# 检索每个块相关的块
retrieved_chunks_dict = {size: semantic_search(query, text_chunks_dict[size], chunk_embeddings_dict[size]) for size in chunk_sizes}


system_prompt = "您是一个 AI 助手，严格根据给定的上下文进行回答。如果无法直接从提供的上下文中得出答案，请回答：'我没有足够的信息来回答这个问题'。"


def generate_response(query, system_prompt, retrieved_chunks, model="meta-llama/Llama-3.2-3B-Instruct"):
    """
    根据检索到的数据分块，生成对应的模型相应

    Args:
    query (str): User query.
    retrieved_chunks (List[str]): 检索到的文本块列表.
    model (str): AI model.

    Returns:
    str: 生成的AI模型相应.
    """
    # 将检索到的文本块合并到上下文字符串中；
    context = "\n".join([f"Context {i + 1}:\n{chunk}" for i, chunk in enumerate(retrieved_chunks)])

    # 根据上下文及用户问题，生成对应的用户提示词；
    user_prompt = f"{context}\n\nQuestion: {query}"

    # 使用模型生成模型响应
    response = client.chat.completions.create(
        model=model,
        temperature=0,
        messages=[
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt}
        ]
    )

    # 返回模型的响应答案；
    return response.choices[0].message.content


# 为每个分块生成对应的模型答案；
ai_responses_dict = {size: generate_response(query, system_prompt, retrieved_chunks_dict[size]) for size in chunk_sizes}

# 定义评估评分系统常量
SCORE_FULL = 1.0 # 完全匹配或完全满意
SCORE_PARTIAL = 0.5 # 部分匹配或比较满意
SCORE_NONE = 0.0 # 不匹配或不满意
# 定义严格的评估提示模板
FAITHFULNESS_PROMPT_TEMPLATE = """
评估 AI 响应与真实答案的忠实度。
用户查询： {question}
AI 响应：{response}
正确答案： {true_answer}

忠实度用于衡量AI回答与真实答案中的事实是否一致，避免产生不实信息或幻觉。

指示：
- 仅使用以下值严格评分：
* {full} = 完全忠实，与真实答案没有矛盾
* {partial} = 部分忠实，轻微矛盾
* {none} = 不忠实，重大矛盾或幻觉
- 仅返回数字分数（{full}、{partial} 或 {none}），不返回说明或附加文本。
"""

RELEVANCY_PROMPT_TEMPLATE = """
评估AI回答与用户查询的相关性。
用户查询: {question}
AI回答: {response}

相关性衡量回答在多大程度上解决了用户的问题。

指示：
- 评分必须严格使用以下值：
    * {full} = 完全相关，直接回答了问题
    * {partial} = 部分相关，回答了某些方面
    * {none} = 不相关，未能回答问题
- 仅返回数值评分（{full}、{partial} 或 {none}），不加任何解释或额外文字。
"""

def evaluate_response(question, response, true_answer):
    """
    评估AI生成的回答的质量，基于忠实度和相关性。

    参数:
    question (str): 用户的原始问题。
    response (str): 被评估的AI生成的回答。
    true_answer (str): 作为真实答案的正确答案。

    返回:
    Tuple[float, float]: 包含 (忠实度评分, 相关性评分) 的元组。
                         每个评分可以是：1.0（完全），0.5（部分），或 0.0（无）。
    """
    # 格式化评估提示
    faithfulness_prompt = FAITHFULNESS_PROMPT_TEMPLATE.format(
        question=question,
        response=response,
        true_answer=true_answer,
        full=SCORE_FULL,
        partial=SCORE_PARTIAL,
        none=SCORE_NONE
    )

    relevancy_prompt = RELEVANCY_PROMPT_TEMPLATE.format(
        question=question,
        response=response,
        full=SCORE_FULL,
        partial=SCORE_PARTIAL,
        none=SCORE_NONE
    )

    # 请求模型评估忠实度
    faithfulness_response = client.chat.completions.create(
        model="meta-llama/Llama-3.2-3B-Instruct",
        temperature=0,
        messages=[
            {"role": "system", "content": "你是一个客观的评估者。只返回数字评分。"},
            {"role": "user", "content": faithfulness_prompt}
        ]
    )

    # 请求模型评估相关性
    relevancy_response = client.chat.completions.create(
        model="meta-llama/Llama-3.2-3B-Instruct",
        temperature=0,
        messages=[
            {"role": "system", "content": "你是一个客观的评估者。只返回数字评分。"},
            {"role": "user", "content": relevancy_prompt}
        ]
    )

    # 提取评分并处理潜在的解析错误
    try:
        faithfulness_score = float(faithfulness_response.choices[0].message.content.strip())
    except ValueError:
        print("警告: 无法解析忠实度评分，设置为 0")
        faithfulness_score = 0.0

    try:
        relevancy_score = float(relevancy_response.choices[0].message.content.strip())
    except ValueError:
        print("警告: 无法解析相关性评分，设置为 0")
        relevancy_score = 0.0

    return faithfulness_score, relevancy_score


# 第一条验证数据的真实答案
true_answer = data[3]['ideal_answer']

# 评估块大小为256和128的回答
faithfulness, relevancy = evaluate_response(query, ai_responses_dict[256], true_answer)
faithfulness2, relevancy2 = evaluate_response(query, ai_responses_dict[128], true_answer)

# 打印评估分数
print(f"忠实度评分 (块大小 256): {faithfulness}")
print(f"相关性评分 (块大小 256): {relevancy}")

print(f"\n")

print(f"忠实度评分 (块大小 128): {faithfulness2}")
print(f"相关性评分 (块大小 128): {relevancy2}")

####################################################################################################################################




# 资料来源：https://github.com/FareedKhan-dev/all-rag-techniques.git


