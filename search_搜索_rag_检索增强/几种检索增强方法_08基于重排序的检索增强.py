
# 检索增强生成 （RAG） 是一种将信息检索与生成模型相结合的混合方法。它通过整合外部知识来增强语言模型的性能，从而提高准确性和事实正确性。

####################################################################################################################################
# 方法8：
# 基于重排序的检索增强
# 初始检索：使用基本相似度搜索的第一步（准确性较低但速度较快）
# 文档评分：评估每个检索到的文档与查询的相关性
# 重新排序：根据相关性评分对文档进行排序
# 选择：仅使用最相关的文档进行响应生成

# 为了演示重排序如何与检索集成，我们实现一个简单的向量存储。
class SimpleVectorStore:
    """
    使用NumPy实现的简单向量存储。
    """
    def __init__(self):
        """
        初始化向量存储。
        """
        self.vectors = []  # 存储嵌入向量的列表
        self.texts = []  # 存储原始文本的列表
        self.metadata = []  # 存储每个文本的元数据的列表

    def add_item(self, text, embedding, metadata=None):
        """
        将项目添加到向量存储中。

        Args:
            text (str): 原始文本。
            embedding (List[float]): 嵌入向量。
            metadata (dict, optional): 额外的元数据。
        """
        self.vectors.append(np.array(embedding))  # 将嵌入向量转换为numpy数组并添加到vectors列表中
        self.texts.append(text)  # 将原始文本添加到texts列表中
        self.metadata.append(metadata or {})  # 将元数据添加到metadata列表中，如果为None则使用空字典

    def similarity_search(self, query_embedding, k=5):
        """
        找到与查询嵌入最相似的项目。

        Args:
            query_embedding (List[float]): 查询嵌入向量。
            k (int): 要返回的结果数。

        Returns:
            List[Dict]: 最相似的前k个项目，包含文本和元数据。
        """
        if not self.vectors:
            return []  # 如果没有存储向量，返回空列表

        # 将查询嵌入转换为numpy数组
        query_vector = np.array(query_embedding)

        # 计算相似性使用余弦相似度
        similarities = []
        for i, vector in enumerate(self.vectors):
            # 计算查询向量和存储向量之间的余弦相似度
            similarity = np.dot(query_vector, vector) / (np.linalg.norm(query_vector) * np.linalg.norm(vector))
            similarities.append((i, similarity))  # 附加索引和相似度分数

        # 按相似度降序排序
        similarities.sort(key=lambda x: x[1], reverse=True)

        # 返回前k个结果
        results = []
        for i in range(min(k, len(similarities))):
            idx, score = similarities[i]
            results.append({
                "text": self.texts[idx],  # 添加对应的文本
                "metadata": self.metadata[idx],  # 添加对应的元数据
                "similarity": score  # 添加相似度分数
            })

        return results  # 返回最相似项目的列表


# 使用OpenAI API实现基于LLM的重排序功能。
def rerank_with_llm(query, results, top_n=3, model="meta-llama/Llama-3.2-3B-Instruct"):
    """
    使用LLM相关性评分对搜索结果进行重排序。

    Args:
        query (str): 用户查询
        results (List[Dict]): 初始搜索结果
        top_n (int): 重排序后要返回的结果数
        model (str): 要使用的模型

    Returns:
        List[Dict]: 重排序后的结果
    """
    print(f"对{len(results)}个文档进行重排序...")  # 打印要重排序的文档数

    scored_results = []  # 初始化一个空列表以存储评分结果

    # 定义LLM的系统提示
    system_prompt = """你是一位擅长评估文档与查询相关性的专家。
    你的任务是根据给定的查询对文档进行评分，评分范围从0到10。

    指南：
    - 评分0-2：文档完全不相关
    - 评分3-5：文档包含一些相关信息，但没有直接回答查询
    - 评分6-8：文档相关且部分回答查询
    - 评分9-10：文档高度相关且直接回答查询

    你必须只返回一个介于0到10之间的整数评分。不要包含任何其他文本。"""

    # 遍历每个结果
    for i, result in enumerate(results):
        # 每5个文档显示进度
        if i % 5 == 0:
            print(f"正在评分第{i + 1}/{len(results)}个文档...")

        # 定义LLM的用户提示
        user_prompt = f"""查询：{query}

文档：
{result['text']}

请对这个文档与查询的相关性进行评分（评分范围0到10）："""

        # 获取LLM的响应
        response = client.chat.completions.create(
            model=model,
            temperature=0,
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt}
            ]
        )

        # 从LLM响应中提取评分
        score_text = response.choices[0].message.content.strip()

        # 使用正则表达式提取数值评分
        score_match = re.search(r'\b(10|[0-9])\b', score_text)
        if score_match:
            score = float(score_match.group(1))
        else:
            # 如果评分提取失败，使用相似度评分作为备用
            print(f"警告：无法从响应中提取评分：'{score_text}'，使用相似度评分代替")
            score = result["similarity"] * 10

        # 将评分结果附加到列表中
        scored_results.append({
            "text": result["text"],
            "metadata": result["metadata"],
            "similarity": result["similarity"],
            "relevance_score": score
        })

    # 按相关性评分降序排序结果
    reranked_results = sorted(scored_results, key=lambda x: x["relevance_score"], reverse=True)

    # 返回前top_n个结果
    return reranked_results[:top_n]


# 简单的基于关键词的重排序
def rerank_with_keywords(query, results, top_n=3):
    """
    基于关键词匹配和位置的简单替代重排序方法。

    Args:
        query (str): 用户查询
        results (List[Dict]): 初始搜索结果
        top_n (int): 重排序后要返回的结果数

    Returns:
        List[Dict]: 重排序后的结果
    """
    # 从查询中提取重要关键词
    keywords = [word.lower() for word in query.split() if len(word) > 3]

    scored_results = []  # 初始化一个列表以存储评分结果

    for result in results:
        document_text = result["text"].lower()  # 将文档文本转换为小写

        # 基评分从向量相似度开始
        base_score = result["similarity"] * 0.5

        # 初始化关键词评分
        keyword_score = 0
        for keyword in keywords:
            if keyword in document_text:
                # 每找到一个关键词加分
                keyword_score += 0.1

                # 如果关键词出现在文本开头附近，额外加分
                first_position = document_text.find(keyword)
                if first_position < len(document_text) / 4:  # 文本的前四分之一
                    keyword_score += 0.1

                # 根据关键词频率加分
                frequency = document_text.count(keyword)
                keyword_score += min(0.05 * frequency, 0.2)  # 最多加0.2分

        # 计算最终评分，结合基评分和关键词评分
        final_score = base_score + keyword_score

        # 将评分结果附加到列表中
        scored_results.append({
            "text": result["text"],
            "metadata": result["metadata"],
            "similarity": result["similarity"],
            "relevance_score": final_score
        })

    # 按最终相关性评分降序排序结果
    reranked_results = sorted(scored_results, key=lambda x: x["relevance_score"], reverse=True)

    # 返回前top_n个结果
    return reranked_results[:top_n]


# 创建一个完整的RAG管道。
def rag_with_reranking(query, vector_store, reranking_method="llm", top_n=3, model="meta-llama/Llama-3.2-3B-Instruct"):
    """
    包含重排序的完整RAG管道。

    Args:
        query (str): 用户查询
        vector_store (SimpleVectorStore): 向量存储
        reranking_method (str): 重排序方法（'llm'或'keywords'）
        top_n (int): 重排序后要返回的结果数
        model (str): 要使用的模型

    Returns:
        Dict: 包含查询、上下文和响应的结果
    """
    # 创建查询嵌入
    query_embedding = create_embeddings(query)

    # 初始检索（获取比需要更多的结果用于重排序）
    initial_results = vector_store.similarity_search(query_embedding, k=10)

    # 应用重排序
    if reranking_method == "llm":
        reranked_results = rerank_with_llm(query, initial_results, top_n=top_n)
    elif reranking_method == "keywords":
        reranked_results = rerank_with_keywords(query, initial_results, top_n=top_n)
    else:
        # 没有重排序，直接使用初始检索的前top_n个结果
        reranked_results = initial_results[:top_n]

    # 从重排序结果中组合上下文
    context = "\n\n===\n\n".join([result["text"] for result in reranked_results])

    # 根据上下文生成响应
    response = generate_response(query, context, model)

    return {
        "query": query,
        "reranking_method": reranking_method,
        "initial_results": initial_results[:top_n],
        "reranked_results": reranked_results,
        "context": context,
        "response": response
    }
# 评估重排序质量

# 从JSON文件中加载验证数据
with open('data/val.json') as f:
    data = json.load(f)

# 从验证数据中提取第一个查询
query = data[0]['question']

# 从验证数据中提取参考答案
reference_answer = data[0]['ideal_answer']

# pdf_path
pdf_path = "data/AI_Information.pdf"

# 处理文档
vector_store = process_document(pdf_path)

# 示例查询
query = "AI是否有潜力改变我们的生活和工作方式？"

# 比较不同的方法
print("比较检索方法...")

# 1. 标准检索（无重排序）
print("\n=== 标准检索 ===")
standard_results = rag_with_reranking(query, vector_store, reranking_method="none")
print(f"\n查询：{query}")
print(f"\n响应：\n{standard_results['response']}")

# 2. 基于LLM的重排序
print("\n=== 基于LLM的重排序 ===")
llm_results = rag_with_reranking(query, vector_store, reranking_method="llm")
print(f"\n查询：{query}")
print(f"\n响应：\n{llm_results['response']}")

# 3. 基于关键词的重排序
print("\n=== 基于关键词的重排序 ===")
keyword_results = rag_with_reranking(query, vector_store, reranking_method="keywords")
print(f"\n查询：{query}")
print(f"\n响应：\n{keyword_results['response']}")



####################################################################################################################################




# 资料来源：https://github.com/FareedKhan-dev/all-rag-techniques.git


