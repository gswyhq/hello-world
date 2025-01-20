#!/usr/bin/env python
# coding=utf-8

import chromadb
from chromadb.api.types import normalize_embeddings
# 连接 chroma 数据库
client = chromadb.HttpClient(host="30.171.80.45", port=8899)

import os
from sentence_transformers import SentenceTransformer
USERNAME = os.getenv("USERNAME")
sentences_1 = ["样例数据-1", "样例数据-2"]
sentences_2 = ["样例数据-3", "样例数据-4"]
model_dir = rf"D:\Users\{USERNAME}\data\bge-small-zh-v1.5"
model = SentenceTransformer(model_dir)

class EmbeddingFunction():
    def __call__(self, input):
        return normalize_embeddings(model.encode(input, normalize_embeddings=True))
embedding_fn = EmbeddingFunction()

# 创建 collection,并指定向量函数
collection = client.create_collection("all-my-documents", embedding_function=embedding_fn)

# 向 collection 添加文档
collection.add(
    documents=["静夜思", "床前明月光","离离原上草", "一岁一枯荣", "桃花潭水深千尺", "不及汪伦送我情"],
    metadatas=[{"source": "notion"}, {"source": "google-docs"}, {"source": "notion"}, {"source": "google-docs"},{"source": "notion"}, {"source": "google-docs"}        ], # filter on these!
    ids=["doc1", "doc2", "doc3", "doc4", "doc5", "doc6"], # unique for each doc
    # embeddings = [[1.2, 2.1, ...], [1.2, 2.1, ...]]
)

# 查询
results = collection.query(
    query_texts=["岁月"],
    n_results=2,
    # where={"metadata_field": "is_equal_to_this"}, # optional filter
    # where_document={"$contains":"search_string"}  # optional filter
)

def main():
    pass


if __name__ == "__main__":
    main()
