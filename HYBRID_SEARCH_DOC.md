# Hybrid Search 集成文档

## 概述

Hybrid Search 是一种结合向量搜索和传统文本搜索的检索技术，能够同时利用语义理解和关键词匹配的优势，提供更准确、更全面的搜索结果。

本项目集成了基于 LangChain 的 Hybrid Search 功能，支持以下搜索模式：
- **混合搜索（Hybrid）**: 组合 BM25 和向量搜索的结果
- **BM25 搜索**: 传统关键词匹配算法
- **向量搜索**: 基于语义相似度的搜索

## 技术栈

| 组件 | 说明 |
|------|------|
| **LangChain** | 检索框架，提供统一的检索接口 |
| **HuggingFace Embeddings** | 文本向量化模型 |
| **FAISS** | Facebook 开源的向量数据库 |
| **BM25Retriever** | BM25 文本检索算法 |
| **EnsembleRetriever** | 组合多个检索器的结果 |

## 快速开始

### 安装依赖

```bash
pip install faiss-cpu langchain langchain-community langchain-text-splitters
```

### 基础使用

```python
from utils.hybrid_search import (
    init_hybrid_search,
    add_document,
    build_hybrid_index,
    hybrid_search
)

# 1. 初始化引擎
init_hybrid_search()

# 2. 添加文档
add_document(
    content="Python 是一种高级编程语言...",
    metadata={"category": "编程语言", "source": "Python 文档"}
)

# 3. 构建索引
build_hybrid_index()

# 4. 执行搜索
results = hybrid_search("Python 机器学习", top_k=5, mode="hybrid")
```

## API 接口

### 1. 引擎管理

#### `init_hybrid_search()`
初始化 Hybrid Search 引擎，尝试加载已保存的索引。

**返回值**:
```python
{
    "success": bool,
    "message": str,
    "document_count": int (可选)
}
```

#### `get_engine_stats()`
获取引擎统计信息。

**返回值**:
```python
{
    "document_count": int,      # 当前文档数量
    "index_ready": bool,        # 索引是否就绪
    "embeddings_available": bool, # Embeddings 模型是否可用
    "bm25_weight": float,       # BM25 权重
    "vector_weight": float,     # 向量搜索权重
    "top_k": int,              # 默认返回结果数
    "embedding_model": str      # 使用的 Embedding 模型
}
```

#### `clear_hybrid_index()`
清空所有索引和文档。

**返回值**:
```python
{
    "success": bool,
    "message": str
}
```

### 2. 文档管理

#### `add_document(content, metadata=None)`
添加文本内容到索引。

**参数**:
- `content`: str - 文档内容
- `metadata`: dict - 文档元数据（可选）

**返回值**:
```python
{
    "success": bool,
    "message": str,
    "doc_id": str (成功时返回)
}
```

#### `add_document_from_file(file_path, metadata=None)`
从文件添加文档。

**参数**:
- `file_path`: str - 文件路径（支持 PDF/TXT/MD）
- `metadata`: dict - 文档元数据（可选）

**返回值**:
```python
{
    "success": bool,
    "message": str,
    "doc_ids": list (成功时返回)
}
```

### 3. 索引构建

#### `build_hybrid_index(chunk_size=500, chunk_overlap=50)`
构建混合索引。

**参数**:
- `chunk_size`: int - 文本切分大小（默认 500）
- `chunk_overlap`: int - 切分重叠大小（默认 50）

**返回值**:
```python
{
    "success": bool,
    "message": str,
    "total_docs": int,          # 原始文档数
    "split_chunks": int,        # 切分后的片段数
    "build_time": str,          # 构建时间
    "bm25_weight": float,       # BM25 权重
    "vector_weight": float,     # 向量搜索权重
    "save_message": str (可选)   # 保存消息
}
```

### 4. 搜索接口

#### `hybrid_search(query, top_k=5, mode="hybrid")`
执行混合搜索。

**参数**:
- `query`: str - 搜索查询
- `top_k`: int - 返回结果数量（默认 5）
- `mode`: str - 搜索模式（hybrid/bm25/vector）

**返回值**:
```python
{
    "success": bool,
    "query": str,
    "mode": str,
    "results": list,            # 搜索结果列表
    "total_results": int        # 结果总数
}
```

**结果格式**:
```python
{
    "content": str,             # 匹配的内容片段
    "metadata": dict,           # 文档元数据
    "score": float              # 相似度分数
}
```

### 5. 便捷函数

#### `search_knowledge_hybrid(query, top_k=5)`
执行混合搜索（便捷函数）。

#### `search_knowledge_bm25(query, top_k=5)`
执行 BM25 搜索（便捷函数）。

#### `search_knowledge_vector(query, top_k=5)`
执行向量搜索（便捷函数）。

## 配置说明

### 环境变量

| 变量名 | 默认值 | 说明 |
|--------|--------|------|
| `EMBEDDING_MODEL` | sentence-transformers/all-MiniLM-L6-v2 | Embedding 模型名称 |
| `HYBRID_TOP_K` | 5 | 默认返回结果数 |
| `BM25_WEIGHT` | 0.3 | BM25 搜索权重 |
| `VECTOR_WEIGHT` | 0.7 | 向量搜索权重 |

### 权重配置

混合搜索使用加权融合策略：
- BM25_WEIGHT + VECTOR_WEIGHT = 1.0
- 增大 BM25_WEIGHT 有利于精确匹配
- 增大 VECTOR_WEIGHT 有利于语义匹配

建议配置：
- **精确搜索场景**: BM25_WEIGHT=0.6, VECTOR_WEIGHT=0.4
- **语义搜索场景**: BM25_WEIGHT=0.3, VECTOR_WEIGHT=0.7
- **平衡场景**: BM25_WEIGHT=0.5, VECTOR_WEIGHT=0.5

## 使用示例

### 示例 1: 基本搜索流程

```python
from utils.hybrid_search import *

# 初始化
init_hybrid_search()

# 添加示例文档
documents = [
    {"content": "Python 是一种高级编程语言...", "metadata": {"category": "编程"}},
    {"content": "机器学习是人工智能的分支...", "metadata": {"category": "AI"}},
    {"content": "深度学习使用神经网络...", "metadata": {"category": "AI"}}
]

for doc in documents:
    add_document(doc["content"], doc["metadata"])

# 构建索引
build_hybrid_index()

# 执行搜索
results = hybrid_search("Python 机器学习", top_k=3)

# 处理结果
for res in results["results"]:
    print(f"来源: {res['metadata'].get('category')}")
    print(f"内容: {res['content'][:100]}...")
    print("-" * 50)
```

### 示例 2: 从文件构建知识库

```python
from utils.hybrid_search import *

# 初始化
init_hybrid_search()

# 从文件添加
add_document_from_file("./knowledge_base/python_tutorial.pdf")
add_document_from_file("./knowledge_base/ml_notes.txt")

# 构建索引
result = build_hybrid_index()
print(f"索引构建完成，共 {result['split_chunks']} 个片段")

# 搜索
results = hybrid_search("神经网络", mode="vector")
```

### 示例 3: 三种搜索模式对比

```python
from utils.hybrid_search import *

query = "深度学习"

# 三种模式搜索
hybrid_results = search_knowledge_hybrid(query)
bm25_results = search_knowledge_bm25(query)
vector_results = search_knowledge_vector(query)

print(f"混合搜索: {len(hybrid_results)} 结果")
print(f"BM25 搜索: {len(bm25_results)} 结果")
print(f"向量搜索: {len(vector_results)} 结果")
```

## 索引持久化

引擎支持索引的保存和加载：

```python
from utils.hybrid_search import *

# 初始化时自动加载已保存的索引
init_hybrid_search()

# 构建索引后自动保存
build_hybrid_index()  # 内部调用 save_index()

# 索引文件位置: ./hybrid_index/
# - vector_index.faiss: FAISS 向量索引
# - docstore.json: 文档存储
# - metadata.json: 元数据
```

## 性能优化建议

### 1. 文本切分策略

- **chunk_size**: 根据文档类型调整，建议 300-1000
- **chunk_overlap**: 建议为 chunk_size 的 10%-20%

### 2. 模型选择

- **轻量级**: all-MiniLM-L6-v2（速度快，内存占用小）
- **高精度**: all-mpnet-base-v2（效果更好，速度较慢）

### 3. 硬件加速

```python
# 如果有 GPU，可以使用 GPU 加速
embeddings = HuggingFaceEmbeddings(
    model_name=EMBEDDING_MODEL,
    model_kwargs={"device": "cuda"},  # 使用 GPU
    encode_kwargs={"normalize_embeddings": True}
)
```

## 故障排除

### 常见问题

**Q1: ImportError: faiss-cpu not found**

```bash
pip install faiss-cpu
```

**Q2: 模型下载失败**

```bash
# 设置代理或手动下载模型
export TRANSFORMERS_OFFLINE=1
```

**Q3: 索引构建速度慢**

- 减少文档数量
- 使用更小的 Embedding 模型
- 增大 chunk_size

**Q4: 搜索结果不准确**

- 调整 BM25_WEIGHT 和 VECTOR_WEIGHT
- 增加训练数据量
- 调整 chunk_size

## 架构说明

```
┌─────────────────────────────────────────────────────────────────┐
│                    Hybrid Search Engine                        │
├─────────────────────────────────────────────────────────────────┤
│                                                               │
│  ┌─────────────────┐    ┌─────────────────┐                   │
│  │   文档输入层    │    │    文件输入层    │                   │
│  │  add_document   │    │add_document_from│                   │
│  └────────┬────────┘    └────────┬────────┘                   │
│           │                      │                            │
│           └───────────┬──────────┘                            │
│                       ▼                                       │
│  ┌───────────────────────────────────────┐                    │
│  │         Text Splitter                │                    │
│  │   RecursiveCharacterTextSplitter     │                    │
│  │   chunk_size=500, chunk_overlap=50   │                    │
│  └──────────────┬───────────────────────┘                    │
│                 ▼                                             │
│  ┌───────────────────────────────────────┐                    │
│  │         索引构建层                    │                    │
│  │  ┌─────────────┐  ┌─────────────┐    │                    │
│  │  │  FAISS      │  │   BM25      │    │                    │
│  │  │ Vector DB   │  │ Retriever   │    │                    │
│  │  └──────┬──────┘  └──────┬──────┘    │                    │
│  │         │               │            │                    │
│  │         └──────┬────────┘            │                    │
│  │                ▼                     │                    │
│  │   ┌───────────────────────┐         │                    │
│  │   │   EnsembleRetriever   │         │                    │
│  │   │   weights=[0.3, 0.7]  │         │                    │
│  │   └───────────┬───────────┘         │                    │
│  └───────────────┼─────────────────────┘                    │
│                 ▼                                             │
│  ┌───────────────────────────────────────┐                    │
│  │         搜索接口层                    │                    │
│  │  hybrid_search / bm25 / vector       │                    │
│  └───────────────────────────────────────┘                    │
│                                                               │
└─────────────────────────────────────────────────────────────────┘
```

## 版本历史

| 版本 | 日期 | 说明 |
|------|------|------|
| 1.0.0 | 2024-01-01 | 初始版本 |
