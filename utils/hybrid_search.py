"""
Hybrid Search 混合搜索模块
========================

本模块实现了向量搜索与传统文本搜索的高效结合，提供以下功能：

功能特性：
1. 向量索引构建与搜索（基于 FAISS + HuggingFace Embeddings）
2. BM25 传统文本搜索
3. Ensemble Retriever 组合搜索
4. 完整的 API 接口
5. 索引持久化与加载

技术栈：
- LangChain: 检索框架
- HuggingFace Embeddings: 文本向量化
- FAISS: 向量数据库
- BM25: 传统文本检索算法
"""

import os
import json
import time
import uuid
from datetime import datetime
from typing import List, Dict, Optional, Tuple
from dotenv import load_dotenv

load_dotenv()

# ==================== 配置常量 ====================
HYBRID_INDEX_DIR = "./hybrid_index"
VECTOR_INDEX_FILE = os.path.join(HYBRID_INDEX_DIR, "vector_index.faiss")
DOC_STORE_FILE = os.path.join(HYBRID_INDEX_DIR, "docstore.json")
METADATA_FILE = os.path.join(HYBRID_INDEX_DIR, "metadata.json")

# 模型配置
EMBEDDING_MODEL = os.getenv("EMBEDDING_MODEL", "sentence-transformers/all-MiniLM-L6-v2")
TOP_K_RESULTS = int(os.getenv("HYBRID_TOP_K", 5))
BM25_WEIGHT = float(os.getenv("BM25_WEIGHT", 0.3))
VECTOR_WEIGHT = float(os.getenv("VECTOR_WEIGHT", 0.7))

# ==================== 导入依赖 ====================
try:
    from langchain_community.document_loaders import TextLoader, PyPDFLoader
    from langchain_community.vectorstores import FAISS
    from langchain_community.retrievers import BM25Retriever
    
    # 尝试从不同位置导入 EnsembleRetriever
    try:
        from langchain.retrievers import EnsembleRetriever
    except ImportError:
        try:
            from langchain_community.retrievers import EnsembleRetriever
        except ImportError:
            # 如果 EnsembleRetriever 不可用，使用自定义实现
            EnsembleRetriever = None
    
    from langchain_huggingface import HuggingFaceEmbeddings
    from langchain_text_splitters import RecursiveCharacterTextSplitter
    
    # 尝试从不同位置导入 Document
    try:
        from langchain.schema import Document
    except ImportError:
        try:
            from langchain_core.documents import Document
        except ImportError:
            try:
                from langchain_core.documents.base import Document
            except ImportError:
                # 如果 Document 不可用，定义一个简单的替代类
                class Document:
                    def __init__(self, page_content, metadata=None):
                        self.page_content = page_content
                        self.metadata = metadata or {}
    
    LANGCHAIN_AVAILABLE = True
except ImportError as e:
    print(f"⚠️ LangChain 相关库未安装: {e}")
    LANGCHAIN_AVAILABLE = False


class HybridSearchEngine:
    """
    Hybrid Search 引擎类
    
    集成向量搜索和 BM25 搜索，通过 EnsembleRetriever 组合结果
    """
    
    def __init__(self):
        """初始化 Hybrid Search 引擎"""
        self.embeddings = None
        self.vector_store = None
        self.bm25_retriever = None
        self.ensemble_retriever = None
        self.documents = []
        self.metadata = {}
        
        # 初始化目录
        self._init_directory()
        
        # 尝试加载预训练模型和索引
        if LANGCHAIN_AVAILABLE:
            self._init_embeddings()
    
    def _init_directory(self):
        """确保索引目录存在"""
        os.makedirs(HYBRID_INDEX_DIR, exist_ok=True)
    
    def _init_embeddings(self):
        """初始化 Embeddings 模型"""
        try:
            print(f"📥 正在加载 Embedding 模型: {EMBEDDING_MODEL}")
            self.embeddings = HuggingFaceEmbeddings(
                model_name=EMBEDDING_MODEL,
                model_kwargs={"device": "cpu"},
                encode_kwargs={"normalize_embeddings": True}
            )
            print("✅ Embedding 模型加载成功")
        except Exception as e:
            print(f"❌ 加载 Embedding 模型失败: {e}")
            self.embeddings = None
    
    def add_document(self, content: str, metadata: Dict = None) -> str:
        """
        添加文档到索引
        
        Args:
            content: 文档内容
            metadata: 文档元数据（可选）
        
        Returns:
            doc_id: 文档唯一标识
        """
        if not LANGCHAIN_AVAILABLE:
            return None
        
        doc_id = str(uuid.uuid4())[:8]
        
        # 创建 Document 对象
        doc_metadata = {
            "doc_id": doc_id,
            "created_at": datetime.now().isoformat(),
            "content_length": len(content)
        }
        if metadata:
            doc_metadata.update(metadata)
        
        document = Document(
            page_content=content,
            metadata=doc_metadata
        )
        
        self.documents.append(document)
        self.metadata[doc_id] = doc_metadata
        
        return doc_id
    
    def add_document_from_file(self, file_path: str, metadata: Dict = None) -> Dict:
        """
        从文件添加文档
        
        Args:
            file_path: 文件路径
            metadata: 文档元数据（可选）
        
        Returns:
            结果字典
        """
        if not LANGCHAIN_AVAILABLE:
            return {"success": False, "message": "LangChain 未安装"}
        
        try:
            # 根据文件类型选择加载器
            ext = file_path.split(".")[-1].lower()
            
            if ext == "pdf":
                loader = PyPDFLoader(file_path)
            elif ext in ["txt", "md"]:
                loader = TextLoader(file_path, encoding="utf-8")
            else:
                return {"success": False, "message": f"不支持的文件格式: {ext}"}
            
            # 加载文档
            docs = loader.load()
            
            # 添加每个页面
            doc_ids = []
            for i, doc in enumerate(docs):
                doc_metadata = {
                    "source": file_path,
                    "page": i + 1,
                    "file_name": os.path.basename(file_path)
                }
                if metadata:
                    doc_metadata.update(metadata)
                
                doc_id = self.add_document(doc.page_content, doc_metadata)
                if doc_id:
                    doc_ids.append(doc_id)
            
            return {
                "success": True,
                "message": f"成功添加 {len(doc_ids)} 个文档片段",
                "doc_ids": doc_ids
            }
        
        except Exception as e:
            return {"success": False, "message": f"文件处理失败: {str(e)}"}
    
    def build_index(self, chunk_size: int = 500, chunk_overlap: int = 50) -> Dict:
        """
        构建混合索引（向量索引 + BM25 索引）
        
        Args:
            chunk_size: 文本切分大小
            chunk_overlap: 切分重叠大小
        
        Returns:
            构建结果
        """
        if not LANGCHAIN_AVAILABLE or not self.embeddings:
            return {"success": False, "message": "依赖未就绪"}
        
        if not self.documents:
            return {"success": False, "message": "没有可索引的文档"}
        
        start_time = time.time()
        print("🔨 正在构建混合索引...")
        
        try:
            # 文本切分
            text_splitter = RecursiveCharacterTextSplitter(
                chunk_size=chunk_size,
                chunk_overlap=chunk_overlap,
                length_function=len,
                separators=["\n\n", "\n", "。", "！", "？", "；", "，", " "]
            )
            
            # 切分所有文档
            split_docs = text_splitter.split_documents(self.documents)
            print(f"📄 文档切分完成: {len(split_docs)} 个片段")
            
            # 构建向量索引
            print("🧠 构建向量索引...")
            self.vector_store = FAISS.from_documents(split_docs, self.embeddings)
            
            # 构建 BM25 索引
            print("🔤 构建 BM25 索引...")
            self.bm25_retriever = BM25Retriever.from_documents(split_docs)
            self.bm25_retriever.k = TOP_K_RESULTS
            
            # 创建 Ensemble Retriever 或使用自定义实现
            if EnsembleRetriever is not None:
                self.ensemble_retriever = EnsembleRetriever(
                    retrievers=[self.bm25_retriever, self.vector_store.as_retriever(search_kwargs={"k": TOP_K_RESULTS})],
                    weights=[BM25_WEIGHT, VECTOR_WEIGHT]
                )
            else:
                # 使用自定义混合搜索实现
                print("⚠️ EnsembleRetriever 不可用，使用自定义混合搜索")
                self.ensemble_retriever = None
            
            build_time = time.time() - start_time
            
            return {
                "success": True,
                "message": f"索引构建完成",
                "total_docs": len(self.documents),
                "split_chunks": len(split_docs),
                "build_time": f"{build_time:.2f}秒",
                "bm25_weight": BM25_WEIGHT,
                "vector_weight": VECTOR_WEIGHT
            }
        
        except Exception as e:
            return {"success": False, "message": f"索引构建失败: {str(e)}"}
    
    def save_index(self) -> Dict:
        """
        保存索引到磁盘
        
        Returns:
            保存结果
        """
        if not self.vector_store:
            return {"success": False, "message": "没有可保存的索引"}
        
        try:
            # 保存向量索引
            self.vector_store.save_local(HYBRID_INDEX_DIR)
            
            # 保存元数据
            with open(METADATA_FILE, "w", encoding="utf-8") as f:
                json.dump(self.metadata, f, ensure_ascii=False, indent=2)
            
            # 保存原始文档（用于 BM25 重建）
            doc_store = {
                "documents": [
                    {"page_content": doc.page_content, "metadata": doc.metadata}
                    for doc in self.documents
                ],
                "created_at": datetime.now().isoformat()
            }
            with open(DOC_STORE_FILE, "w", encoding="utf-8") as f:
                json.dump(doc_store, f, ensure_ascii=False, indent=2)
            
            return {"success": True, "message": "索引保存成功"}
        
        except Exception as e:
            return {"success": False, "message": f"保存失败: {str(e)}"}
    
    def load_index(self) -> Dict:
        """
        从磁盘加载索引
        
        Returns:
            加载结果
        """
        if not LANGCHAIN_AVAILABLE or not self.embeddings:
            return {"success": False, "message": "依赖未就绪"}
        
        try:
            # 检查索引文件是否存在
            if not os.path.exists(VECTOR_INDEX_FILE.replace(".faiss", "") + ".faiss"):
                return {"success": False, "message": "索引文件不存在"}
            
            # 加载向量索引
            print("📥 加载向量索引...")
            self.vector_store = FAISS.load_local(
                HYBRID_INDEX_DIR,
                self.embeddings,
                allow_dangerous_deserialization=True
            )
            
            # 加载文档存储（用于 BM25）
            if os.path.exists(DOC_STORE_FILE):
                with open(DOC_STORE_FILE, "r", encoding="utf-8") as f:
                    doc_store = json.load(f)
                
                # 重建 Document 对象列表
                self.documents = [
                    Document(page_content=doc["page_content"], metadata=doc["metadata"])
                    for doc in doc_store["documents"]
                ]
                
                # 重建 BM25 索引
                print("📥 重建 BM25 索引...")
                self.bm25_retriever = BM25Retriever.from_documents(self.documents)
                self.bm25_retriever.k = TOP_K_RESULTS
                
                # 创建 Ensemble Retriever 或使用自定义实现
                if EnsembleRetriever is not None:
                    self.ensemble_retriever = EnsembleRetriever(
                        retrievers=[self.bm25_retriever, self.vector_store.as_retriever(search_kwargs={"k": TOP_K_RESULTS})],
                        weights=[BM25_WEIGHT, VECTOR_WEIGHT]
                    )
                else:
                    self.ensemble_retriever = None
            
            # 加载元数据
            if os.path.exists(METADATA_FILE):
                with open(METADATA_FILE, "r", encoding="utf-8") as f:
                    self.metadata = json.load(f)
            
            return {
                "success": True,
                "message": "索引加载成功",
                "document_count": len(self.documents)
            }
        
        except Exception as e:
            return {"success": False, "message": f"加载失败: {str(e)}"}
    
    def search(self, query: str, top_k: int = None, mode: str = "hybrid") -> List[Dict]:
        """
        执行搜索
        
        Args:
            query: 搜索查询
            top_k: 返回结果数量（默认使用配置值）
            mode: 搜索模式 (hybrid/bm25/vector)
        
        Returns:
            搜索结果列表
        """
        if not LANGCHAIN_AVAILABLE:
            return []
        
        k = top_k or TOP_K_RESULTS
        
        try:
            if mode == "hybrid":
                # 混合搜索（默认）
                if self.ensemble_retriever is not None:
                    results = self.ensemble_retriever.get_relevant_documents(query)
                else:
                    # 使用自定义混合搜索实现
                    results = self._custom_hybrid_search(query, k)
                
            elif mode == "bm25":
                # BM25 搜索
                if not self.bm25_retriever:
                    return []
                
                # 尝试不同的 API 调用方式
                try:
                    self.bm25_retriever.k = k
                    results = self.bm25_retriever.get_relevant_documents(query)
                except AttributeError:
                    # 新版本 BM25Retriever 使用 invoke 方法
                    try:
                        results = self.bm25_retriever.invoke(query)
                    except Exception as e:
                        print(f"BM25 搜索失败: {e}")
                        return []
                
            elif mode == "vector":
                # 向量搜索
                if not self.vector_store:
                    return []
                
                results = self.vector_store.similarity_search(query, k=k)
            
            else:
                return []
            
            # 格式化结果
            formatted_results = []
            for doc in results:
                formatted_results.append({
                    "content": doc.page_content,
                    "metadata": doc.metadata,
                    "score": doc.metadata.get("score", 0.0)
                })
            
            return formatted_results
        
        except Exception as e:
            print(f"搜索失败: {e}")
            return []
    
    def search_with_scores(self, query: str, top_k: int = None) -> List[Dict]:
        """
        执行搜索并返回相似度分数
        
        Args:
            query: 搜索查询
            top_k: 返回结果数量
        
        Returns:
            搜索结果列表（含分数）
        """
        if not LANGCHAIN_AVAILABLE or not self.vector_store:
            return []
        
        k = top_k or TOP_K_RESULTS
        
        try:
            # 使用向量搜索获取分数
            results = self.vector_store.similarity_search_with_score(query, k=k)
            
            formatted_results = []
            for doc, score in results:
                formatted_results.append({
                    "content": doc.page_content,
                    "metadata": doc.metadata,
                    "similarity_score": score
                })
            
            return formatted_results
        
        except Exception as e:
            print(f"搜索失败: {e}")
            return []
    
    def _custom_hybrid_search(self, query: str, top_k: int) -> List:
        """
        自定义混合搜索实现（当 EnsembleRetriever 不可用时使用）
        
        Args:
            query: 搜索查询
            top_k: 返回结果数量
        
        Returns:
            搜索结果列表（Document 对象）
        """
        if not self.bm25_retriever or not self.vector_store:
            return []
        
        # 分别执行两种搜索
        # BM25 搜索
        try:
            self.bm25_retriever.k = top_k
            bm25_results = self.bm25_retriever.get_relevant_documents(query)
        except AttributeError:
            try:
                bm25_results = self.bm25_retriever.invoke(query)
            except Exception as e:
                print(f"BM25 搜索失败，仅使用向量搜索: {e}")
                bm25_results = []
        
        vector_results = self.vector_store.similarity_search(query, k=top_k)
        
        # 合并结果并去重
        results_dict = {}
        
        # 添加 BM25 结果
        for i, doc in enumerate(bm25_results):
            key = doc.page_content[:200]  # 使用内容片段作为唯一键
            if key not in results_dict:
                results_dict[key] = {
                    "doc": doc,
                    "score": (top_k - i) * BM25_WEIGHT  # 位置分数
                }
            else:
                results_dict[key]["score"] += (top_k - i) * BM25_WEIGHT
        
        # 添加向量搜索结果
        for i, doc in enumerate(vector_results):
            key = doc.page_content[:200]
            if key not in results_dict:
                results_dict[key] = {
                    "doc": doc,
                    "score": (top_k - i) * VECTOR_WEIGHT
                }
            else:
                results_dict[key]["score"] += (top_k - i) * VECTOR_WEIGHT
        
        # 按综合分数排序
        sorted_results = sorted(
            results_dict.values(),
            key=lambda x: x["score"],
            reverse=True
        )
        
        # 返回前 top_k 个结果
        return [item["doc"] for item in sorted_results[:top_k]]
    
    def get_stats(self) -> Dict:
        """获取引擎统计信息"""
        # 索引就绪的条件：有文档且至少一种搜索方式可用
        index_ready = len(self.documents) > 0 and (
            self.ensemble_retriever is not None or 
            (self.bm25_retriever is not None and self.vector_store is not None)
        )
        return {
            "document_count": len(self.documents),
            "index_ready": index_ready,
            "embeddings_available": self.embeddings is not None,
            "bm25_weight": BM25_WEIGHT,
            "vector_weight": VECTOR_WEIGHT,
            "top_k": TOP_K_RESULTS,
            "embedding_model": EMBEDDING_MODEL
        }
    
    def clear_index(self) -> Dict:
        """清空索引"""
        self.documents = []
        self.metadata = {}
        self.vector_store = None
        self.bm25_retriever = None
        self.ensemble_retriever = None
        
        # 删除索引文件
        for f in [VECTOR_INDEX_FILE, DOC_STORE_FILE, METADATA_FILE]:
            if os.path.exists(f):
                os.remove(f)
        # 删除 FAISS 索引文件（有多个文件）
        for f in os.listdir(HYBRID_INDEX_DIR):
            if f.startswith("index_"):
                os.remove(os.path.join(HYBRID_INDEX_DIR, f))
        
        return {"success": True, "message": "索引已清空"}


# ==================== 全局实例 ====================
_hybrid_engine = None


def get_hybrid_search_engine() -> HybridSearchEngine:
    """获取 Hybrid Search 引擎单例"""
    global _hybrid_engine
    if _hybrid_engine is None:
        _hybrid_engine = HybridSearchEngine()
    return _hybrid_engine


# ==================== API 接口函数 ====================
def init_hybrid_search() -> Dict:
    """初始化 Hybrid Search 引擎"""
    engine = get_hybrid_search_engine()
    
    # 尝试加载已保存的索引
    load_result = engine.load_index()
    if load_result["success"]:
        return load_result
    
    # 如果没有已保存的索引，返回初始化成功
    return {"success": True, "message": "Hybrid Search 引擎初始化完成（无预加载索引）"}


def add_document(content: str, metadata: Dict = None) -> Dict:
    """添加文档"""
    engine = get_hybrid_search_engine()
    doc_id = engine.add_document(content, metadata)
    
    if doc_id:
        return {
            "success": True,
            "message": "文档添加成功",
            "doc_id": doc_id
        }
    return {"success": False, "message": "文档添加失败"}


def add_document_from_file(file_path: str, metadata: Dict = None) -> Dict:
    """从文件添加文档"""
    engine = get_hybrid_search_engine()
    return engine.add_document_from_file(file_path, metadata)


def build_hybrid_index() -> Dict:
    """构建混合索引"""
    engine = get_hybrid_search_engine()
    result = engine.build_index()
    
    # 如果构建成功，自动保存索引
    if result["success"]:
        save_result = engine.save_index()
        if save_result["success"]:
            result["save_message"] = save_result["message"]
    
    return result


def hybrid_search(query: str, top_k: int = 5, mode: str = "hybrid") -> Dict:
    """执行混合搜索"""
    engine = get_hybrid_search_engine()
    
    # 如果索引未就绪，尝试加载
    stats = engine.get_stats()
    if not stats["index_ready"]:
        load_result = engine.load_index()
        if not load_result["success"]:
            return {"success": False, "message": "索引未就绪，请先构建索引"}
    
    results = engine.search(query, top_k, mode)
    
    return {
        "success": True,
        "query": query,
        "mode": mode,
        "results": results,
        "total_results": len(results)
    }


def get_engine_stats() -> Dict:
    """获取引擎统计信息"""
    engine = get_hybrid_search_engine()
    return engine.get_stats()


def clear_hybrid_index() -> Dict:
    """清空索引"""
    engine = get_hybrid_search_engine()
    return engine.clear_index()


# ==================== 便捷函数 ====================
def search_knowledge_hybrid(query: str, top_k: int = 5) -> List[Dict]:
    """
    便捷搜索函数 - 搜索知识库
    
    Args:
        query: 搜索查询
        top_k: 返回结果数量
    
    Returns:
        搜索结果列表
    """
    result = hybrid_search(query, top_k, mode="hybrid")
    return result.get("results", [])


def search_knowledge_bm25(query: str, top_k: int = 5) -> List[Dict]:
    """
    便捷搜索函数 - 使用 BM25 搜索
    
    Args:
        query: 搜索查询
        top_k: 返回结果数量
    
    Returns:
        搜索结果列表
    """
    result = hybrid_search(query, top_k, mode="bm25")
    return result.get("results", [])


def search_knowledge_vector(query: str, top_k: int = 5) -> List[Dict]:
    """
    便捷搜索函数 - 使用向量搜索
    
    Args:
        query: 搜索查询
        top_k: 返回结果数量
    
    Returns:
        搜索结果列表
    """
    result = hybrid_search(query, top_k, mode="vector")
    return result.get("results", [])
