"""
向量索引系统
==============

实现知识库文档的文本向量化处理和向量相似度检索。

功能特性：
1. 文本分块处理（支持段落、句子、固定长度分块）
2. 向量索引构建（使用 sentence-transformers）
3. 向量相似度匹配（余弦相似度）
4. 索引持久化和增量更新
5. 混合检索（向量+关键词）
6. 性能优化和缓存机制
"""

import os
import json
import re
import hashlib
import numpy as np
from datetime import datetime
from typing import List, Dict, Any, Optional, Tuple
from dataclasses import dataclass, field
from dotenv import load_dotenv

load_dotenv()

# ==================== 配置参数 ====================
VECTOR_DIR = "./knowledge_base/vector_index"
VECTOR_INDEX_FILE = os.path.join(VECTOR_DIR, "vector_index.json")
VECTOR_DATA_FILE = os.path.join(VECTOR_DIR, "vector_data.npz")
CHUNK_SIZE = int(os.getenv("VECTOR_CHUNK_SIZE", 300))  # 分块大小（字符数）
CHUNK_OVERLAP = int(os.getenv("VECTOR_CHUNK_OVERLAP", 50))  # 分块重叠
TOP_K_RESULTS = int(os.getenv("VECTOR_TOP_K", 5))  # 默认返回结果数
VECTOR_DIMENSION = 768  # 向量维度（all-MiniLM-L6-v2 是 384，paraphrase-multilingual-MiniLM-L12-v2 是 384）


# ==================== 数据结构 ====================
@dataclass
class DocumentChunk:
    """文档分块"""
    chunk_id: str
    file_id: str
    file_name: str
    content: str
    start_index: int
    end_index: int
    vector: Optional[np.ndarray] = None
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class VectorSearchResult:
    """向量搜索结果"""
    chunk_id: str
    file_id: str
    file_name: str
    content: str
    score: float
    similarity: float
    metadata: Dict[str, Any] = field(default_factory=dict)


# ==================== 文本分块模块 ====================
class TextChunker:
    """文本分块器"""
    
    def __init__(self, chunk_size: int = 100, overlap: int = 20):
        self.chunk_size = chunk_size
        self.overlap = overlap
    
    def chunk_by_paragraph(self, text: str) -> List[str]:
        """按段落分块"""
        # 按空行分割
        paragraphs = re.split(r'\n\s*\n', text)
        
        chunks = []
        for para in paragraphs:
            para = para.strip()
            if len(para) < 30:  # 跳过过短的段落
                continue
                
            # 如果段落太长，进一步分割
            if len(para) > self.chunk_size:
                chunks.extend(self._split_long_paragraph(para))
            else:
                chunks.append(para)
        
        return chunks
    
    def chunk_by_sentence(self, text: str) -> List[str]:
        """按句子分块（中文友好）"""
        # 中文句子分割
        sentences = re.split(r'(?<=[。！？.!?])\s+', text)
        
        chunks = []
        current_chunk = ""
        
        for sentence in sentences:
            sentence = sentence.strip()
            if not sentence:
                continue
                
            if len(current_chunk) + len(sentence) < self.chunk_size:
                current_chunk += sentence
            else:
                if current_chunk:
                    chunks.append(current_chunk)
                current_chunk = sentence
        
        if current_chunk:
            chunks.append(current_chunk)
        
        return chunks
    
    def chunk_by_size(self, text: str) -> List[str]:
        """按固定大小分块（带重叠）"""
        chunks = []
        start = 0
        text_length = len(text)
        
        while start < text_length:
            end = min(start + self.chunk_size, text_length)
            chunk = text[start:end]
            
            # 避免截断单词/短语
            if end < text_length:
                # 尝试在标点处截断
                last_punct = max(
                    chunk.rfind('。'),
                    chunk.rfind('，'),
                    chunk.rfind('.'),
                    chunk.rfind(','),
                    chunk.rfind('\n'),
                    default=-1
                )
                if last_punct > len(chunk) // 2:
                    chunk = chunk[:last_punct + 1]
                    end = start + len(chunk)
            
            if chunk.strip():
                chunks.append(chunk.strip())
            
            # 移动start，考虑重叠
            start = end - self.overlap
        
        return chunks
    
    def _split_long_paragraph(self, paragraph: str) -> List[str]:
        """分割长段落"""
        chunks = []
        current = ""
        
        # 按逗号/句号拆分
        parts = re.split(r'(?<=[，。,.])', paragraph)
        
        for part in parts:
            if len(current) + len(part) < self.chunk_size:
                current += part
            else:
                if current:
                    chunks.append(current)
                current = part
        
        if current:
            chunks.append(current)
        
        return chunks
    
    def chunk_text(self, text: str, strategy: str = "hybrid") -> List[Tuple[int, int, str]]:
        """
        综合分块策略
        
        Args:
            text: 输入文本
            strategy: 分块策略 (paragraph/sentence/hybrid)
        
        Returns:
            [(start_index, end_index, content), ...]
        """
        if strategy == "paragraph":
            raw_chunks = self.chunk_by_paragraph(text)
        elif strategy == "sentence":
            raw_chunks = self.chunk_by_sentence(text)
        else:  # hybrid
            raw_chunks = self.chunk_by_paragraph(text)
            if not raw_chunks:
                raw_chunks = self.chunk_by_sentence(text)
        
        # 计算索引并返回
        chunks = []
        current_pos = 0
        
        for chunk_content in raw_chunks:
            # 在原文中查找这个chunk
            chunk_pos = text.find(chunk_content, current_pos)
            if chunk_pos >= 0:
                start = chunk_pos
                end = start + len(chunk_content)
                chunks.append((start, end, chunk_content))
                current_pos = end
            else:
                # 如果找不到，简单按位置估算
                start = current_pos
                end = min(start + len(chunk_content), len(text))
                chunks.append((start, end, chunk_content))
                current_pos = end
        
        return chunks


# ==================== 向量嵌入模块 ====================
class VectorEmbedder:
    """向量嵌入器"""
    
    def __init__(self, model_name: str = "paraphrase-multilingual-MiniLM-L12-v2"):
        self.model_name = model_name
        self.model = None
        self._initialize_model()
    
    def _initialize_model(self):
        """初始化嵌入模型"""
        try:
            from sentence_transformers import SentenceTransformer
            print(f"正在加载嵌入模型: {self.model_name}...")
            self.model = SentenceTransformer(self.model_name)
            print(f"模型加载成功! 向量维度: {self.model.get_sentence_embedding_dimension()}")
        except Exception as e:
            print(f"嵌入模型加载失败: {e}")
            print("将使用基础关键词匹配作为替代方案")
            self.model = None
    
    def is_available(self) -> bool:
        """检查嵌入模型是否可用"""
        return self.model is not None
    
    def encode(self, texts: List[str], batch_size: int = 32, show_progress: bool = False) -> np.ndarray:
        """
        将文本编码为向量
        
        Args:
            texts: 文本列表
            batch_size: 批处理大小
            show_progress: 是否显示进度
        
        Returns:
            向量矩阵 (n_samples, embedding_dim)
        """
        if not self.is_available():
            return self._encode_basic(texts)
        
        try:
            vectors = self.model.encode(
                texts,
                batch_size=batch_size,
                show_progress_bar=show_progress,
                convert_to_numpy=True,
                normalize_embeddings=True  # 归一化，方便后续计算余弦相似度
            )
            return np.array(vectors)
        except Exception as e:
            print(f"向量编码失败: {e}")
            return self._encode_basic(texts)
    
    def encode_single(self, text: str) -> np.ndarray:
        """编码单个文本"""
        if not self.is_available():
            return self._encode_basic_single(text)
        
        try:
            vector = self.model.encode(
                text,
                convert_to_numpy=True,
                normalize_embeddings=True
            )
            return vector
        except Exception as e:
            print(f"单个文本编码失败: {e}")
            return self._encode_basic_single(text)
    
    def _encode_basic(self, texts: List[str]) -> np.ndarray:
        """基础编码（TF-IDF风格的伪向量）"""
        # 简单的词频向量（作为备用方案）
        vocab = set()
        for text in texts:
            words = re.findall(r'[\u4e00-\u9fa5]+|[a-zA-Z]+', text.lower())
            vocab.update(words)
        
        vocab = list(vocab)
        word_to_idx = {word: i for i, word in enumerate(vocab)}
        
        vectors = []
        for text in texts:
            vec = np.zeros(len(vocab), dtype=np.float32)
            words = re.findall(r'[\u4e00-\u9fa5]+|[a-zA-Z]+', text.lower())
            for word in words:
                if word in word_to_idx:
                    vec[word_to_idx[word]] += 1
            
            # 归一化
            norm = np.linalg.norm(vec)
            if norm > 0:
                vec = vec / norm
            vectors.append(vec)
        
        return np.array(vectors)
    
    def _encode_basic_single(self, text: str) -> np.ndarray:
        """基础编码单个文本"""
        words = re.findall(r'[\u4e00-\u9fa5]+|[a-zA-Z]+', text.lower())
        # 使用一个简单的哈希向量
        vec = np.zeros(128, dtype=np.float32)
        for word in words:
            hash_val = int(hashlib.md5(word.encode()).hexdigest(), 16)
            idx = hash_val % 128
            vec[idx] += 1
        
        norm = np.linalg.norm(vec)
        if norm > 0:
            vec = vec / norm
        return vec


# ==================== 向量索引管理 ====================
class VectorIndexManager:
    """向量索引管理器"""
    
    def __init__(self):
        self.chunker = TextChunker()
        self.embedder = VectorEmbedder()
        self.chunks: List[DocumentChunk] = []
        self.vector_matrix: Optional[np.ndarray] = None
        self._initialize_index()
    
    def _initialize_index(self):
        """初始化向量索引目录"""
        os.makedirs(VECTOR_DIR, exist_ok=True)
    
    def build_index(self, documents: List[Tuple[str, str, str]]) -> int:
        """
        构建向量索引
        
        Args:
            documents: [(file_id, file_name, content), ...]
        
        Returns:
            成功索引的文档数量
        """
        print(f"开始构建向量索引，文档数: {len(documents)}")
        
        all_chunks = []
        
        for file_id, file_name, content in documents:
            if not content or len(content.strip()) < 50:
                print(f"跳过 {file_name}: 内容过短")
                continue
            
            # 分块
            chunk_tuples = self.chunker.chunk_text(content)
            print(f"  {file_name}: {len(chunk_tuples)} 个分块")
            
            # 创建分块对象
            for i, (start, end, chunk_content) in enumerate(chunk_tuples):
                chunk_id = f"{file_id}_chunk_{i}"
                chunk = DocumentChunk(
                    chunk_id=chunk_id,
                    file_id=file_id,
                    file_name=file_name,
                    content=chunk_content,
                    start_index=start,
                    end_index=end,
                    metadata={"chunk_index": i, "total_chunks": len(chunk_tuples)}
                )
                all_chunks.append(chunk)
        
        print(f"总分块数: {len(all_chunks)}")
        
        if not all_chunks:
            print("警告：没有有效的分块")
            return 0
        
        # 向量化
        print("开始向量化...")
        chunk_texts = [chunk.content for chunk in all_chunks]
        vectors = self.embedder.encode(chunk_texts, show_progress=True)
        
        # 保存向量
        for i, chunk in enumerate(all_chunks):
            chunk.vector = vectors[i]
        
        self.chunks = all_chunks
        self.vector_matrix = vectors
        
        # 保存索引
        self._save_index()
        
        print(f"索引构建完成！共 {len(self.chunks)} 个分块")
        return len(set(chunk.file_id for chunk in self.chunks))
    
    def add_document(self, file_id: str, file_name: str, content: str) -> bool:
        """添加单个文档到索引"""
        if not content or len(content.strip()) < 50:
            print(f"跳过 {file_name}: 内容过短")
            return False
        
        # 移除该文件的旧分块
        self.chunks = [c for c in self.chunks if c.file_id != file_id]
        
        # 分块
        chunk_tuples = self.chunker.chunk_text(content)
        print(f"添加 {file_name}: {len(chunk_tuples)} 个分块")
        
        if not chunk_tuples:
            return False
        
        # 向量化
        chunk_texts = [c[2] for c in chunk_tuples]
        vectors = self.embedder.encode(chunk_texts)
        
        # 创建分块对象
        for i, (start, end, chunk_content) in enumerate(chunk_tuples):
            chunk_id = f"{file_id}_chunk_{i}"
            chunk = DocumentChunk(
                chunk_id=chunk_id,
                file_id=file_id,
                file_name=file_name,
                content=chunk_content,
                start_index=start,
                end_index=end,
                vector=vectors[i],
                metadata={"chunk_index": i, "total_chunks": len(chunk_tuples)}
            )
            self.chunks.append(chunk)
        
        # 更新向量矩阵
        self._rebuild_vector_matrix()
        
        # 保存索引
        self._save_index()
        
        return True
    
    def remove_document(self, file_id: str) -> int:
        """从索引中移除文档"""
        original_count = len(self.chunks)
        self.chunks = [c for c in self.chunks if c.file_id != file_id]
        removed_count = original_count - len(self.chunks)
        
        if removed_count > 0:
            self._rebuild_vector_matrix()
            self._save_index()
        
        return removed_count
    
    def search(self, query: str, top_k: int = TOP_K_RESULTS, min_similarity: float = 0.1) -> List[VectorSearchResult]:
        """
        向量相似度搜索
        
        Args:
            query: 查询文本
            top_k: 返回结果数
            min_similarity: 最小相似度阈值
        
        Returns:
            搜索结果列表，按相似度降序排列
        """
        if not self.chunks or self.vector_matrix is None:
            return []
        
        # 编码查询
        query_vector = self.embedder.encode_single(query)
        
        # 计算余弦相似度
        if self.embedder.is_available():
            # 使用向量化的高效计算
            similarities = np.dot(self.vector_matrix, query_vector)
        else:
            # 基础相似度计算
            similarities = np.array([
                self._cosine_similarity(chunk.vector, query_vector)
                for chunk in self.chunks
            ])
        
        # 获取top_k结果
        if len(similarities) <= top_k:
            top_indices = np.argsort(similarities)[::-1]
        else:
            top_indices = np.argsort(similarities)[::-1][:top_k]
        
        results = []
        for idx in top_indices:
            similarity = float(similarities[idx])
            if similarity < min_similarity:
                continue
            
            chunk = self.chunks[idx]
            result = VectorSearchResult(
                chunk_id=chunk.chunk_id,
                file_id=chunk.file_id,
                file_name=chunk.file_name,
                content=chunk.content,
                score=similarity,
                similarity=similarity,
                metadata={
                    "start_index": chunk.start_index,
                    "end_index": chunk.end_index,
                    **chunk.metadata
                }
            )
            results.append(result)
        
        return results
    
    def hybrid_search(self, query: str, top_k: int = TOP_K_RESULTS, 
                     vector_weight: float = 0.7, keyword_weight: float = 0.3) -> List[VectorSearchResult]:
        """
        混合搜索（向量+关键词）
        
        Args:
            query: 查询文本
            top_k: 返回结果数
            vector_weight: 向量搜索权重
            keyword_weight: 关键词搜索权重
        
        Returns:
            搜索结果列表
        """
        if not self.chunks:
            return []
        
        # 1. 向量搜索得分
        vector_results = self.search(query, top_k=len(self.chunks), min_similarity=0.0)
        vector_scores = {r.chunk_id: r.score for r in vector_results}
        
        # 2. 关键词搜索得分
        keyword_scores = self._keyword_match(query)
        
        # 3. 混合得分
        all_chunk_ids = set(vector_scores.keys()) | set(keyword_scores.keys())
        combined_scores = {}
        
        for chunk_id in all_chunk_ids:
            v_score = vector_scores.get(chunk_id, 0.0)
            k_score = keyword_scores.get(chunk_id, 0.0)
            combined_scores[chunk_id] = v_score * vector_weight + k_score * keyword_weight
        
        # 4. 排序并返回top_k
        sorted_chunks = sorted(combined_scores.items(), key=lambda x: x[1], reverse=True)[:top_k]
        
        # 构建结果
        results = []
        chunk_map = {c.chunk_id: c for c in self.chunks}
        
        for chunk_id, score in sorted_chunks:
            if score < 0.05:  # 最低阈值
                continue
                
            chunk = chunk_map.get(chunk_id)
            if chunk:
                result = VectorSearchResult(
                    chunk_id=chunk.chunk_id,
                    file_id=chunk.file_id,
                    file_name=chunk.file_name,
                    content=chunk.content,
                    score=score,
                    similarity=vector_scores.get(chunk_id, 0.0),
                    metadata={
                        "start_index": chunk.start_index,
                        "end_index": chunk.end_index,
                        "keyword_score": keyword_scores.get(chunk_id, 0.0),
                        "vector_score": vector_scores.get(chunk_id, 0.0),
                        **chunk.metadata
                    }
                )
                results.append(result)
        
        return results
    
    def _keyword_match(self, query: str) -> Dict[str, float]:
        """关键词匹配得分"""
        # 提取查询关键词
        query_words = set(re.findall(r'[\u4e00-\u9fa5]+|[a-zA-Z]+', query.lower()))
        if not query_words:
            return {}
        
        scores = {}
        for chunk in self.chunks:
            chunk_lower = chunk.content.lower()
            match_count = sum(1 for word in query_words if word in chunk_lower)
            if match_count > 0:
                # 归一化得分
                scores[chunk.chunk_id] = min(match_count / len(query_words), 1.0)
        
        return scores
    
    def _cosine_similarity(self, vec1: np.ndarray, vec2: np.ndarray) -> float:
        """计算余弦相似度"""
        if vec1 is None or vec2 is None:
            return 0.0
        
        dot_product = np.dot(vec1, vec2)
        norm1 = np.linalg.norm(vec1)
        norm2 = np.linalg.norm(vec2)
        
        if norm1 == 0 or norm2 == 0:
            return 0.0
        
        return float(dot_product / (norm1 * norm2))
    
    def _rebuild_vector_matrix(self):
        """重建向量矩阵"""
        if not self.chunks:
            self.vector_matrix = None
            return
        
        # 收集所有向量
        vectors = [chunk.vector for chunk in self.chunks if chunk.vector is not None]
        if vectors:
            self.vector_matrix = np.array(vectors)
        else:
            self.vector_matrix = None
    
    def _save_index(self):
        """保存索引到磁盘"""
        try:
            # 保存元数据
            index_data = {
                "chunks": [
                    {
                        "chunk_id": chunk.chunk_id,
                        "file_id": chunk.file_id,
                        "file_name": chunk.file_name,
                        "content": chunk.content,
                        "start_index": chunk.start_index,
                        "end_index": chunk.end_index,
                        "metadata": chunk.metadata
                    }
                    for chunk in self.chunks
                ],
                "timestamp": datetime.now().isoformat(),
                "total_chunks": len(self.chunks),
                "vector_dim": VECTOR_DIMENSION if self.embedder.is_available() else 128
            }
            
            with open(VECTOR_INDEX_FILE, "w", encoding="utf-8") as f:
                json.dump(index_data, f, ensure_ascii=False, indent=2)
            
            # 保存向量数据
            if self.vector_matrix is not None:
                np.savez_compressed(VECTOR_DATA_FILE, vectors=self.vector_matrix)
            
            print(f"索引已保存: {VECTOR_INDEX_FILE}")
            
        except Exception as e:
            print(f"保存索引失败: {e}")
    
    def load_index(self) -> bool:
        """从磁盘加载索引"""
        try:
            if not os.path.exists(VECTOR_INDEX_FILE):
                print("索引文件不存在")
                return False
            
            with open(VECTOR_INDEX_FILE, "r", encoding="utf-8") as f:
                index_data = json.load(f)
            
            # 加载向量数据
            vectors = None
            if os.path.exists(VECTOR_DATA_FILE):
                data = np.load(VECTOR_DATA_FILE)
                vectors = data["vectors"]
            
            # 重建分块
            self.chunks = []
            for i, chunk_data in enumerate(index_data["chunks"]):
                chunk = DocumentChunk(
                    chunk_id=chunk_data["chunk_id"],
                    file_id=chunk_data["file_id"],
                    file_name=chunk_data["file_name"],
                    content=chunk_data["content"],
                    start_index=chunk_data["start_index"],
                    end_index=chunk_data["end_index"],
                    vector=vectors[i] if vectors is not None else None,
                    metadata=chunk_data.get("metadata", {})
                )
                self.chunks.append(chunk)
            
            self.vector_matrix = vectors
            print(f"索引加载成功: {len(self.chunks)} 个分块")
            return True
            
        except Exception as e:
            print(f"加载索引失败: {e}")
            return False
    
    def get_stats(self) -> Dict[str, Any]:
        """获取索引统计信息"""
        file_ids = list(set(chunk.file_id for chunk in self.chunks))
        return {
            "total_chunks": len(self.chunks),
            "total_documents": len(file_ids),
            "vector_dim": self.vector_matrix.shape[1] if self.vector_matrix is not None else 0,
            "has_vector_model": self.embedder.is_available(),
            "documents": file_ids
        }


# ==================== 全局实例 ====================
_vector_index_instance = None


def get_vector_index() -> VectorIndexManager:
    """获取向量索引单例"""
    global _vector_index_instance
    if _vector_index_instance is None:
        _vector_index_instance = VectorIndexManager()
    return _vector_index_instance


# ==================== 便捷函数 ====================
def build_vector_index_from_knowledge() -> int:
    """从知识库构建向量索引"""
    try:
        from utils.knowledge_manager import _load_index, get_knowledge_content
        
        # 获取所有文件
        kb_index = _load_index()
        documents = []
        
        for file_info in kb_index.get("files", []):
            file_id = file_info["file_id"]
            file_name = file_info.get("original_name", file_id)
            content = get_knowledge_content(file_id)
            
            if content and not content.startswith("文件解析失败"):
                documents.append((file_id, file_name, content))
        
        if not documents:
            print("知识库中没有有效文档")
            return 0
        
        # 构建索引
        index = get_vector_index()
        return index.build_index(documents)
        
    except Exception as e:
        print(f"从知识库构建索引失败: {e}")
        import traceback
        traceback.print_exc()
        return 0


def vector_search(query: str, top_k: int = 5) -> List[Dict[str, Any]]:
    """便捷函数：向量搜索"""
    index = get_vector_index()
    
    # 尝试加载现有索引
    if not index.chunks:
        index.load_index()
    
    results = index.search(query, top_k=top_k)
    
    # 转换为字典格式
    return [
        {
            "file_name": r.file_name,
            "file_id": r.file_id,
            "content": r.content[:300],
            "similarity": r.similarity,
            "score": r.score,
            "metadata": r.metadata
        }
        for r in results
    ]


def hybrid_vector_search(query: str, top_k: int = 5) -> List[Dict[str, Any]]:
    """便捷函数：混合向量搜索"""
    index = get_vector_index()
    
    # 尝试加载现有索引
    if not index.chunks:
        index.load_index()
    
    results = index.hybrid_search(query, top_k=top_k)
    
    # 转换为字典格式
    return [
        {
            "file_name": r.file_name,
            "file_id": r.file_id,
            "content": r.content[:300],
            "similarity": r.similarity,
            "score": r.score,
            "keyword_score": r.metadata.get("keyword_score", 0),
            "vector_score": r.metadata.get("vector_score", 0),
            "metadata": r.metadata
        }
        for r in results
    ]
