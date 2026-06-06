import os
import json
import uuid
import hashlib
from datetime import datetime
from dotenv import load_dotenv
load_dotenv()

# 知识库配置
KNOWLEDGE_DIR = "./knowledge_base"
MAX_FILE_SIZE_MB = 50
MAX_FILE_SIZE_BYTES = MAX_FILE_SIZE_MB * 1024 * 1024
SUPPORTED_FORMATS = {"pdf", "txt", "md", "doc", "docx"}
INDEX_FILE = os.path.join(KNOWLEDGE_DIR, "index.json")


def init_knowledge_base():
    """初始化知识库目录"""
    os.makedirs(KNOWLEDGE_DIR, exist_ok=True)
    if not os.path.exists(INDEX_FILE):
        with open(INDEX_FILE, "w", encoding="utf-8") as f:
            json.dump({"files": [], "md5_index": {}}, f, ensure_ascii=False)


def _calculate_md5(content: bytes) -> str:
    """计算文件内容的MD5哈希值"""
    md5_hash = hashlib.md5()
    md5_hash.update(content)
    return md5_hash.hexdigest()


def _validate_file(file_path: str) -> tuple[bool, str]:
    """验证文件是否有效"""
    if not os.path.exists(file_path):
        return False, "文件不存在"

    file_size = os.path.getsize(file_path)
    if file_size == 0:
        return False, "文件为空"

    if file_size > MAX_FILE_SIZE_BYTES:
        return False, f"文件过大 ({file_size / 1024 / 1024:.1f}MB)，最大支持 {MAX_FILE_SIZE_MB}MB"

    ext = file_path.split(".")[-1].lower()
    if ext not in SUPPORTED_FORMATS:
        return False, f"不支持的文件格式: {ext}，支持格式: {', '.join(sorted(SUPPORTED_FORMATS))}"

    return True, f"文件有效 ({file_size / 1024:.1f} KB)"


def _load_index() -> dict:
    """加载知识库索引"""
    init_knowledge_base()
    if os.path.exists(INDEX_FILE):
        try:
            with open(INDEX_FILE, "r", encoding="utf-8") as f:
                return json.load(f)
        except:
            return {"files": [], "md5_index": {}}
    return {"files": [], "md5_index": {}}


def _save_index(index: dict):
    """保存知识库索引"""
    with open(INDEX_FILE, "w", encoding="utf-8") as f:
        json.dump(index, f, ensure_ascii=False, indent=2)


def _extract_text_from_file(file_path: str) -> str:
    """从文件中提取文本内容"""
    ext = file_path.split(".")[-1].lower()
    
    try:
        if ext == "pdf":
            try:
                from langchain_community.document_loaders import PyPDFLoader
                loader = PyPDFLoader(file_path)
                docs = loader.load()
                return "\n".join([doc.page_content for doc in docs])
            except Exception as e:
                return f"PDF解析失败: {e}"
        
        elif ext in ["txt", "md"]:
            try:
                with open(file_path, "r", encoding="utf-8") as f:
                    return f.read()
            except UnicodeDecodeError:
                with open(file_path, "r", encoding="gbk") as f:
                    return f.read()
        
        elif ext in ["doc", "docx"]:
            try:
                from docx import Document
                doc = Document(file_path)
                return "\n".join([para.text for para in doc.paragraphs])
            except Exception as e:
                return f"Word文档解析失败: {e}"
        
        else:
            return "不支持的文件格式"
            
    except Exception as e:
        return f"文件解析失败: {e}"


def upload_knowledge_file(file_name: str, file_content: bytes) -> dict:
    """上传知识库文件（带MD5去重）"""
    # 计算MD5
    file_md5 = _calculate_md5(file_content)
    
    # 检查是否已存在相同MD5的文件
    index = _load_index()
    
    # 检查MD5是否已存在
    if file_md5 in index.get("md5_index", {}):
        existing_file_id = index["md5_index"][file_md5]
        # 找到对应的文件信息
        for f in index.get("files", []):
            if f["file_id"] == existing_file_id:
                return {
                    "success": True,
                    "message": f"文件已存在，使用缓存版本: {f['original_name']}",
                    "file_id": existing_file_id,
                    "file_info": f,
                    "is_duplicate": True
                }
    
    # 文件不存在，执行上传
    ext = file_name.split(".")[-1].lower()
    file_id = f"{uuid.uuid4().hex[:8]}_{datetime.now().strftime('%Y%m%d')}"
    local_path = os.path.join(KNOWLEDGE_DIR, f"{file_id}.{ext}")

    # 保存文件
    with open(local_path, "wb") as f:
        f.write(file_content)

    # 验证文件
    valid, message = _validate_file(local_path)
    if not valid:
        os.remove(local_path)
        return {
            "success": False,
            "message": message,
            "file_id": None
        }

    # 提取文本内容（用于搜索）
    content = _extract_text_from_file(local_path)
    
    # 更新索引（包含MD5）
    file_info = {
        "file_id": file_id,
        "original_name": file_name,
        "local_path": local_path,
        "extension": ext,
        "size": len(file_content),
        "md5": file_md5,
        "upload_time": datetime.now().isoformat(),
        "content_length": len(content),
        "content_preview": content[:500] if content else "",
        "content": content  # 缓存完整内容，避免重复解析
    }
    index["files"].append(file_info)
    index["md5_index"] = index.get("md5_index", {})
    index["md5_index"][file_md5] = file_id
    _save_index(index)

    return {
        "success": True,
        "message": f"文件上传成功！",
        "file_id": file_id,
        "file_info": file_info,
        "is_duplicate": False
    }


def get_knowledge_list() -> list:
    """获取所有知识库文件列表"""
    index = _load_index()
    return index.get("files", [])


def get_knowledge_file(file_id: str) -> dict:
    """获取单个知识库文件信息"""
    index = _load_index()
    for f in index.get("files", []):
        if f["file_id"] == file_id:
            return f
    return None


def delete_knowledge_file(file_id: str) -> bool:
    """删除知识库文件（同时删除MD5索引）"""
    index = _load_index()
    file_info = None
    
    # 找到并删除索引记录
    for i, f in enumerate(index.get("files", [])):
        if f["file_id"] == file_id:
            file_info = f
            del index["files"][i]
            break
    
    if not file_info:
        return False
    
    # 删除MD5索引
    if "md5" in file_info and "md5_index" in index:
        md5_value = file_info["md5"]
        if md5_value in index["md5_index"]:
            del index["md5_index"][md5_value]
    
    # 删除物理文件
    if os.path.exists(file_info["local_path"]):
        os.remove(file_info["local_path"])
    
    # 保存更新后的索引
    _save_index(index)
    return True


def search_knowledge(query: str, max_results: int = 5) -> list:
    """搜索知识库内容"""
    results = []
    index = _load_index()
    query_lower = query.lower()
    
    # 构建搜索关键词列表
    import re
    query_words = []
    
    # 提取英文单词（保留完整的英文单词）
    english_words = re.findall(r'[a-zA-Z]+', query_lower)
    query_words.extend([word for word in english_words if len(word) >= 2])
    
    # 使用 jieba 分词处理中文（但先移除已提取的英文单词部分）
    import jieba
    
    # 先移除已提取的英文单词，避免重复处理
    cleaned_query = query_lower
    for word in english_words:
        cleaned_query = cleaned_query.replace(word, " ")
    
    chinese_words = [word for word in jieba.lcut(cleaned_query) if len(word) >= 2 and not re.match(r'^[a-zA-Z]+$', word)]
    query_words.extend(chinese_words)
    
    # 如果没有找到关键词，使用原始查询词
    if not query_words:
        query_words = [query_lower]
    
    # 去重
    query_words = list(set(query_words))
    
    for file_info in index.get("files", []):
        # 先在文件名和内容预览中搜索
        search_text = f"{file_info['original_name']} {file_info.get('content_preview', '')}"
        search_text_lower = search_text.lower()
        
        # 如果预览中没找到，尝试搜索完整内容
        full_content = None
        if not _has_match(query_words, search_text_lower):
            content = get_knowledge_content(file_info["file_id"])
            if content and not content.startswith("文件解析失败"):
                full_content = content
                search_text_lower = f"{file_info['original_name']} {content}".lower()
        
        # 检查是否有任何关键词匹配
        matched_words = _find_matches(query_words, search_text_lower)
        
        if matched_words:
            # 计算匹配得分（匹配的关键词数量 + 每个关键词出现次数）
            score = len(matched_words)
            for word in matched_words:
                score += search_text_lower.count(word)
            
            # 提取预览
            if full_content:
                preview = full_content[:300]
            else:
                preview = file_info.get("content_preview", "")[:300]
            
            results.append({
                "file_id": file_info["file_id"],
                "file_name": file_info["original_name"],
                "score": score,
                "preview": preview,
                "matched_words": matched_words
            })
    
    # 按匹配度排序
    results.sort(key=lambda x: x["score"], reverse=True)
    return results[:max_results]


import re

def _has_match(keywords: list, text: str) -> bool:
    """检查是否有任何关键词匹配（支持灵活匹配）"""
    for keyword in keywords:
        # 直接匹配
        if keyword in text:
            return True
        
        # 对于英文单词，尝试带空格的变体
        if re.match(r'^[a-zA-Z]+$', keyword):
            # 例如 springboot -> spring boot
            # 尝试在单词中插入空格进行匹配
            for i in range(2, len(keyword)):
                spaced_version = keyword[:i] + " " + keyword[i:]
                if spaced_version in text:
                    return True
    
    return False


def _find_matches(keywords: list, text: str) -> list:
    """查找所有匹配的关键词（支持灵活匹配）"""
    matched = []
    
    for keyword in keywords:
        # 直接匹配
        if keyword in text:
            matched.append(keyword)
            continue
        
        # 对于英文单词，尝试带空格的变体
        if re.match(r'^[a-zA-Z]+$', keyword):
            for i in range(2, len(keyword)):
                spaced_version = keyword[:i] + " " + keyword[i:]
                if spaced_version in text:
                    matched.append(keyword)
                    break
    
    return matched


def get_knowledge_content(file_id: str) -> str:
    """获取知识库文件的完整内容（优先使用缓存）"""
    file_info = get_knowledge_file(file_id)
    if not file_info:
        return "文件不存在"
    
    # 优先返回缓存的内容
    if "content" in file_info and file_info["content"]:
        return file_info["content"]
    
    # 回退到文件解析
    return _extract_text_from_file(file_info["local_path"])


def get_knowledge_stats() -> dict:
    """获取知识库统计信息"""
    index = _load_index()
    files = index.get("files", [])
    
    stats = {
        "total_files": len(files),
        "total_size": sum(f["size"] for f in files),
        "formats": {},
        "upload_dates": [],
        "unique_md5_count": len(index.get("md5_index", {}))
    }
    
    for f in files:
        ext = f["extension"]
        stats["formats"][ext] = stats["formats"].get(ext, 0) + 1
        stats["upload_dates"].append(f["upload_time"][:10])
    
    return stats


# ========== Hybrid Search 集成 ==========

def init_hybrid_search_index():
    """初始化 Hybrid Search 索引（从现有知识库构建）"""
    try:
        from utils.hybrid_search import (
            init_hybrid_search,
            add_document,
            build_hybrid_index,
            get_engine_stats
        )
        
        # 初始化引擎
        init_hybrid_search()
        
        # 获取现有知识库文件
        index = _load_index()
        
        # 将现有文档添加到 Hybrid Search
        for file_info in index.get("files", []):
            content = get_knowledge_content(file_info["file_id"])
            if content and not content.startswith("文件解析失败"):
                metadata = {
                    "file_id": file_info["file_id"],
                    "file_name": file_info["original_name"],
                    "source": "knowledge_base"
                }
                add_document(content, metadata)
        
        # 构建索引
        result = build_hybrid_index()
        
        # 获取统计信息
        stats = get_engine_stats()
        
        return {
            "success": result["success"],
            "message": result.get("message", ""),
            "document_count": stats.get("document_count", 0)
        }
    
    except ImportError:
        return {"success": False, "message": "Hybrid Search 依赖未安装"}
    except Exception as e:
        return {"success": False, "message": f"初始化失败: {str(e)}"}


def search_knowledge_hybrid(query: str, top_k: int = 5) -> list:
    """使用 Hybrid Search 搜索知识库"""
    try:
        from utils.hybrid_search import hybrid_search
        
        result = hybrid_search(query, top_k=top_k, mode="hybrid")
        
        if not result["success"]:
            # 如果 Hybrid Search 不可用，回退到传统搜索
            return search_knowledge(query, max_results=top_k)
        
        # 格式化结果
        formatted_results = []
        for res in result["results"]:
            formatted_results.append({
                "file_id": res["metadata"].get("file_id", ""),
                "file_name": res["metadata"].get("file_name", res["metadata"].get("source", "未知")),
                "score": res.get("score", 0.0),
                "preview": res["content"][:200] + "..." if len(res["content"]) > 200 else res["content"]
            })
        
        return formatted_results
    
    except ImportError:
        return search_knowledge(query, max_results=top_k)
    except Exception as e:
        return search_knowledge(query, max_results=top_k)


# ========== 面试文本与知识库匹配功能 ==========

def _calculate_cosine_similarity(vec1, vec2):
    """计算余弦相似度"""
    if not vec1 or not vec2:
        return 0.0
    
    dot_product = sum(a * b for a, b in zip(vec1, vec2))
    norm1 = sum(a * a for a in vec1) ** 0.5
    norm2 = sum(b * b for b in vec2) ** 0.5
    
    if norm1 == 0 or norm2 == 0:
        return 0.0
    
    return dot_product / (norm1 * norm2)


def split_text_into_paragraphs(text: str, min_length: int = 50) -> list:
    """
    将文本按段落进行分割
    
    段落边界识别标准：
    1. 连续两个或多个换行符（空行）作为主要段落分隔符
    2. 标题行（以【、第章、第节、#开头）作为段落边界
    3. 不同类型列表项之间的换行作为段落边界
    4. 确保每个段落有最小长度，避免过短的段落
    
    Args:
        text: 输入文本
        min_length: 段落最小长度（字符数）
    
    Returns:
        段落列表
    """
    import re
    
    # 标准化换行符
    text = text.replace('\r\n', '\n').replace('\r', '\n')
    
    # 段落分割策略：
    # 1. 以连续换行符（空行）作为主要分隔符
    raw_paragraphs = re.split(r'\n\n+', text)
    
    # 2. 进一步处理每个原始段落，按标题分割
    paragraphs = []
    for raw_p in raw_paragraphs:
        raw_p = raw_p.strip()
        if not raw_p:
            continue
        
        # 按标题分割：标题行单独成为段落
        # 匹配标题格式：【...】、第...章、第...节、#开头
        title_pattern = r'(\n?【[^】]+】\n?|\n?第[^章章节节]+[章节]\n?|\n?#+ .+\n?)'
        parts = re.split(title_pattern, raw_p)
        
        for part in parts:
            part = part.strip()
            if part:
                paragraphs.append(part)
    
    # 3. 清理和合并列表项
    cleaned_paragraphs = []
    list_types = {
        'ordered': re.compile(r'^(\d+[.\uff0e]|\u2460-\u2473)\s+'),
        'unordered': re.compile(r'^([\u2022\u2023\u25cf\u25cb\u25aa]|[-*+>])\s+'),
        'other': re.compile(r'^(\d+\.\d+|①|②|③|㈠|㈡|㈢)\s+')
    }
    
    for p in paragraphs:
        # 去除多余的空白字符
        p = re.sub(r'[ \t]+', ' ', p)
        p = re.sub(r'\n+', '\n', p)
        
        # 判断当前段落类型
        current_type = None
        if list_types['ordered'].search(p):
            current_type = 'ordered'
        elif list_types['unordered'].search(p):
            current_type = 'unordered'
        elif list_types['other'].search(p):
            current_type = 'other'
        
        # 过滤过短的段落（除非是标题或列表项）
        if len(p) >= min_length:
            cleaned_paragraphs.append({
                'content': p,
                'type': current_type if current_type else 'normal'
            })
        elif len(p) > 0:
            # 检查是否是标题
            if re.match(r'^【[^】]+】$|^第[^章章节节]+[章节]$|^#+ .+$', p):
                # 标题作为独立段落
                cleaned_paragraphs.append({
                    'content': p,
                    'type': 'title'
                })
            elif current_type:
                # 列表项
                if cleaned_paragraphs:
                    # 检查前一个段落是否是同类型的列表
                    prev_p = cleaned_paragraphs[-1]
                    if prev_p['type'] == current_type:
                        # 合并到前一个同类型列表
                        cleaned_paragraphs[-1]['content'] += '\n' + p
                    else:
                        # 作为新段落
                        cleaned_paragraphs.append({
                            'content': p,
                            'type': current_type
                        })
                else:
                    cleaned_paragraphs.append({
                        'content': p,
                        'type': current_type
                    })
    
    # 提取纯内容列表
    return [p['content'] for p in cleaned_paragraphs]


def _text_to_vector(text: str) -> dict:
    """将文本转换为词频向量（支持中文分词）"""
    # 尝试使用 jieba 进行中文分词
    try:
        import jieba
        words = jieba.lcut(text.lower())
    except ImportError:
        # 如果没有安装 jieba，使用简单的分词方法
        words = text.lower().split()
    
    # 过滤标点符号和空白字符
    import re
    word_count = {}
    for word in words:
        # 只保留字母和数字
        cleaned_word = re.sub(r'[^a-zA-Z0-9\u4e00-\u9fa5]', '', word)
        if cleaned_word:
            word_count[cleaned_word] = word_count.get(cleaned_word, 0) + 1
    
    return word_count


def _get_common_vector(vec1: dict, vec2: dict) -> tuple:
    """获取两个向量的公共维度"""
    all_words = set(vec1.keys()).union(set(vec2.keys()))
    v1 = [vec1.get(word, 0) for word in all_words]
    v2 = [vec2.get(word, 0) for word in all_words]
    return v1, v2


def match_knowledge_with_interview(interview_text: str, top_k: int = 3, similarity_threshold: float = 0.05, use_vector: bool = True) -> list:
    """
    匹配面试文本与知识库内容
    使用余弦相似度算法进行匹配，支持段落级别的精细匹配和向量搜索
    
    Args:
        interview_text: 面试文本
        top_k: 返回前K个最相关的结果
        similarity_threshold: 相似度阈值，低于此值的结果不返回
        use_vector: 是否使用向量搜索
    
    Returns:
        匹配结果列表，按相似度降序排列
    """
    results = []
    index = _load_index()
    
    # 方法一：尝试向量搜索（如果启用且有向量索引）
    vector_results = []
    if use_vector:
        try:
            vector_results = search_knowledge_by_vector(interview_text, max_results=top_k, use_hybrid=True)
        except Exception as e:
            print(f"向量搜索失败: {e}")
    
    # 如果向量搜索结果格式转换
    vector_file_ids = set()
    for vr in vector_results:
        file_id = vr.get("file_id")
        if file_id not in vector_file_ids:
            vector_file_ids.add(file_id)
            # 转换为统一格式
            results.append({
                "file_id": vr["file_id"],
                "file_name": vr["file_name"],
                "similarity": vr.get("similarity", 0),
                "key_fragments": [{"text": vr.get("preview", "")}],
                "content_preview": vr.get("preview", ""),
                "search_type": "vector"
            })
    
    # 方法二：传统段落匹配（补充向量搜索未覆盖的结果
    if len(results) < top_k:
        interview_vector = _text_to_vector(interview_text)
        
        for file_info in index.get("files", []):
            file_id = file_info["file_id"]
            # 跳过已经在向量搜索结果中的文件
            if file_id in vector_file_ids:
                continue
                
            content = get_knowledge_content(file_id)
            
            if not content or content.startswith("文件解析失败") or content.startswith("不支持"):
                continue
            
            # 使用段落级匹配
            paragraph_matches = _match_by_paragraph(interview_text, content, similarity_threshold)
            
            if paragraph_matches:
                # 计算整体相似度（取最高段落相似度）
                max_similarity = max(p["similarity"] for p in paragraph_matches)
                
                # 提取关键信息片段（来自匹配的段落）
                key_fragments = [{"text": p["paragraph"]} for p in paragraph_matches[:3]]
                
                results.append({
                    "file_id": file_id,
                    "file_name": file_info["original_name"],
                    "similarity": max_similarity,
                    "key_fragments": key_fragments,
                    "content_preview": content[:300],
                    "matched_paragraphs": paragraph_matches,
                    "search_type": "paragraph"
                })
    
    # 按相似度排序
    results.sort(key=lambda x: x["similarity"], reverse=True)
    
    return results[:top_k]


def _match_by_paragraph(interview_text: str, knowledge_text: str, threshold: float = 0.05) -> list:
    """
    按段落级别匹配面试文本与知识库内容
    
    Args:
        interview_text: 面试文本
        knowledge_text: 知识库文本
        threshold: 相似度阈值
    
    Returns:
        匹配的段落列表，包含段落内容和相似度
    """
    matches = []
    
    # 将知识库文本按段落分割
    paragraphs = split_text_into_paragraphs(knowledge_text)
    
    if not paragraphs:
        return matches
    
    # 将面试文本转换为向量
    interview_vector = _text_to_vector(interview_text)
    
    for paragraph in paragraphs:
        # 将段落转换为向量
        paragraph_vector = _text_to_vector(paragraph)
        
        # 计算余弦相似度
        v1, v2 = _get_common_vector(interview_vector, paragraph_vector)
        similarity = _calculate_cosine_similarity(v1, v2)
        
        if similarity >= threshold:
            matches.append({
                "paragraph": paragraph,
                "similarity": similarity
            })
    
    # 按相似度排序
    matches.sort(key=lambda x: x["similarity"], reverse=True)
    
    return matches


def _extract_key_fragments(query_text: str, knowledge_text: str, max_fragments: int = 3, fragment_length: int = 100) -> list:
    """
    从知识库文本中提取与查询相关的关键片段
    
    Args:
        query_text: 查询文本
        knowledge_text: 知识库文本
        max_fragments: 最大提取片段数
        fragment_length: 每个片段的长度
    
    Returns:
        关键片段列表
    """
    fragments = []
    query_words = set(query_text.lower().split())
    
    # 将知识库文本分段
    sentences = knowledge_text.replace("\n", " ").split(".")
    
    for sentence in sentences:
        sentence = sentence.strip()
        if len(sentence) < 10:
            continue
        
        # 计算句子与查询的匹配度
        sentence_words = set(sentence.lower().split())
        common_words = query_words.intersection(sentence_words)
        
        if common_words:
            fragments.append({
                "text": sentence[:fragment_length] + "..." if len(sentence) > fragment_length else sentence,
                "match_words": list(common_words),
                "match_count": len(common_words)
            })
    
    # 按匹配词数排序
    fragments.sort(key=lambda x: x["match_count"], reverse=True)
    
    return fragments[:max_fragments]


def generate_prompt_from_knowledge(interview_text: str, job_description: str = "") -> str:
    """
    根据面试文本和知识库内容生成大模型提示词
    
    Args:
        interview_text: 面试文本
        job_description: 岗位描述（可选）
    
    Returns:
        格式化的提示词
    """
    # 匹配知识库
    matched_results = match_knowledge_with_interview(interview_text, top_k=3)
    
    # 构建提示词
    prompt_parts = []
    
    if job_description:
        prompt_parts.append(f"【岗位描述】:\n{job_description}\n")
    
    prompt_parts.append(f"【面试文本】:\n{interview_text}\n")
    
    if matched_results:
        prompt_parts.append("【参考知识库】:")
        for i, result in enumerate(matched_results, 1):
            prompt_parts.append(f"\n{i}. 来源: {result['file_name']} (相似度: {result['similarity']:.2f})")
            if result["key_fragments"]:
                prompt_parts.append("   关键信息:")
                for fragment in result["key_fragments"]:
                    prompt_parts.append(f"   - {fragment['text']}")
    
    prompt_parts.append("\n请基于以上信息，对本次面试进行深入分析，包括技术能力评估、回答质量评价、改进建议等。")
    
    return "\n".join(prompt_parts)


def get_relevant_knowledge_summary(interview_text: str) -> str:
    """获取与面试文本相关的知识库摘要"""
    matched_results = match_knowledge_with_interview(interview_text, top_k=5)
    
    if not matched_results:
        return "知识库中未找到与本次面试相关的内容。"
    
    summary_parts = []
    summary_parts.append("📚 知识库匹配结果:")
    
    for i, result in enumerate(matched_results, 1):
        summary_parts.append(f"\n{i}. **{result['file_name']}**")
        summary_parts.append(f"   相似度: {result['similarity']:.2%}")
        
        if result["key_fragments"]:
            summary_parts.append("   相关片段:")
            for fragment in result["key_fragments"]:
                summary_parts.append(f"   - {fragment['text']}")
    
    return "\n".join(summary_parts)


# ==================== 向量搜索功能 ====================
def search_knowledge_by_vector(query: str, max_results: int = 5, use_hybrid: bool = True) -> list:
    """
    使用向量搜索知识库内容
    
    Args:
        query: 查询文本
        max_results: 返回结果数
        use_hybrid: 是否使用混合搜索（向量+关键词）
    
    Returns:
        搜索结果列表
    """
    try:
        from utils.vector_index import vector_search, hybrid_vector_search, build_vector_index_from_knowledge
        
        # 检查是否有可用的向量索引
        from utils.vector_index import get_vector_index
        v_index = get_vector_index()
        
        if not v_index.chunks:
            # 尝试加载现有索引
            if not v_index.load_index():
                # 如果没有索引，尝试构建
                print("未找到向量索引，正在构建...")
                build_vector_index_from_knowledge()
        
        # 执行搜索
        if use_hybrid:
            results = hybrid_vector_search(query, top_k=max_results)
        else:
            results = vector_search(query, top_k=max_results)
        
        # 格式化结果
        formatted_results = []
        for r in results:
            formatted_results.append({
                "file_id": r["file_id"],
                "file_name": r["file_name"],
                "score": r.get("score", r.get("similarity", 0)),
                "similarity": r.get("similarity", 0),
                "preview": r.get("content", "")[:300],
                "matched_words": [],
                "search_type": "vector_hybrid" if use_hybrid else "vector"
            })
        
        return formatted_results
        
    except Exception as e:
        print(f"向量搜索失败: {e}")
        # 回退到普通关键词搜索
        return search_knowledge(query, max_results)


# ==================== 智能混合检索策略 ====================
def search_knowledge_smart(query: str, max_results: int = 5) -> dict:
    """
    智能混合检索策略：自动切换向量检索和文本匹配检索
    
    检索逻辑：
    1. 优先执行向量检索（语义理解能力强）
    2. 检查向量检索结果：
       - 如果为空列表 → 直接使用文本匹配检索
       - 如果相关性均低于阈值 → 补充执行文本匹配检索
       - 如果结果质量良好 → 返回向量检索结果
    3. 整合两种检索结果，去重并按相关性排序
    
    Args:
        query: 查询文本
        max_results: 返回结果数
    
    Returns:
        dict: {
            "results": [...],  # 合并后的搜索结果
            "vector_results": [...],  # 向量检索结果
            "keyword_results": [...],  # 文本匹配结果
            "strategy_used": str,  # 使用的检索策略
            "total_found": int,  # 总结果数
            "relevance_threshold_met": bool  # 是否达到相关性阈值
        }
    """
    # 相关性阈值配置
    VECTOR_SIMILARITY_THRESHOLD = 0.15  # 向量相似度阈值
    MIN_VECTOR_RESULTS = 1  # 最少需要的向量检索结果数
    KEYWORD_SCORE_THRESHOLD = 2  # 关键词匹配得分阈值
    
    results_info = {
        "results": [],
        "vector_results": [],
        "keyword_results": [],
        "strategy_used": "unknown",
        "total_found": 0,
        "relevance_threshold_met": False,
        "search_metadata": {
            "query": query,
            "max_results": max_results,
            "vector_threshold": VECTOR_SIMILARITY_THRESHOLD,
            "keyword_threshold": KEYWORD_SCORE_THRESHOLD,
            "timestamp": datetime.now().isoformat()
        }
    }
    
    # 第一步：执行向量检索
    try:
        vector_results = search_knowledge_by_vector(query, max_results=max_results, use_hybrid=True)
        results_info["vector_results"] = vector_results
        print(f"[智能检索] 向量检索完成，返回 {len(vector_results)} 个结果")
        
        # 检查向量检索结果质量
        vector_quality_good = False
        
        if vector_results:
            # 计算平均相似度和最高相似度
            similarities = [r.get("similarity", 0) for r in vector_results]
            max_similarity = max(similarities) if similarities else 0
            avg_similarity = sum(similarities) / len(similarities) if similarities else 0
            
            results_info["search_metadata"]["vector_max_similarity"] = max_similarity
            results_info["search_metadata"]["vector_avg_similarity"] = avg_similarity
            
            # 判断向量检索结果质量
            # 条件1：最高相似度超过阈值
            # 条件2：至少有MIN_VECTOR_RESULTS个结果，且平均相似度较高
            vector_quality_good = (
                max_similarity >= VECTOR_SIMILARITY_THRESHOLD or
                (len(vector_results) >= MIN_VECTOR_RESULTS and avg_similarity >= VECTOR_SIMILARITY_THRESHOLD * 0.5)
            )
            
            print(f"[智能检索] 向量检索质量评估: max_sim={max_similarity:.4f}, avg_sim={avg_similarity:.4f}, quality_good={vector_quality_good}")
        
    except Exception as e:
        print(f"[智能检索] 向量检索失败: {e}")
        vector_results = []
        results_info["vector_results"] = []
    
    # 第二步：根据向量检索结果决定是否执行文本匹配
    keyword_results = []
    
    if not vector_quality_good:
        # 情况1：向量检索结果为空
        if not vector_results:
            print("[智能检索] 向量检索结果为空，切换到文本匹配检索")
            results_info["strategy_used"] = "keyword_fallback"
        
        # 情况2：向量检索结果相关性都较低
        else:
            print("[智能检索] 向量检索相关性较低，补充文本匹配检索")
            results_info["strategy_used"] = "hybrid_supplement"
        
        # 执行文本匹配检索
        try:
            keyword_results = search_knowledge(query, max_results=max_results)
            results_info["keyword_results"] = keyword_results
            print(f"[智能检索] 文本匹配完成，返回 {len(keyword_results)} 个结果")
        except Exception as e:
            print(f"[智能检索] 文本匹配失败: {e}")
            keyword_results = []
            results_info["keyword_results"] = []
    
    else:
        # 向量检索结果质量良好
        print("[智能检索] 向量检索结果质量良好")
        results_info["strategy_used"] = "vector_primary"
    
    # 第三步：整合两种检索结果
    final_results = _merge_search_results(
        vector_results,
        keyword_results,
        max_results
    )
    
    results_info["results"] = final_results
    results_info["total_found"] = len(final_results)
    results_info["relevance_threshold_met"] = vector_quality_good or len(keyword_results) > 0
    
    # 更新搜索元数据
    results_info["search_metadata"]["final_results_count"] = len(final_results)
    results_info["search_metadata"]["vector_result_count"] = len(vector_results)
    results_info["search_metadata"]["keyword_result_count"] = len(keyword_results)
    
    return results_info


def _merge_search_results(
    vector_results: list,
    keyword_results: list,
    max_results: int = 5
) -> list:
    """
    合并向量检索和文本匹配检索结果
    
    合并策略：
    1. 去重：相同文档只保留一个结果
    2. 评分归一化：将向量相似度和关键词得分归一化到同一尺度
    3. 加权融合：向量检索权重0.7，关键词权重0.3
    4. 排序：按综合得分降序排列
    
    Args:
        vector_results: 向量检索结果
        keyword_results: 文本匹配结果
        max_results: 最大返回结果数
    
    Returns:
        合并后的结果列表
    """
    if not vector_results and not keyword_results:
        return []
    
    # 构建文档ID到结果的映射
    doc_map = {}
    
    # 处理向量检索结果（相似度 0-1）
    for result in vector_results:
        file_id = result.get("file_id")
        if file_id:
            # 归一化向量相似度（已经是 0-1 范围）
            vector_score = result.get("similarity", 0)
            
            doc_map[file_id] = {
                "file_id": file_id,
                "file_name": result.get("file_name", ""),
                "score": vector_score * 0.7,  # 向量权重 0.7
                "similarity": vector_score,
                "preview": result.get("preview", ""),
                "matched_words": result.get("matched_words", []),
                "search_type": "vector",
                "vector_score": vector_score,
                "keyword_score": 0,
                "sources": ["vector"]
            }
    
    # 处理关键词检索结果（得分需要归一化）
    if keyword_results:
        # 找出关键词得分的最大值，用于归一化
        max_keyword_score = max(
            (r.get("score", 0) for r in keyword_results),
            default=1
        )
        if max_keyword_score == 0:
            max_keyword_score = 1
        
        for result in keyword_results:
            file_id = result.get("file_id")
            if file_id:
                keyword_score = result.get("score", 0)
                normalized_keyword = keyword_score / max_keyword_score  # 归一化到 0-1
                
                if file_id in doc_map:
                    # 文档已存在，更新分数
                    existing = doc_map[file_id]
                    existing["score"] = existing["score"] + normalized_keyword * 0.3  # 关键词权重 0.3
                    existing["keyword_score"] = normalized_keyword
                    existing["sources"].append("keyword")
                    # 更新最优预览
                    if keyword_score > existing.get("_keyword_score_raw", 0):
                        existing["preview"] = result.get("preview", "")
                        existing["matched_words"] = result.get("matched_words", [])
                        existing["_keyword_score_raw"] = keyword_score
                else:
                    # 新文档
                    doc_map[file_id] = {
                        "file_id": file_id,
                        "file_name": result.get("file_name", ""),
                        "score": normalized_keyword * 0.3,  # 只有关键词得分
                        "similarity": 0,  # 无向量相似度
                        "preview": result.get("preview", ""),
                        "matched_words": result.get("matched_words", []),
                        "search_type": "keyword",
                        "vector_score": 0,
                        "keyword_score": normalized_keyword,
                        "sources": ["keyword"]
                    }
    
    # 转换为列表并排序
    merged_results = list(doc_map.values())
    merged_results.sort(key=lambda x: x["score"], reverse=True)
    
    # 返回前max_results个结果
    return merged_results[:max_results]


def build_vector_index_for_knowledge() -> dict:
    """
    为知识库构建向量索引
    
    Returns:
        构建结果信息
    """
    try:
        from utils.vector_index import build_vector_index_from_knowledge, get_vector_index
        
        doc_count = build_vector_index_from_knowledge()
        
        v_index = get_vector_index()
        stats = v_index.get_stats()
        
        return {
            "success": True,
            "documents_indexed": doc_count,
            "total_chunks": stats["total_chunks"],
            "vector_dimension": stats["vector_dim"],
            "has_vector_model": stats["has_vector_model"],
            "message": f"成功索引 {doc_count} 个文档，生成 {stats['total_chunks']} 个分块"
        }
        
    except Exception as e:
        print(f"构建向量索引失败: {e}")
        import traceback
        traceback.print_exc()
        return {
            "success": False,
            "documents_indexed": 0,
            "total_chunks": 0,
            "vector_dimension": 0,
            "has_vector_model": False,
            "message": f"构建失败: {e}"
        }


def get_vector_index_stats() -> dict:
    """获取向量索引统计信息"""
    try:
        from utils.vector_index import get_vector_index
        v_index = get_vector_index()
        
        # 尝试加载索引
        if not v_index.chunks:
            v_index.load_index()
        
        stats = v_index.get_stats()
        return stats
        
    except Exception as e:
        print(f"获取索引统计失败: {e}")
        return {
            "total_chunks": 0,
            "total_documents": 0,
            "vector_dim": 0,
            "has_vector_model": False,
            "error": str(e)
        }

