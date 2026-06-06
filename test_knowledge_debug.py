#!/usr/bin/env python3
"""
知识库匹配功能调试测试
"""

import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from utils.knowledge_manager import (
    _text_to_vector,
    _get_common_vector,
    _calculate_cosine_similarity,
    match_knowledge_with_interview,
    get_knowledge_content,
    get_knowledge_list
)

def test_vector_conversion():
    """测试文本向量化"""
    print("="*60)
    print("🧪 测试文本向量化")
    print("="*60)
    
    test_texts = [
        "Python 机器学习",
        "机器学习入门",
        "深度学习神经网络"
    ]
    
    for text in test_texts:
        vec = _text_to_vector(text)
        print(f"\n文本: '{text}'")
        print(f"向量: {dict(sorted(vec.items()))}")

def test_cosine_similarity():
    """测试余弦相似度计算"""
    print("\n" + "="*60)
    print("🧪 测试余弦相似度")
    print("="*60)
    
    test_pairs = [
        ("Python 机器学习", "机器学习入门"),
        ("深度学习神经网络", "神经网络深度学习"),
        ("面试技巧", "简历优化"),
        ("Python编程", "Java编程")
    ]
    
    for text1, text2 in test_pairs:
        vec1 = _text_to_vector(text1)
        vec2 = _text_to_vector(text2)
        v1, v2 = _get_common_vector(vec1, vec2)
        similarity = _calculate_cosine_similarity(v1, v2)
        print(f"\n'{text1}' vs '{text2}'")
        print(f"   相似度: {similarity:.4f}")

def test_knowledge_matching_debug():
    """调试知识库匹配"""
    print("\n" + "="*60)
    print("🧪 调试知识库匹配")
    print("="*60)
    
    # 获取知识库列表
    knowledge_list = get_knowledge_list()
    print(f"\n知识库文件: {len(knowledge_list)} 个")
    
    # 面试文本
    interview_text = "面试官问了我关于Python机器学习的问题，包括神经网络和深度学习的基础知识。"
    
    print(f"\n面试文本: {interview_text}")
    interview_vec = _text_to_vector(interview_text)
    print(f"面试文本向量: {dict(sorted(interview_vec.items()))}")
    
    # 逐一检查知识库文件
    for file_info in knowledge_list:
        content = get_knowledge_content(file_info["file_id"])
        if not content or content.startswith("文件解析失败"):
            continue
        
        print(f"\n📄 {file_info['original_name']}")
        print(f"内容: {content[:100]}...")
        
        # 计算相似度
        content_vec = _text_to_vector(content)
        v1, v2 = _get_common_vector(interview_vec, content_vec)
        similarity = _calculate_cosine_similarity(v1, v2)
        print(f"相似度: {similarity:.4f}")
        
        # 显示公共词
        common_words = set(interview_vec.keys()) & set(content_vec.keys())
        if common_words:
            print(f"公共词: {common_words}")

if __name__ == "__main__":
    test_vector_conversion()
    test_cosine_similarity()
    test_knowledge_matching_debug()
