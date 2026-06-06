#!/usr/bin/env python3
"""
调试搜索问题
"""

import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from utils.knowledge_manager import (
    _load_index,
    search_knowledge,
    get_knowledge_content
)

# 加载索引
index = _load_index()

print("知识库文件列表:")
for i, file_info in enumerate(index.get("files", [])):
    print(f"\n📄 文件 {i+1}: {file_info['original_name']}")
    print(f"   file_id: {file_info['file_id']}")
    print(f"   content_preview 长度: {len(file_info.get('content_preview', ''))}")
    print(f"   content_preview 前100字符: '{file_info.get('content_preview', '')[:100]}'")
    
    # 检查是否包含 SpringBoot
    search_text = f"{file_info['original_name']} {file_info.get('content_preview', '')}".lower()
    print(f"   包含 'springboot': {'springboot' in search_text}")
    print(f"   包含 'spring': {'spring' in search_text}")
    
    # 获取完整内容检查
    content = get_knowledge_content(file_info['file_id'])
    print(f"   完整内容包含 'SpringBoot': {'SpringBoot' in content}")

# 测试搜索
print("\n" + "="*60)
print("测试搜索:")
print("="*60)

query = "SpringBoot"
print(f"\n搜索 '{query}'")
results = search_knowledge(query)
print(f"结果: {len(results)} 个")

query = "spring"
print(f"\n搜索 '{query}'")
results = search_knowledge(query)
print(f"结果: {len(results)} 个")

query = "Spring"
print(f"\n搜索 '{query}'")
results = search_knowledge(query)
print(f"结果: {len(results)} 个")

# 检查搜索逻辑
print("\n" + "="*60)
print("搜索逻辑检查:")
print("="*60)

query = "SpringBoot 是什么？"
query_lower = query.lower()
print(f"查询词: '{query}'")
print(f"查询词(小写): '{query_lower}'")

for file_info in index.get("files", []):
    search_text = f"{file_info['original_name']} {file_info.get('content_preview', '')}".lower()
    print(f"\n文件: {file_info['original_name']}")
    print(f"搜索文本长度: {len(search_text)}")
    print(f"查询词是否在搜索文本中: {query_lower in search_text}")
    
    # 检查是否有部分匹配
    for word in query_lower.split():
        if word in search_text:
            print(f"   部分匹配: '{word}'")
