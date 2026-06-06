#!/usr/bin/env python3
"""
验证 SpringBoot 搜索结果
"""

import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from utils.knowledge_manager import search_knowledge, get_knowledge_content

print("测试搜索 'SpringBoot'")
results = search_knowledge("SpringBoot")
print(f"找到 {len(results)} 个结果")
for result in results:
    print(f"\n文件: {result['file_name']}")
    print(f"匹配关键词: {result['matched_words']}")
    print(f"得分: {result['score']}")

print("\n" + "="*60)
print("测试搜索 'SpringBoot 是什么？'")
results = search_knowledge("SpringBoot 是什么？")
print(f"找到 {len(results)} 个结果")
for result in results:
    print(f"\n文件: {result['file_name']}")
    print(f"匹配关键词: {result['matched_words']}")
    print(f"得分: {result['score']}")
    print(f"预览:\n{result['preview']}")
