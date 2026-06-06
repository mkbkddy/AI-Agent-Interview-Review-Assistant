#!/usr/bin/env python3
"""
调试关键词提取问题
"""

import re
import jieba

query = "SpringBoot 是什么？"
query_lower = query.lower()
print(f"原始查询: '{query}'")
print(f"小写查询: '{query_lower}'")

# 提取英文单词
english_words = re.findall(r'[a-zA-Z]+', query_lower)
print(f"\n提取的英文单词: {english_words}")

# 移除英文单词
cleaned_query = query_lower
for word in english_words:
    cleaned_query = cleaned_query.replace(word, " ")
print(f"移除英文后: '{cleaned_query}'")

# jieba 分词
chinese_words = [word for word in jieba.lcut(cleaned_query) if len(word) >= 2 and not re.match(r'^[a-zA-Z]+$', word)]
print(f"中文分词结果: {chinese_words}")

# 合并关键词
query_words = [word for word in english_words if len(word) >= 2]
query_words.extend(chinese_words)
query_words = list(set(query_words))
print(f"\n最终关键词列表: {query_words}")

# 测试搜索文本
search_text = "Spring Boot 的启动核⼼⼊⼝是 SpringApplication.run()"
search_text_lower = search_text.lower()
print(f"\n搜索文本: '{search_text_lower}'")

for word in query_words:
    if word in search_text_lower:
        print(f"✓ '{word}' 在搜索文本中")
    else:
        print(f"✗ '{word}' 不在搜索文本中")
