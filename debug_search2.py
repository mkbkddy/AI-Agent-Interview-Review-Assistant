#!/usr/bin/env python3
"""
详细调试搜索问题
"""

import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from utils.knowledge_manager import (
    _load_index,
    search_knowledge,
    get_knowledge_content
)

# 手动模拟搜索逻辑
def manual_search(query):
    print(f"手动搜索: '{query}'")
    query_lower = query.lower()
    print(f"查询词(小写): '{query_lower}'")
    
    index = _load_index()
    
    for file_info in index.get("files", []):
        print(f"\n📄 文件: {file_info['original_name']}")
        
        # 先在文件名和预览中搜索
        search_text = f"{file_info['original_name']} {file_info.get('content_preview', '')}"
        search_text_lower = search_text.lower()
        print(f"预览文本长度: {len(search_text)}")
        
        if query_lower in search_text_lower:
            print(f"✅ 在预览中找到匹配")
            return True
        else:
            print(f"❌ 预览中未找到")
            print(f"预览文本前200字符: '{search_text[:200]}'")
            
            # 搜索完整内容
            content = get_knowledge_content(file_info["file_id"])
            print(f"完整内容长度: {len(content)}")
            
            if content and not content.startswith("文件解析失败"):
                content_lower = content.lower()
                if query_lower in content_lower:
                    print(f"✅ 在完整内容中找到匹配")
                    
                    # 查找匹配位置
                    idx = content_lower.find(query_lower)
                    print(f"匹配位置: {idx}")
                    print(f"匹配上下文: '{content_lower[idx-20:idx+50]}'")
                    return True
                else:
                    print(f"❌ 完整内容中也未找到")
                    
                    # 检查部分匹配
                    for word in query_lower.split():
                        if word in content_lower:
                            print(f"   部分匹配: '{word}'")
    
    return False

# 测试
print("="*60)
print("手动搜索测试")
print("="*60)

result = manual_search("SpringBoot 是什么？")
print(f"\n最终结果: {'找到' if result else '未找到'}")

print("\n" + "="*60)
print("实际 search_knowledge 测试")
print("="*60)
results = search_knowledge("SpringBoot 是什么？")
print(f"search_knowledge 返回: {len(results)} 个结果")
print(f"结果: {results}")
