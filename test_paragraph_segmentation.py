#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
测试段落级分词功能
"""
from utils.knowledge_manager import split_text_into_paragraphs, match_knowledge_with_interview, get_knowledge_list

def test_paragraph_segmentation():
    """测试段落分割功能"""
    print("=" * 60)
    print("测试段落级分词功能")
    print("=" * 60)
    
    # 测试文本1：包含空行分隔的段落
    test_text1 = """一、类加载过程（类的生命周期）

一个类从被加载到 JVM 开始，到卸载出内存为止，其生命周期包括 7 个阶段：加载、验证、准备、解析、初始化、使用、卸载。

其中，验证、准备、解析统称为连接（Linking）阶段。

1. 加载（Loading）
做什么：
通过类的全限定名获取定义此类的二进制字节流（来源可以是 class 文件、jar 包、网络、动态代理生成等）。
将字节流的静态存储结构转化为方法区的运行时数据结构。

2. 验证（Verification）
验证阶段的目的是确保 Class 文件的字节流中包含的信息符合当前 JVM 的要求，并且不会危害 JVM 自身的安全。"""
    
    print("\n测试文本1：包含空行分隔的段落")
    print("-" * 40)
    paragraphs = split_text_into_paragraphs(test_text1)
    print(f"段落数量: {len(paragraphs)}")
    for i, p in enumerate(paragraphs, 1):
        print(f"\n段落 {i}:")
        print(f"长度: {len(p)} 字符")
        print(f"内容预览: {p[:100]}...")
    
    # 测试文本2：包含列表项的段落
    test_text2 = """Spring Boot 启动流程：
1. 创建 SpringApplication 实例
2. 运行 run 方法
3. 创建 ApplicationContext 上下文
4. 刷新上下文
5. 执行扩展点

为什么要使用 Spring Boot？
- 简化配置
- 内嵌服务器
- 自动配置
- 社区支持"""
    
    print("\n\n测试文本2：包含列表项的段落")
    print("-" * 40)
    paragraphs = split_text_into_paragraphs(test_text2)
    print(f"段落数量: {len(paragraphs)}")
    for i, p in enumerate(paragraphs, 1):
        print(f"\n段落 {i}:")
        print(f"长度: {len(p)} 字符")
        print(f"内容:\n{p}")
    
    # 测试文本3：包含中文标题的段落
    test_text3 = """【第一章】Java 基础

Java 是一种面向对象的编程语言，具有平台无关性、安全性和健壮性等特点。

【第二章】集合框架

Java 集合框架提供了一套性能优良、使用方便的数据结构和算法。主要包括：
• List：有序可重复的集合
• Set：无序不可重复的集合
• Map：键值对集合"""
    
    print("\n\n测试文本3：包含中文标题的段落")
    print("-" * 40)
    paragraphs = split_text_into_paragraphs(test_text3)
    print(f"段落数量: {len(paragraphs)}")
    for i, p in enumerate(paragraphs, 1):
        print(f"\n段落 {i}:")
        print(f"长度: {len(p)} 字符")
        print(f"内容预览: {p[:80]}...")


def test_knowledge_paragraph_match():
    """测试知识库段落级匹配"""
    print("\n" + "=" * 60)
    print("测试知识库段落级匹配")
    print("=" * 60)
    
    # 获取知识库列表
    knowledge_list = get_knowledge_list()
    print(f"\n知识库文件数量: {len(knowledge_list)}")
    for item in knowledge_list:
        print(f"  - {item['original_name']}")
    
    # 测试段落级匹配
    test_queries = [
        "Spring Boot 启动流程",
        "类加载过程",
        "Java 集合框架"
    ]
    
    for query in test_queries:
        print(f"\n\n查询: {query}")
        print("-" * 40)
        results = match_knowledge_with_interview(query, top_k=2)
        
        if results:
            for i, result in enumerate(results, 1):
                print(f"\n结果 {i}: {result['file_name']}")
                print(f"相似度: {result['similarity']:.4f}")
                print(f"匹配段落数: {len(result.get('matched_paragraphs', []))}")
                
                # 显示匹配的段落
                matched_paragraphs = result.get('matched_paragraphs', [])[:2]
                for j, para in enumerate(matched_paragraphs, 1):
                    print(f"\n  段落 {j} (相似度: {para['similarity']:.4f}):")
                    print(f"  {para['paragraph'][:150]}...")
        else:
            print("  未找到匹配结果")


if __name__ == "__main__":
    test_paragraph_segmentation()
    test_knowledge_paragraph_match()