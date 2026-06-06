#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
向量搜索功能测试脚本
===========================
测试内容：
1. 文本分块功能
2. 向量索引构建
3. 向量搜索功能
4. 混合搜索功能
5. 搜索结果准确性评估
6. 知识库集成测试
"""

import sys
import os
from datetime import datetime

def test_text_chunking():
    """测试文本分块功能"""
    print("\n" + "=" * 60)
    print("【测试 1】文本分块功能测试")
    print("=" * 60)
    
    try:
        from utils.vector_index import TextChunker
        
        chunker = TextChunker(chunk_size=200, overlap=30)
        
        # 测试文本 1: 包含段落的中文文档
        test_text_1 = """Spring Boot 是一个基于 Spring 框架的快速开发框架，用于简化 Spring 应用的创建和部署。
它提供了自动配置、嵌入式服务器、starter 依赖等功能，极大地简化了 Spring 应用的开发。

Spring Boot 的主要特点包括：
1. 自动配置：根据项目依赖自动配置 Spring 应用
2. 嵌入式服务器：内置 Tomcat、Jetty 等服务器
3. starter 依赖：提供一站式的依赖管理方案
4. Actuator：提供生产级监控和管理功能

Spring Boot 的启动流程包括创建 SpringApplication 实例、推断应用类型、加载初始化器和监听器、运行 run 方法等步骤。
在启动过程中，Spring Boot 会准备 Environment、打印 Banner、创建 ApplicationContext、刷新上下文、执行 CommandLineRunner 等。"""
        
        chunks = chunker.chunk_text(test_text_1)
        print(f"\n✓ 测试文本 1: {len(chunks)} 个分块")
        for i, (start, end, content) in enumerate(chunks[:3], 1):
            print(f"  块 {i}: [{start}-{end}] {content[:50]}...")
        
        # 测试文本 2: 简短的段落
        test_text_2 = """Java 虚拟机（JVM）是 Java 程序的运行环境。
它负责加载字节码、解释执行或编译执行、管理内存。

垃圾回收（GC）是 JVM 的重要功能，主要回收堆内存中不再使用的对象。
常见的垃圾收集器包括 Serial GC、Parallel GC、CMS、G1、ZGC 等。"""
        
        chunks = chunker.chunk_text(test_text_2)
        print(f"\n✓ 测试文本 2: {len(chunks)} 个分块")
        
        if chunks:
            print(f"  分块大小范围: {min(len(c[2]) for c in chunks)} - {max(len(c[2]) for c in chunks)} 字符")
        
        return True
        
    except Exception as e:
        print(f"✗ 文本分块测试失败: {e}")
        import traceback
        traceback.print_exc()
        return False


def test_vector_embedding():
    """测试向量嵌入功能"""
    print("\n" + "=" * 60)
    print("【测试 2】向量嵌入功能测试")
    print("=" * 60)
    
    try:
        from utils.vector_index import VectorEmbedder
        
        embedder = VectorEmbedder()
        print(f"\n✓ 向量模型可用性: {'可用' if embedder.is_available() else '未安装（使用基础编码）'}")
        
        # 测试文本编码
        test_texts = [
            "Spring Boot 是 Java 后端开发框架",
            "JVM 垃圾回收机制详解",
            "数据库索引优化方法",
            "REST API 设计原则"
        ]
        
        vectors = []
        for i, text in enumerate(test_texts, 1):
            vector = embedder.encode_single(text)
            vectors.append((text, vector))
            print(f"  文本 {i}: '{text}'")
            print(f"    向量维度: {len(vector)}")
        
        # 测试相似度计算
        if embedder.is_available() and len(vectors) >= 2:
            import numpy as np
            print(f"\n✓ 向量相似度测试:")
            
            # 计算第一句与其他句的相似度
            for i in range(1, len(vectors)):
                sim = np.dot(vectors[0][1], vectors[i][1])
                print(f"  '{vectors[0][0]}' vs '{vectors[i][0]}': {sim:.4f}")
        
        return True
        
    except Exception as e:
        print(f"✗ 向量嵌入测试失败: {e}")
        import traceback
        traceback.print_exc()
        return False


def test_vector_index():
    """测试向量索引构建和搜索"""
    print("\n" + "=" * 60)
    print("【测试 3】向量索引构建与搜索测试")
    print("=" * 60)
    
    try:
        from utils.vector_index import VectorIndexManager
        
        # 准备测试文档
        test_docs = [
            ("doc1", "Spring Boot 入门", 
             "Spring Boot 是一个基于 Spring 框架的快速开发框架，用于简化 Spring 应用的创建和部署。它提供了自动配置、嵌入式服务器、starter 依赖等功能。"),
            ("doc2", "JVM 垃圾回收", 
             "Java 虚拟机（JVM）的垃圾回收（GC）是自动回收堆内存中不再使用的对象。常见的垃圾收集器包括 Serial GC、Parallel GC、G1、ZGC 等。"),
            ("doc3", "数据库优化", 
             "数据库性能优化包括索引优化、查询优化、表结构优化等。正确的索引设计可以显著提升查询性能。"),
            ("doc4", "微服务架构", 
             "微服务架构是一种将单体应用拆分为多个小型服务的架构风格。每个服务独立运行，通过轻量级通信协议交互。"),
            ("doc5", "REST API 设计", 
             "REST API 是一种基于 HTTP 协议的 API 设计风格。它使用标准的 HTTP 方法（GET、POST、PUT、DELETE）来表示资源操作。")
        ]
        
        index = VectorIndexManager()
        doc_count = index.build_index(test_docs)
        
        print(f"\n✓ 索引构建完成: {doc_count} 个文档")
        print(f"✓ 总分块数: {len(index.chunks)}")
        print(f"✓ 向量维度: {index.vector_matrix.shape[1] if index.vector_matrix is not None else 0}")
        
        # 测试搜索
        print(f"\n✓ 搜索测试:")
        test_queries = [
            "Spring Boot 的启动流程是什么？",
            "JVM 如何进行垃圾回收？",
            "数据库如何优化查询性能？"
        ]
        
        for query in test_queries:
            results = index.search(query, top_k=3)
            print(f"\n  查询: '{query}'")
            if results:
                for i, r in enumerate(results, 1):
                    print(f"    {i}. {r.file_name} (相似度: {r.similarity:.4f})")
                    print(f"       内容: {r.content[:60]}...")
            else:
                print("    未找到匹配结果")
        
        # 测试混合搜索
        print(f"\n✓ 混合搜索测试:")
        for query in test_queries[:1]:
            results = index.hybrid_search(query, top_k=3)
            print(f"  查询: '{query}'")
            if results:
                for i, r in enumerate(results, 1):
                    print(f"    {i}. {r.file_name} (混合得分: {r.score:.4f}, 向量: {r.similarity:.4f})")
        
        return True
        
    except Exception as e:
        print(f"✗ 向量索引测试失败: {e}")
        import traceback
        traceback.print_exc()
        return False


def test_knowledge_integration():
    """测试与现有知识库集成"""
    print("\n" + "=" * 60)
    print("【测试 4】知识库集成测试")
    print("=" * 60)
    
    try:
        from utils.knowledge_manager import (
            get_knowledge_list,
            search_knowledge,
            search_knowledge_by_vector,
            build_vector_index_for_knowledge,
            get_vector_index_stats
        )
        
        # 检查知识库状态
        knowledge_list = get_knowledge_list()
        print(f"\n✓ 知识库状态: {len(knowledge_list)} 个文件")
        
        for i, file_info in enumerate(knowledge_list, 1):
            print(f"  {i}. {file_info['original_name']} ({file_info['size'] / 1024:.1f} KB)")
        
        # 测试关键词搜索（传统方法）
        test_query = "Spring Boot"
        print(f"\n✓ 传统关键词搜索 - 查询: '{test_query}'")
        results = search_knowledge(test_query, max_results=3)
        if results:
            for i, r in enumerate(results, 1):
                print(f"  {i}. {r['file_name']}")
                print(f"     关键词匹配: {r.get('score', 0)} 分")
                print(f"     预览: {r.get('preview', '')[:50]}...")
        else:
            print("  未找到匹配（关键词不匹配）")
        
        # 构建向量索引
        print(f"\n✓ 构建向量索引...")
        build_result = build_vector_index_for_knowledge()
        if build_result.get("success"):
            print(f"  ✓ {build_result.get('message', '')}")
        else:
            print(f"  ! {build_result.get('message', '构建失败')}")
            print("  (这可能是因为知识库为空或缺少向量模型)")
        
        # 获取索引统计
        stats = get_vector_index_stats()
        print(f"\n✓ 索引统计:")
        print(f"  - 文档分块数: {stats.get('total_chunks', 0)}")
        print(f"  - 索引文档数: {stats.get('total_documents', 0)}")
        print(f"  - 向量维度: {stats.get('vector_dim', 0)}")
        print(f"  - 向量模型可用: {stats.get('has_vector_model', False)}")
        
        # 测试向量搜索
        print(f"\n✓ 向量搜索测试 - 查询: '{test_query}'")
        vector_results = search_knowledge_by_vector(test_query, max_results=3, use_hybrid=True)
        if vector_results:
            for i, r in enumerate(vector_results, 1):
                print(f"  {i}. {r['file_name']}")
                print(f"     相似度: {r.get('similarity', 0):.4f}")
                print(f"     搜索类型: {r.get('search_type', 'N/A')}")
                print(f"     预览: {r.get('preview', '')[:50]}...")
        else:
            print("  未找到向量匹配结果")
        
        return True
        
    except Exception as e:
        print(f"✗ 知识库集成测试失败: {e}")
        import traceback
        traceback.print_exc()
        return False


def test_search_accuracy():
    """测试搜索结果准确性评估"""
    print("\n" + "=" * 60)
    print("【测试 5】搜索结果准确性评估")
    print("=" * 60)
    
    try:
        from utils.vector_index import VectorIndexManager
        
        # 构建测试用例：已知相关性的文档-查询对
        test_cases = [
            {
                "query": "Spring Boot 启动流程",
                "expected_docs": ["doc1"],  # 期望匹配 doc1
                "description": "Spring Boot 相关查询应该匹配 Spring Boot 文档"
            },
            {
                "query": "垃圾回收 G1 收集器",
                "expected_docs": ["doc2"],  # 期望匹配 doc2
                "description": "JVM GC 查询应该匹配 JVM 文档"
            },
            {
                "query": "索引优化查询性能",
                "expected_docs": ["doc3"],  # 期望匹配 doc3
                "description": "数据库查询应该匹配数据库文档"
            }
        ]
        
        # 准备测试文档
        test_docs = [
            ("doc1", "Spring Boot 指南", 
             "Spring Boot 是一个基于 Spring 框架的快速开发框架。启动流程包括创建 SpringApplication、推断应用类型、加载配置、创建上下文、刷新上下文等步骤。"),
            ("doc2", "JVM GC 详解", 
             "Java 虚拟机的垃圾回收机制包括年轻代的 Minor GC 和老年代的 Major GC/Full GC。常见的垃圾收集器包括 Serial、Parallel、CMS、G1、ZGC。"),
            ("doc3", "数据库优化", 
             "数据库性能优化包括索引设计、SQL 查询优化、表结构优化、缓存策略等。合理使用索引可以避免全表扫描，大幅提升查询速度。"),
            ("doc4", "前端开发", 
             "现代前端开发使用 React、Vue、Angular 等框架。组件化开发、虚拟 DOM、响应式设计是重要概念。")
        ]
        
        index = VectorIndexManager()
        index.build_index(test_docs)
        
        print(f"\n✓ 测试用例: {len(test_cases)} 个")
        correct = 0
        
        for case in test_cases:
            query = case["query"]
            expected = set(case["expected_docs"])
            
            results = index.search(query, top_k=5)
            if not results:
                results = index.hybrid_search(query, top_k=5)
            
            # 检查前3个结果是否包含期望的文档
            top_results = [r.file_id for r in results[:3]] if results else []
            matched = expected.intersection(set(top_results))
            
            status = "✓" if matched else "!"
            if matched:
                correct += 1
            
            print(f"\n  {status} {case['description']}")
            print(f"     查询: '{query}'")
            print(f"     期望匹配: {', '.join(expected)}")
            print(f"     实际匹配前3: {', '.join(top_results)}" if top_results else "     无匹配结果")
            if results:
                for r in results[:3]:
                    print(f"       - {r.file_id} (相似度: {r.similarity:.4f})")
        
        accuracy = correct / len(test_cases) if test_cases else 0
        print(f"\n✓ 整体准确率: {accuracy:.1%} ({correct}/{len(test_cases)})")
        
        return accuracy > 0.5  # 至少 50% 准确率视为通过
        
    except Exception as e:
        print(f"✗ 搜索准确性评估失败: {e}")
        import traceback
        traceback.print_exc()
        return False


def test_edge_cases():
    """测试边界情况"""
    print("\n" + "=" * 60)
    print("【测试 6】边界情况测试")
    print("=" * 60)
    
    try:
        from utils.vector_index import VectorIndexManager
        
        # 测试 1: 空文本
        print(f"\n✓ 空文本处理")
        index = VectorIndexManager()
        try:
            result = index.build_index([("empty", "Empty", "")])
            print(f"  空文本处理: {'成功' if result >= 0 else '失败'}")
        except Exception as e:
            print(f"  空文本: 正确处理异常 {e}")
        
        # 测试 2: 空查询
        print(f"\n✓ 空查询处理")
        test_docs = [("doc1", "测试", "这是一个测试文档，包含一些内容。")]
        index2 = VectorIndexManager()
        index2.build_index(test_docs)
        
        results = index2.search("")
        print(f"  空字符串搜索: 返回 {len(results)} 个结果")
        
        # 测试 3: 非常长的文本
        print(f"\n✓ 长文本处理")
        long_text = "这是一个包含很多内容的文档。" * 100  # 创建约 1500 字符的文本
        results = index2.search(long_text)
        print(f"  长文本查询: 返回 {len(results)} 个结果")
        
        return True
        
    except Exception as e:
        print(f"✗ 边界情况测试失败: {e}")
        import traceback
        traceback.print_exc()
        return False


def main():
    """主测试函数"""
    print("\n" + "🚀" * 30)
    print("向量搜索系统功能测试")
    print("开始时间:", datetime.now().strftime("%Y-%m-%d %H:%M:%S"))
    print("🚀" * 30)
    
    results = []
    
    # 运行所有测试
    results.append(("文本分块", test_text_chunking()))
    results.append(("向量嵌入", test_vector_embedding()))
    results.append(("向量索引", test_vector_index()))
    results.append(("知识库集成", test_knowledge_integration()))
    results.append(("搜索准确性", test_search_accuracy()))
    results.append(("边界情况", test_edge_cases()))
    
    # 汇总结果
    print("\n" + "=" * 60)
    print("📊 测试结果汇总")
    print("=" * 60)
    
    passed = sum(1 for _, result in results if result)
    total = len(results)
    
    for name, result in results:
        status = "✅ 通过" if result else "❌ 失败"
        print(f"  {name}: {status}")
    
    print(f"\n总计: {passed}/{total} 个测试通过 ({passed/total*100:.1f}%)")
    
    if passed == total:
        print("\n🎉 所有测试通过！向量搜索系统运行正常。")
    elif passed > total * 0.7:
        print(f"\n⚠️ 大部分测试通过，但仍有 {total - passed} 个失败，建议检查。")
    else:
        print(f"\n❌ 多个测试失败，请检查系统配置和依赖安装。")
    
    print("\n结束时间:", datetime.now().strftime("%Y-%m-%d %H:%M:%S"))
    return passed == total


if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)
