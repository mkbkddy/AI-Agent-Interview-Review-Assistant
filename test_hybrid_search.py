#!/usr/bin/env python3
"""
Hybrid Search 功能测试
======================

测试功能：
1. 引擎初始化
2. 文档添加
3. 索引构建
4. 混合搜索（三种模式）
5. 索引持久化
6. 性能测试
"""

import os
import sys
import time

# 添加项目路径
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# 解决编码问题
if sys.platform == "win32":
    sys.stdout.reconfigure(encoding="utf-8")
    sys.stderr.reconfigure(encoding="utf-8")

from utils.hybrid_search import (
    init_hybrid_search,
    add_document,
    add_document_from_file,
    build_hybrid_index,
    hybrid_search,
    get_engine_stats,
    clear_hybrid_index,
    search_knowledge_hybrid,
    search_knowledge_bm25,
    search_knowledge_vector
)


def test_init():
    """测试引擎初始化"""
    print("="*60)
    print("🧪 测试 1: 引擎初始化")
    print("="*60)
    
    result = init_hybrid_search()
    print(f"初始化结果: {'成功' if result['success'] else '失败'}")
    if result.get("message"):
        print(f"消息: {result['message']}")
    
    return result["success"]


def test_add_documents():
    """测试文档添加"""
    print("\n" + "="*60)
    print("🧪 测试 2: 文档添加")
    print("="*60)
    
    # 测试添加文本文档
    test_docs = [
        {
            "content": "Python 是一种高级编程语言，广泛应用于数据科学、机器学习和 Web 开发。Python 具有简洁的语法和强大的生态系统。",
            "metadata": {"category": "编程语言", "source": "Python 官方文档"}
        },
        {
            "content": "机器学习是人工智能的一个分支，使用算法从数据中学习模式。监督学习、无监督学习和强化学习是常见的机器学习类型。",
            "metadata": {"category": "人工智能", "source": "机器学习入门"}
        },
        {
            "content": "深度学习是机器学习的一个子领域，使用多层神经网络来模拟人脑的学习过程。深度学习在图像识别和自然语言处理方面取得了巨大成功。",
            "metadata": {"category": "人工智能", "source": "深度学习实战"}
        },
        {
            "content": "向量数据库是一种专门用于存储和检索高维向量的数据库。FAISS 是 Facebook 开发的一个高效的向量搜索库。",
            "metadata": {"category": "数据库", "source": "FAISS 文档"}
        },
        {
            "content": "RAG（Retrieval-Augmented Generation）是一种将检索与生成相结合的 AI 技术，能够提高大模型回答的准确性和可靠性。",
            "metadata": {"category": "人工智能", "source": "RAG 论文"}
        }
    ]
    
    success_count = 0
    for i, doc in enumerate(test_docs, 1):
        result = add_document(doc["content"], doc["metadata"])
        print(f"文档 {i}: {'✅ 成功' if result['success'] else '❌ 失败'}")
        if result["success"]:
            success_count += 1
    
    print(f"\n添加结果: {success_count}/{len(test_docs)} 成功")
    return success_count == len(test_docs)


def test_build_index():
    """测试索引构建"""
    print("\n" + "="*60)
    print("🧪 测试 3: 索引构建")
    print("="*60)
    
    result = build_hybrid_index()
    print(f"构建结果: {'✅ 成功' if result['success'] else '❌ 失败'}")
    
    if result["success"]:
        print(f"总文档数: {result.get('total_docs', 0)}")
        print(f"切分片段: {result.get('split_chunks', 0)}")
        print(f"构建时间: {result.get('build_time', 'N/A')}")
        print(f"BM25 权重: {result.get('bm25_weight', 0)}")
        print(f"向量权重: {result.get('vector_weight', 0)}")
        if "save_message" in result:
            print(f"保存消息: {result['save_message']}")
    else:
        print(f"错误: {result.get('message', '未知错误')}")
    
    return result["success"]


def test_search():
    """测试搜索功能"""
    print("\n" + "="*60)
    print("🧪 测试 4: 搜索功能")
    print("="*60)
    
    test_queries = [
        "Python 机器学习",
        "深度学习神经网络",
        "向量数据库 FAISS",
        "RAG 检索增强生成"
    ]
    
    search_modes = [
        ("hybrid", "混合搜索"),
        ("bm25", "BM25 搜索"),
        ("vector", "向量搜索")
    ]
    
    for query in test_queries:
        print(f"\n🔍 查询: '{query}'")
        
        for mode, mode_name in search_modes:
            result = hybrid_search(query, top_k=3, mode=mode)
            
            if result["success"]:
                count = len(result.get("results", []))
                print(f"   {mode_name}: {count} 个结果")
                
                for i, res in enumerate(result["results"], 1):
                    content_preview = res["content"][:80] + "..." if len(res["content"]) > 80 else res["content"]
                    source = res["metadata"].get("source", "未知来源")
                    print(f"      {i}. {content_preview}")
                    print(f"         来源: {source}")
            else:
                print(f"   {mode_name}: ❌ 失败 - {result.get('message', '')}")
    
    return True


def test_convenience_functions():
    """测试便捷搜索函数"""
    print("\n" + "="*60)
    print("🧪 测试 5: 便捷搜索函数")
    print("="*60)
    
    query = "机器学习"
    
    print(f"🔍 查询: '{query}'")
    
    # 测试混合搜索
    results = search_knowledge_hybrid(query, top_k=2)
    print(f"\n混合搜索结果: {len(results)} 条")
    
    # 测试 BM25 搜索
    results = search_knowledge_bm25(query, top_k=2)
    print(f"BM25 搜索结果: {len(results)} 条")
    
    # 测试向量搜索
    results = search_knowledge_vector(query, top_k=2)
    print(f"向量搜索结果: {len(results)} 条")
    
    return True


def test_stats():
    """测试统计信息"""
    print("\n" + "="*60)
    print("🧪 测试 6: 统计信息")
    print("="*60)
    
    stats = get_engine_stats()
    print("引擎统计:")
    for key, value in stats.items():
        print(f"  {key}: {value}")
    
    return True


def test_performance():
    """测试性能"""
    print("\n" + "="*60)
    print("🧪 测试 7: 性能测试")
    print("="*60)
    
    queries = [
        "Python 编程",
        "深度学习",
        "向量数据库",
        "RAG 技术",
        "机器学习算法"
    ]
    
    # 测试搜索性能
    start_time = time.time()
    for query in queries:
        hybrid_search(query, top_k=3, mode="hybrid")
    
    total_time = time.time() - start_time
    avg_time = total_time / len(queries)
    
    print(f"搜索 {len(queries)} 个查询的总时间: {total_time:.3f} 秒")
    print(f"平均搜索时间: {avg_time:.3f} 秒/查询")
    
    return True


def test_clear_index():
    """测试清空索引"""
    print("\n" + "="*60)
    print("🧪 测试 8: 清空索引")
    print("="*60)
    
    result = clear_hybrid_index()
    print(f"清空结果: {'✅ 成功' if result['success'] else '❌ 失败'}")
    
    # 验证索引已清空
    stats = get_engine_stats()
    print(f"清空后文档数: {stats.get('document_count', 0)}")
    
    return result["success"]


def run_all_tests():
    """运行所有测试"""
    print("\n" + "#"*60)
    print("# 🎯 Hybrid Search 功能测试套件")
    print("#"*60)
    
    tests = [
        ("引擎初始化", test_init),
        ("文档添加", test_add_documents),
        ("索引构建", test_build_index),
        ("搜索功能", test_search),
        ("便捷函数", test_convenience_functions),
        ("统计信息", test_stats),
        ("性能测试", test_performance),
        ("清空索引", test_clear_index),
    ]
    
    results = []
    for name, test_func in tests:
        try:
            result = test_func()
            results.append((name, result))
        except Exception as e:
            print(f"\n❌ 测试 '{name}' 发生异常: {e}")
            import traceback
            traceback.print_exc()
            results.append((name, False))
    
    # 打印总结
    print("\n" + "#"*60)
    print("# 📊 测试总结")
    print("#"*60)
    
    passed = sum(1 for _, r in results if r)
    total = len(results)
    
    for name, result in results:
        status = "✅ 通过" if result else "❌ 失败"
        print(f"  {status} - {name}")
    
    print(f"\n总计: {passed}/{total} 测试通过")
    
    if passed == total:
        print("\n🎉 所有测试通过！")
    else:
        print(f"\n⚠️ {total - passed} 个测试失败")
    
    return passed == total


if __name__ == "__main__":
    success = run_all_tests()
    sys.exit(0 if success else 1)
