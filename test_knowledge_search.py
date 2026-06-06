#!/usr/bin/env python3
"""
知识库检索功能测试
===================

测试知识库上传、检索、混合搜索等功能
"""

import os
import sys
import tempfile

# 添加项目路径
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# 解决编码问题
if sys.platform == "win32":
    sys.stdout.reconfigure(encoding="utf-8")
    sys.stderr.reconfigure(encoding="utf-8")

from utils.knowledge_manager import (
    init_knowledge_base,
    upload_knowledge_file,
    get_knowledge_list,
    get_knowledge_content,
    search_knowledge,
    delete_knowledge_file,
    match_knowledge_with_interview,
    get_relevant_knowledge_summary
)
from utils.hybrid_search import (
    init_hybrid_search,
    add_document,
    build_hybrid_index,
    hybrid_search
)


def test_knowledge_upload():
    """测试知识库上传功能"""
    print("="*60)
    print("📤 测试 1: 上传测试文件到知识库")
    print("="*60)
    
    # 创建测试文件
    test_files = [
        ("python基础知识.txt", """Python 是一种高级编程语言，具有简洁的语法和强大的功能。
Python支持多种编程范式，包括面向对象、函数式和过程式编程。
常用的Python库包括NumPy、Pandas、TensorFlow等。
Python在数据科学、机器学习和Web开发领域广泛应用。"""),
        
        ("机器学习入门.txt", """机器学习是人工智能的一个分支，使计算机能够从数据中学习。
常见的机器学习算法包括线性回归、决策树、随机森林和神经网络。
监督学习、无监督学习和强化学习是机器学习的三大类型。
深度学习是机器学习的一个子领域，使用多层神经网络。"""),
        
        ("面试技巧.txt", """面试前要充分准备，了解公司背景和职位要求。
常见的面试问题包括自我介绍、项目经验和技术问题。
技术面试通常包括算法题、系统设计和技术栈相关问题。
回答问题时要使用STAR法则：情境、任务、行动、结果。"""),
        
        ("深度学习实战.txt", """深度学习使用神经网络进行特征学习和模式识别。
卷积神经网络(CNN)主要用于图像处理任务。
循环神经网络(RNN)适合处理序列数据。
Transformer架构在NLP领域取得了革命性突破。
PyTorch和TensorFlow是主流的深度学习框架。"""),
        
        ("简历优化指南.txt", """简历应该简洁明了，突出重点。
项目经验要量化成果，使用具体数字。
技术栈部分要列出熟悉的编程语言和工具。
避免冗长描述，保持每段不超过3-4行。""")
    ]
    
    uploaded_ids = []
    
    try:
        init_knowledge_base()
        
        for filename, content in test_files:
            # 创建临时文件
            with tempfile.NamedTemporaryFile(mode='w', suffix='.txt', delete=False, encoding='utf-8') as f:
                f.write(content)
                temp_path = f.name
            
            try:
                # 读取文件内容
                with open(temp_path, 'rb') as f:
                    file_content = f.read()
                
                # 上传到知识库
                result = upload_knowledge_file(filename, file_content)
                print(f"📄 {filename}: {result['message']}")
                
                if result['success'] and 'file_id' in result:
                    uploaded_ids.append(result['file_id'])
            finally:
                # 删除临时文件
                os.unlink(temp_path)
        
        print(f"\n✅ 成功上传 {len(uploaded_ids)} 个测试文件")
        return uploaded_ids
    
    except Exception as e:
        print(f"❌ 上传失败: {e}")
        import traceback
        traceback.print_exc()
        return []


def test_knowledge_search():
    """测试知识库搜索功能"""
    print("\n" + "="*60)
    print("🔍 测试 2: 知识库关键词搜索")
    print("="*60)
    
    test_queries = [
        "Python",
        "机器学习",
        "面试",
        "深度学习",
        "神经网络",
        "简历"
    ]
    
    for query in test_queries:
        print(f"\n🔹 搜索: '{query}'")
        results = search_knowledge(query)
        if results:
            print(f"   找到 {len(results)} 个结果:")
            for i, res in enumerate(results, 1):
                print(f"   {i}. {res['file_name']}")
                print(f"      相似度: {res.get('similarity', 'N/A')}")
                print(f"      预览: {res['preview'][:50]}...")
        else:
            print(f"   ❌ 未找到匹配结果")


def test_hybrid_search():
    """测试混合搜索功能"""
    print("\n" + "="*60)
    print("🔄 测试 3: Hybrid Search 混合搜索")
    print("="*60)
    
    # 初始化混合搜索
    init_hybrid_search()
    
    # 添加测试文档
    test_docs = [
        ("Python 基础教程", "Python 是一种简单易学的编程语言，适合初学者入门。"),
        ("机器学习入门", "机器学习算法可以分为监督学习和无监督学习两大类。"),
        ("深度学习实战", "深度学习使用多层神经网络来学习数据的特征表示。"),
        ("面试技巧", "技术面试中常问的数据结构和算法问题包括链表、树、图等。"),
        ("简历编写指南", "简历应该突出你的核心竞争力和项目经验。")
    ]
    
    for title, content in test_docs:
        add_document(content, {"source": title})
    
    # 构建索引
    print("📊 构建混合索引...")
    build_result = build_hybrid_index()
    print(f"   {build_result['message']}")
    
    # 测试不同搜索模式
    test_queries = [
        "Python 编程",
        "神经网络",
        "面试准备",
        "机器学习算法"
    ]
    
    modes = ["hybrid", "bm25", "vector"]
    
    for query in test_queries:
        print(f"\n🔹 查询: '{query}'")
        
        for mode in modes:
            results = hybrid_search(query, top_k=3, mode=mode)
            if results['success'] and results['results']:
                print(f"   {mode}: 找到 {len(results['results'])} 个结果")
                for res in results['results'][:2]:
                    print(f"      - {res['score']:.3f}: {res['content'][:30]}...")
            else:
                print(f"   {mode}: 未找到结果")


def test_knowledge_matching():
    """测试面试文本与知识库匹配"""
    print("\n" + "="*60)
    print("🎯 测试 4: 面试文本与知识库匹配")
    print("="*60)
    
    # 模拟面试文本
    interview_text = """面试官问了我关于Python机器学习的问题，包括神经网络和深度学习的基础知识。
我回答了监督学习和无监督学习的区别，以及卷积神经网络的应用。
面试官还问了一些关于面试技巧和简历优化的建议。"""
    
    print("📝 面试文本摘要:")
    print(f"   {interview_text[:80]}...")
    
    # 测试匹配功能
    print("\n🔹 匹配结果:")
    # 使用较低的阈值来测试
    matched = match_knowledge_with_interview(interview_text, top_k=5, similarity_threshold=0.05)
    
    if matched:
        print(f"   找到 {len(matched)} 个匹配结果:")
        for i, match in enumerate(matched, 1):
            print(f"   {i}. {match['file_name']}")
            print(f"      相似度: {match['similarity']:.4f}")
            if match.get('key_fragments'):
                print(f"      关键片段: {match['key_fragments'][0][:50]}...")
            else:
                print(f"      预览: {match.get('content_preview', '')[:50]}...")
    else:
        print("   ❌ 未找到匹配的知识库内容")
        # 调试：检查知识库是否有内容
        knowledge_list = get_knowledge_list()
        print(f"\n   📋 当前知识库文件列表 ({len(knowledge_list)} 个):")
        for f in knowledge_list:
            content = get_knowledge_content(f['file_id'])
            print(f"      - {f['original_name']}: {len(content)} 字符")
    
    # 测试知识摘要生成
    print("\n🔹 知识摘要:")
    summary = get_relevant_knowledge_summary(interview_text)
    print(f"   {summary[:150]}...")


def test_cleanup(uploaded_ids):
    """清理测试数据"""
    print("\n" + "="*60)
    print("🗑️ 测试 5: 清理测试数据")
    print("="*60)
    
    try:
        for file_id in uploaded_ids:
            result = delete_knowledge_file(file_id)
            status = "✅ 成功" if result else "❌ 失败"
            print(f"{status}: 删除文件 {file_id}")
        
        print("\n✅ 清理完成")
        return True
    except Exception as e:
        print(f"❌ 清理失败: {e}")
        import traceback
        traceback.print_exc()
        return False


def run_all_tests():
    """运行所有测试"""
    print("\n" + "#"*60)
    print("# 📚 知识库检索功能测试套件")
    print("#"*60)
    
    # 测试上传
    uploaded_ids = test_knowledge_upload()
    
    if not uploaded_ids:
        print("\n❌ 上传测试失败，无法继续其他测试")
        return False
    
    # 测试关键词搜索
    test_knowledge_search()
    
    # 测试混合搜索
    test_hybrid_search()
    
    # 测试匹配功能
    test_knowledge_matching()
    
    # 清理测试数据
    test_cleanup(uploaded_ids)
    
    print("\n" + "#"*60)
    print("# 🎉 所有测试完成！")
    print("#"*60)
    
    return True


if __name__ == "__main__":
    success = run_all_tests()
    sys.exit(0 if success else 1)
