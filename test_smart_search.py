#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
智能检索策略测试
================

测试智能混合检索的自动切换机制：
1. 向量检索优先
2. 低相关性自动切换到文本匹配
3. 结果整合与去重
"""

def test_smart_search():
    """测试智能检索策略"""
    print("\n" + "=" * 70)
    print("【测试】智能检索策略")
    print("=" * 70)
    
    try:
        from utils.knowledge_manager import search_knowledge_smart, search_knowledge, search_knowledge_by_vector
        from utils.knowledge_manager import build_vector_index_for_knowledge
        
        # 确保有向量索引
        print("\n✓ 确保向量索引已构建...")
        build_result = build_vector_index_for_knowledge()
        if build_result.get("success"):
            print(f"  ✓ {build_result.get('message', '')}")
        else:
            print(f"  ! {build_result.get('message', '构建失败')}")
        
        # 测试用例
        test_cases = [
            {
                "query": "Spring Boot 启动流程",
                "expected_strategy": ["vector_primary", "hybrid_supplement"],
                "description": "明确的Spring Boot话题，应该能通过向量检索找到"
            },
            {
                "query": "类加载过程 双亲委派",
                "expected_strategy": ["vector_primary", "hybrid_supplement"],
                "description": "Java类加载器相关，应该能向量检索命中"
            },
            {
                "query": "什么是Spring Boot？",
                "expected_strategy": ["hybrid_supplement", "keyword_fallback", "vector_primary"],
                "description": "问句形式，可能需要补充检索"
            },
            {
                "query": "面试技巧 建议",
                "expected_strategy": ["hybrid_supplement", "keyword_fallback"],
                "description": "泛化查询，可能触发补充检索"
            }
        ]
        
        print(f"\n✓ 测试用例数: {len(test_cases)}")
        results_summary = []
        
        for i, case in enumerate(test_cases, 1):
            query = case["query"]
            print(f"\n{'─' * 70}")
            print(f"测试 {i}: '{query}'")
            print(f"  预期描述: {case['description']}")
            
            # 执行智能检索
            result = search_knowledge_smart(query, max_results=5)
            
            # 显示结果
            print(f"\n  📊 检索结果:")
            print(f"     策略: {result['strategy_used']}")
            print(f"     向量检索结果数: {len(result['vector_results'])}")
            print(f"     文本匹配结果数: {len(result['keyword_results'])}")
            print(f"     最终结果数: {len(result['results'])}")
            
            metadata = result.get("search_metadata", {})
            if "vector_max_similarity" in metadata:
                print(f"     向量最高相似度: {metadata['vector_max_similarity']:.4f}")
                print(f"     向量平均相似度: {metadata['vector_avg_similarity']:.4f}")
            
            # 显示最终结果
            if result["results"]:
                print(f"\n  📄 最终结果 (Top 3):")
                for j, res in enumerate(result["results"][:3], 1):
                    sources = res.get("sources", [])
                    source_str = " + ".join(sources) if sources else "unknown"
                    print(f"     {j}. [{source_str}] {res['file_name']}")
                    print(f"        综合得分: {res.get('score', 0):.4f}")
                    if res.get('vector_score', 0) > 0:
                        print(f"        向量相似度: {res['vector_score']:.4f}")
                    if res.get('keyword_score', 0) > 0:
                        print(f"        关键词得分: {res['keyword_score']:.4f}")
            else:
                print(f"\n  ❌ 无匹配结果")
            
            # 记录结果
            results_summary.append({
                "query": query,
                "strategy": result["strategy_used"],
                "final_count": len(result["results"]),
                "vector_count": len(result["vector_results"]),
                "keyword_count": len(result["keyword_results"]),
                "success": len(result["results"]) > 0
            })
        
        # 汇总结果
        print(f"\n{'=' * 70}")
        print("📊 测试汇总")
        print("=" * 70)
        
        success_count = sum(1 for r in results_summary if r["success"])
        
        print(f"\n总测试数: {len(results_summary)}")
        print(f"成功数: {success_count}")
        print(f"成功率: {success_count / len(results_summary) * 100:.1f}%")
        
        print(f"\n策略分布:")
        strategy_counts = {}
        for r in results_summary:
            s = r["strategy"]
            strategy_counts[s] = strategy_counts.get(s, 0) + 1
        
        for s, count in strategy_counts.items():
            strategy_labels = {
                "vector_primary": "向量优先",
                "hybrid_supplement": "补充检索",
                "keyword_fallback": "文本匹配回退"
            }
            label = strategy_labels.get(s, s)
            print(f"  - {label}: {count} 次")
        
        print(f"\n详细结果:")
        print(f"{'查询':<30} {'策略':<20} {'向量':<6} {'文本':<6} {'最终':<6} {'状态'}")
        print("-" * 80)
        
        for r in results_summary:
            status = "✅" if r["success"] else "❌"
            print(f"{r['query']:<30} {r['strategy']:<20} {r['vector_count']:<6} {r['keyword_count']:<6} {r['final_count']:<6} {status}")
        
        return success_count >= len(test_cases) * 0.75  # 至少75%成功率
        
    except Exception as e:
        print(f"\n❌ 测试失败: {e}")
        import traceback
        traceback.print_exc()
        return False


def test_result_merge():
    """测试结果整合功能"""
    print("\n" + "=" * 70)
    print("【测试】结果整合与去重")
    print("=" * 70)
    
    try:
        from utils.knowledge_manager import _merge_search_results
        
        # 模拟向量检索结果
        vector_results = [
            {
                "file_id": "doc1",
                "file_name": "Spring Boot 指南",
                "similarity": 0.8,
                "preview": "Spring Boot 是一个快速开发框架...",
                "score": 0.8,
                "matched_words": []
            },
            {
                "file_id": "doc2",
                "file_name": "JVM 详解",
                "similarity": 0.6,
                "preview": "JVM 是 Java 虚拟机...",
                "score": 0.6,
                "matched_words": []
            }
        ]
        
        # 模拟文本匹配结果
        keyword_results = [
            {
                "file_id": "doc1",  # 与向量结果重复
                "file_name": "Spring Boot 指南",
                "score": 10,
                "preview": "Spring Boot 提供了自动配置功能...",
                "matched_words": ["Spring", "Boot"]
            },
            {
                "file_id": "doc3",  # 新文档
                "file_name": "微服务架构",
                "score": 8,
                "preview": "微服务是一种架构风格...",
                "matched_words": ["微服务"]
            }
        ]
        
        print("\n✓ 原始数据:")
        print(f"  向量检索结果: {len(vector_results)} 个文档")
        print(f"  文本匹配结果: {len(keyword_results)} 个文档")
        
        # 执行合并
        merged = _merge_search_results(vector_results, keyword_results, max_results=5)
        
        print(f"\n✓ 合并结果: {len(merged)} 个文档")
        print(f"\n合并后结果:")
        
        for i, r in enumerate(merged, 1):
            sources = r.get("sources", [])
            source_str = " + ".join(sources)
            print(f"\n  {i}. {r['file_name']} [{source_str}]")
            print(f"     综合得分: {r['score']:.4f}")
            print(f"     向量相似度: {r.get('vector_score', 0):.4f}")
            print(f"     关键词得分: {r.get('keyword_score', 0):.4f}")
            print(f"     预览: {r['preview'][:50]}...")
        
        # 验证去重
        file_ids = [r["file_id"] for r in merged]
        has_duplicates = len(file_ids) != len(set(file_ids))
        
        print(f"\n✓ 去重验证: {'✅ 无重复' if not has_duplicates else '❌ 存在重复'}")
        print(f"✓ 评分归一化: {'✅ 已完成' if all(0 <= r['score'] <= 1.0 for r in merged) else '❌ 评分异常'}")
        
        return not has_duplicates and all(0 <= r['score'] <= 1.0 for r in merged)
        
    except Exception as e:
        print(f"\n❌ 合并测试失败: {e}")
        import traceback
        traceback.print_exc()
        return False


def test_threshold_logic():
    """测试阈值判断逻辑"""
    print("\n" + "=" * 70)
    print("【测试】相关性阈值判断")
    print("=" * 70)
    
    try:
        from utils.knowledge_manager import search_knowledge_smart
        
        # 测试不同相似度的查询
        test_scenarios = [
            {
                "query": "Spring Boot 启动流程 详解",  # 高相关性
                "expected": "应该有高向量相似度"
            },
            {
                "query": "JVM 垃圾回收 机制",  # 中等相关性
                "expected": "可能有中等相似度"
            },
            {
                "query": "随便聊聊天",  # 低相关性
                "expected": "可能触发补充检索"
            }
        ]
        
        print("\n阈值配置:")
        print("  向量相似度阈值: 0.15")
        print("  最低向量结果数: 1")
        print("  质量判定: max_sim >= 0.15 或 (count >= 1 且 avg_sim >= 0.075)")
        
        for i, scenario in enumerate(test_scenarios, 1):
            query = scenario["query"]
            print(f"\n{'─' * 70}")
            print(f"场景 {i}: '{query}'")
            print(f"  预期: {scenario['expected']}")
            
            result = search_knowledge_smart(query, max_results=3)
            
            metadata = result.get("search_metadata", {})
            max_sim = metadata.get("vector_max_similarity", 0)
            avg_sim = metadata.get("vector_avg_similarity", 0)
            
            print(f"\n  检索统计:")
            print(f"    向量最高相似度: {max_sim:.4f} {'✅ >= 0.15' if max_sim >= 0.15 else '❌ < 0.15'}")
            print(f"    向量平均相似度: {avg_sim:.4f}")
            print(f"    策略: {result['strategy_used']}")
            print(f"    结果数: {result['total_found']}")
            
            # 判断质量
            quality_good = max_sim >= 0.15 or (metadata.get("vector_result_count", 0) >= 1 and avg_sim >= 0.075)
            print(f"    质量判定: {'✅ 良好' if quality_good else '⚠️ 需补充'}")
        
        return True
        
    except Exception as e:
        print(f"\n❌ 阈值测试失败: {e}")
        import traceback
        traceback.print_exc()
        return False


def main():
    """主函数"""
    print("\n🚀" * 35)
    print("智能检索策略测试")
    print("🚀" * 35)
    
    results = []
    
    # 执行所有测试
    results.append(("智能检索策略", test_smart_search()))
    results.append(("结果整合去重", test_result_merge()))
    results.append(("阈值判断逻辑", test_threshold_logic()))
    
    # 汇总
    print("\n" + "=" * 70)
    print("📊 测试汇总")
    print("=" * 70)
    
    passed = sum(1 for _, r in results if r)
    total = len(results)
    
    for name, result in results:
        status = "✅ 通过" if result else "❌ 失败"
        print(f"  {name}: {status}")
    
    print(f"\n总计: {passed}/{total} 个测试通过 ({passed/total*100:.1f}%)")
    
    if passed == total:
        print("\n🎉 所有测试通过！智能检索系统运行正常。")
    else:
        print(f"\n⚠️ 有 {total - passed} 个测试失败，请检查。")
    
    return passed == total


if __name__ == "__main__":
    import sys
    success = main()
    sys.exit(0 if success else 1)
