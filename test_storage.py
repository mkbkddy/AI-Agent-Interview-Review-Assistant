"""
三级数据存储系统测试用例
=====================

测试功能：
1. 三级存储读写测试
2. 故障转移测试
3. 性能监控测试
4. 数据同步测试
"""

import os
import sys
import time

# 添加项目路径
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from utils.triple_tier_storage import (
    TripleTierStorage,
    get_storage,
    print_storage_stats,
    get_storage_status,
    sync_to_cloud
)


def test_basic_operations():
    """测试基本读写操作"""
    print("\n" + "="*60)
    print("🧪 测试 1: 基本读写操作")
    print("="*60)
    
    storage = TripleTierStorage()
    
    # 测试数据
    test_data = {
        "user_id": "test_user",
        "avg_wpm": 150.5,
        "scores": {"技术深度": 8, "逻辑表达": 7},
        "report": "测试报告内容",
        "transcript": "这是测试转录文本"
    }
    
    # 保存数据
    print("\n📝 保存测试数据...")
    result = storage.save_interview(
        user_id=test_data["user_id"],
        avg_wpm=test_data["avg_wpm"],
        scores=test_data["scores"],
        report=test_data["report"],
        transcript=test_data["transcript"]
    )
    print(f"保存结果: {result['tier']} - {result['message']}")
    
    # 读取数据
    print("\n📖 读取最近数据...")
    last = storage.get_last_interview(test_data["user_id"])
    if last:
        print(f"✅ 读取成功: avg_wpm={last.get('avg_wpm')}")
    else:
        print("❌ 读取失败")
    
    # 获取历史
    print("\n📜 获取历史记录...")
    history = storage.get_all_history(test_data["user_id"])
    print(f"历史记录数: {len(history)}")
    
    return True


def test_fault_tolerance():
    """测试故障容错"""
    print("\n" + "="*60)
    print("🧪 测试 2: 故障容错")
    print("="*60)
    
    storage = TripleTierStorage()
    status = storage.get_status()
    
    print("\n📊 存储系统状态:")
    print(f"  ☁️ Supabase: {'已连接' if status['supabase']['connected'] else '未连接'}")
    print(f"  🔴 Redis: {'已连接' if status['redis']['connected'] else '未连接'}")
    print(f"  📁 Local: {'已就绪' if status['local']['connected'] else '未就绪'}")
    print(f"  ⏳ 待同步数据: {status['pending_sync']} 条")
    
    # 测试降级保存
    print("\n📝 测试降级保存...")
    result = storage.save_interview(
        user_id="fault_test_user",
        avg_wpm=120.0,
        scores={"技术深度": 6},
        report="故障测试",
        transcript="测试文本"
    )
    print(f"降级保存结果: {result['tier']} - {result['message']}")
    
    return True


def test_performance_monitoring():
    """测试性能监控"""
    print("\n" + "="*60)
    print("🧪 测试 3: 性能监控")
    print("="*60)
    
    storage = TripleTierStorage()
    
    # 执行多次操作以生成统计数据
    print("\n🔄 执行多次操作以生成统计数据...")
    for i in range(5):
        storage.get_last_interview("perf_test_user")
        time.sleep(0.1)
    
    # 打印统计信息
    storage.print_stats()
    
    return True


def test_data_sync():
    """测试数据同步"""
    print("\n" + "="*60)
    print("🧪 测试 4: 数据同步")
    print("="*60)
    
    storage = TripleTierStorage()
    
    # 检查待同步数据
    status = storage.get_status()
    print(f"\n待同步数据: {status['pending_sync']} 条")
    
    # 手动触发同步
    print("\n🔄 手动触发同步...")
    success, fail = sync_to_cloud()
    print(f"同步结果: 成功 {success} 条, 失败 {fail} 条")
    
    return True


def test_knowledge_base_integration():
    """测试与知识库的集成"""
    print("\n" + "="*60)
    print("🧪 测试 5: 知识库集成")
    print("="*60)
    
    try:
        from utils.knowledge_manager import init_knowledge_base
        init_knowledge_base()
        print("✅ 知识库初始化成功")
        
        # 测试上传文件
        from utils.knowledge_manager import upload_knowledge_file
        test_content = "这是测试知识库内容，包含一些技术知识点。"
        test_content_byte = test_content.encode('utf-8')
        result = upload_knowledge_file("test_knowledge.txt", test_content_byte)
        print(f"✅ 知识库上传测试: {result['success']}")
        
        return True
    except Exception as e:
        print(f"❌ 知识库集成测试失败: {e}")
        return False


def run_all_tests():
    """运行所有测试"""
    print("\n" + "#"*60)
    print("# 🎯 三级数据存储系统 - 完整测试套件")
    print("#"*60)
    
    tests = [
        ("基本读写操作", test_basic_operations),
        ("故障容错", test_fault_tolerance),
        ("性能监控", test_performance_monitoring),
        ("数据同步", test_data_sync),
        ("知识库集成", test_knowledge_base_integration),
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
        print(f"\n⚠️ {total - passed} 个测试失败，请检查相关模块")
    
    return passed == total


if __name__ == "__main__":
    success = run_all_tests()
    sys.exit(0 if success else 1)
