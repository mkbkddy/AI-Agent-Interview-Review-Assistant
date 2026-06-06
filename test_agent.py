#!/usr/bin/env python3
"""
Agent 功能测试
=============

测试具备"思考-行动-观察"循环能力的Agent系统
"""

import os
import sys

# 添加项目路径
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# 解决编码问题
if sys.platform == "win32":
    sys.stdout.reconfigure(encoding="utf-8")
    sys.stderr.reconfigure(encoding="utf-8")

from utils.agent import get_agent, agent_chat


def test_agent_init():
    """测试Agent初始化"""
    print("="*60)
    print("🧪 测试 1: Agent 初始化")
    print("="*60)
    
    try:
        agent = get_agent()
        print("✅ Agent 初始化成功")
        
        # 检查工具注册情况
        tools = agent.tool_registry.list_tools()
        print(f"已注册工具数量: {len(tools)}")
        for tool in tools:
            print(f"  - {tool['name']}: {tool['description']}")
        
        return True
    except Exception as e:
        print(f"❌ Agent 初始化失败: {e}")
        return False


def test_agent_chat():
    """测试Agent聊天功能"""
    print("\n" + "="*60)
    print("🧪 测试 2: Agent 聊天功能")
    print("="*60)
    
    try:
        # 测试问题1：简单问题（不需要调用工具）
        print("\n🔹 测试简单问题:")
        response = agent_chat("你好，我想了解一下面试技巧")
        print(f"用户: 你好，我想了解一下面试技巧")
        print(f"Agent: {response[:200]}...")
        
        # 测试问题2：需要调用知识库的问题
        print("\n🔹 测试知识库查询:")
        response = agent_chat("请搜索一下关于SpringBoot的知识库内容")
        print(f"用户: 请搜索一下关于机器学习的知识库内容")
        print(f"Agent: {response[:300]}...")
        
        return True
    except Exception as e:
        print(f"❌ Agent 聊天失败: {e}")
        import traceback
        traceback.print_exc()
        return False


def test_tool_calls():
    """测试工具调用"""
    print("\n" + "="*60)
    print("🧪 测试 3: 工具调用")
    print("="*60)
    
    agent = get_agent()
    
    # 测试 search_knowledge 工具
    print("\n🔹 测试 search_knowledge 工具:")
    tool = agent.tool_registry.get_tool("search_knowledge")
    if tool:
        result = tool.call(query="Python", max_results=3)
        print(f"工具调用结果: {result['success']}")
        if result['success']:
            print(f"返回结果数量: {len(result['result'])}")
    else:
        print("❌ 工具未找到")
    
    # 测试 get_knowledge_list 工具
    print("\n🔹 测试 get_knowledge_list 工具:")
    tool = agent.tool_registry.get_tool("get_knowledge_list")
    if tool:
        result = tool.call()
        print(f"工具调用结果: {result['success']}")
        print(f"知识库文件数量: {len(result['result'])}")
    else:
        print("❌ 工具未找到")
    
    return True


def test_memory():
    """测试记忆系统"""
    print("\n" + "="*60)
    print("🧪 测试 4: 记忆系统")
    print("="*60)
    
    agent = get_agent()
    
    # 添加短期记忆
    agent.memory.add_short_term("你好", "您好！我是面试教练Agent")
    agent.memory.add_short_term("今天天气怎么样", "今天天气晴朗")
    
    # 获取短期记忆
    short_term = agent.memory.get_short_term()
    print("短期记忆内容:")
    print(short_term)
    
    # 更新长期记忆
    agent.memory.update_long_term("user_name", "张三")
    agent.memory.update_long_term("experience", "3年Python开发经验")
    
    # 获取长期记忆
    user_name = agent.memory.get_long_term("user_name")
    print(f"\n长期记忆 - 用户姓名: {user_name}")
    
    return True


def run_all_tests():
    """运行所有测试"""
    print("\n" + "#"*60)
    print("# 🎯 Agent 功能测试套件")
    print("#"*60)
    
    tests = [
        ("Agent 初始化", test_agent_init),
        ("Agent 聊天功能", test_agent_chat),
        ("工具调用", test_tool_calls),
        ("记忆系统", test_memory),
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
