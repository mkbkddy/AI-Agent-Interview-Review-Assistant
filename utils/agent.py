"""
AI Agent 核心模块
===============

实现具备"思考-行动-观察"循环能力的Agent系统。

功能特性：
1. 工具定义与注册
2. 思考-行动-观察循环
3. ReAct风格提示词
4. 记忆管理
5. 多轮对话支持
"""

import os
import json
import re
from datetime import datetime
from typing import List, Dict, Any, Optional, Callable
from dotenv import load_dotenv

load_dotenv()

# ==================== 工具基类 ====================
class Tool:
    """工具类定义"""
    
    def __init__(
        self,
        name: str,
        description: str,
        func: Callable,
        params: List[Dict] = None
    ):
        self.name = name
        self.description = description
        self.func = func
        self.params = params or []
    
    def call(self, **kwargs) -> Dict:
        """调用工具"""
        try:
            result = self.func(**kwargs)
            return {
                "success": True,
                "result": result,
                "tool_name": self.name
            }
        except Exception as e:
            return {
                "success": False,
                "result": str(e),
                "tool_name": self.name
            }


# ==================== Agent 工具集 ====================
class ToolRegistry:
    """工具注册表"""
    
    def __init__(self):
        self.tools: Dict[str, Tool] = {}
    
    def register(self, tool: Tool):
        """注册工具"""
        self.tools[tool.name] = tool
    
    def get_tool(self, name: str) -> Optional[Tool]:
        """获取工具"""
        return self.tools.get(name)
    
    def list_tools(self) -> List[Dict]:
        """列出所有工具"""
        return [
            {
                "name": tool.name,
                "description": tool.description,
                "params": tool.params
            }
            for tool in self.tools.values()
        ]
    
    def get_tool_descriptions(self) -> str:
        """获取工具描述字符串（用于提示词）"""
        descriptions = []
        for name, tool in self.tools.items():
            param_str = ", ".join([f"{p['name']}: {p['description']}" for p in tool.params])
            descriptions.append(f"- {name}: {tool.description} (参数: {param_str})")
        return "\n".join(descriptions)


# ==================== Agent 核心类 ====================
class InterviewAgent:
    """面试教练Agent"""
    
    def __init__(self):
        self.tool_registry = ToolRegistry()
        self.memory = AgentMemory()
        self.max_iterations = int(os.getenv("AGENT_MAX_ITERATIONS", 5))
        self.verbose = os.getenv("AGENT_VERBOSE", "true").lower() == "true"
        
        # 注册工具
        self._register_tools()
    
    def _register_tools(self):
        """注册Agent可用的工具"""
        from utils.knowledge_manager import (
            search_knowledge,
            search_knowledge_hybrid,
            get_knowledge_content,
            get_knowledge_list,
            match_knowledge_with_interview,
            get_relevant_knowledge_summary
        )
        from utils.triple_tier_storage import (
            get_all_history,
            get_user_profile,
            get_history_fragments
        )
        
        # 知识库搜索工具
        self.tool_registry.register(Tool(
            name="search_knowledge",
            description="搜索知识库获取相关技术信息",
            func=search_knowledge,
            params=[
                {"name": "query", "description": "搜索关键词"},
                {"name": "max_results", "description": "最大返回结果数"}
            ]
        ))
        
        # 混合搜索工具
        self.tool_registry.register(Tool(
            name="search_knowledge_hybrid",
            description="使用混合搜索（向量+BM25）搜索知识库",
            func=search_knowledge_hybrid,
            params=[
                {"name": "query", "description": "搜索关键词"},
                {"name": "top_k", "description": "最大返回结果数"}
            ]
        ))
        
        # 获取知识库内容工具
        self.tool_registry.register(Tool(
            name="get_knowledge_content",
            description="获取指定知识库文件的完整内容",
            func=get_knowledge_content,
            params=[
                {"name": "file_id", "description": "知识库文件ID"}
            ]
        ))
        
        # 获取知识库列表工具
        self.tool_registry.register(Tool(
            name="get_knowledge_list",
            description="获取所有知识库文件列表",
            func=get_knowledge_list,
            params=[]
        ))
        
        # 面试匹配工具
        self.tool_registry.register(Tool(
            name="match_knowledge_with_interview",
            description="匹配面试文本与知识库内容",
            func=match_knowledge_with_interview,
            params=[
                {"name": "interview_text", "description": "面试文本"},
                {"name": "top_k", "description": "返回结果数"}
            ]
        ))
        
        # 获取相关知识摘要工具
        self.tool_registry.register(Tool(
            name="get_relevant_knowledge_summary",
            description="获取与面试文本相关的知识库摘要",
            func=get_relevant_knowledge_summary,
            params=[
                {"name": "interview_text", "description": "面试文本"}
            ]
        ))
        
        # 获取历史面试记录工具
        self.tool_registry.register(Tool(
            name="get_interview_history",
            description="获取用户的历史面试记录",
            func=get_all_history,
            params=[
                {"name": "user_id", "description": "用户ID"}
            ]
        ))
        
        # 获取用户画像工具
        self.tool_registry.register(Tool(
            name="get_user_profile",
            description="获取用户的长期画像信息",
            func=get_user_profile,
            params=[
                {"name": "user_id", "description": "用户ID"}
            ]
        ))
        
        # 获取历史片段工具
        self.tool_registry.register(Tool(
            name="get_history_fragments",
            description="获取最近N次面试的转录片段",
            func=get_history_fragments,
            params=[
                {"name": "user_id", "description": "用户ID"},
                {"name": "limit", "description": "获取数量"}
            ]
        ))
    
    def _build_prompt(self, user_input: str, thought_history: List[str] = None) -> str:
        """构建Agent提示词"""
        tools_desc = self.tool_registry.get_tool_descriptions()
        
        prompt = f"""
你是一个专业的AI面试教练Agent，具备思考-行动-观察的循环能力。

## 你的任务
帮助用户分析面试表现，提供专业的反馈和建议。

## 可用工具
{tools_desc}

## 操作格式
请按照以下格式输出你的思考和行动：

思考：[你的分析和思考]
行动：[工具名称](参数1=值1, 参数2=值2)

## 输出要求
1. 如果需要调用工具，请输出"行动：工具名(参数)"格式
2. 如果不需要调用工具，可以直接回答用户
3. 工具调用参数必须使用正确的格式，参数值用引号包围

## 当前对话
用户问题：{user_input}

{"\n".join(thought_history) if thought_history else ""}
"""
        return prompt.strip()
    
    def _parse_action(self, response: str) -> Optional[Dict]:
        """解析模型响应中的工具调用"""
        # 匹配 "行动：工具名(参数)" 格式
        action_pattern = r"行动：(\w+)\((.*?)\)"
        match = re.search(action_pattern, response)
        
        if not match:
            return None
        
        tool_name = match.group(1)
        params_str = match.group(2)
        
        # 解析参数
        params = {}
        param_pattern = r"(\w+)=['\"]([^'\"]+)['\"]"
        for param_match in re.finditer(param_pattern, params_str):
            params[param_match.group(1)] = param_match.group(2)
        
        return {"tool_name": tool_name, "params": params}
    
    def _should_finish(self, response: str) -> bool:
        """判断是否应该结束对话"""
        # 检查是否包含"总结"、"结束"、"答案是"等关键词
        finish_keywords = ["总结", "结束", "答案是", "最终回答", "综上所述"]
        return any(keyword in response for keyword in finish_keywords)
    
    def _summarize(self, observations: List[Dict], user_input: str) -> str:
        """总结所有观察结果，生成最终回答"""
        summary_prompt = f"""
请根据以下观察结果，对用户的问题进行总结回答：

用户问题：{user_input}

观察结果：
{json.dumps(observations, ensure_ascii=False, indent=2)}

请用自然、友好的语言总结回答用户的问题。
"""
        return self._call_llm(summary_prompt)
    
    def _call_llm(self, prompt: str) -> str:
        """调用大模型"""
        from utils.rag_engine import _get_client
        
        client, error = _get_client()
        if error:
            return f"调用大模型失败: {error}"
        
        try:
            response = client.chat.completions.create(
                model=os.getenv("QWEN_MODEL", "qwen-max"),
                messages=[{"role": "user", "content": prompt}],
                max_tokens=2000
            )
            return response.choices[0].message.content or ""
        except Exception as e:
            return f"调用大模型失败: {str(e)}"
    
    def run(self, user_input: str, user_id: str = "default_user") -> str:
        """运行Agent主循环"""
        if self.verbose:
            print(f"🤖 Agent 启动，用户输入: {user_input[:50]}...")
        
        observations = []
        thought_history = []
        iterations = 0
        
        while iterations < self.max_iterations:
            iterations += 1
            
            # 构建提示词
            prompt = self._build_prompt(user_input, thought_history)
            
            # 调用大模型获取思考
            response = self._call_llm(prompt)
            
            if self.verbose:
                print(f"💭 思考 {iterations}: {response}")
            
            # 检查是否应该结束
            if self._should_finish(response):
                if self.verbose:
                    print("🏁 决定结束对话")
                return self._summarize(observations, user_input)
            
            # 解析工具调用
            action = self._parse_action(response)
            
            if not action:
                # 没有工具调用，直接返回响应
                if self.verbose:
                    print("📝 直接回答用户")
                return response
            
            # 执行工具调用
            tool_name = action["tool_name"]
            params = action["params"].copy()  # 复制参数，避免修改原字典
            
            # 只在工具需要 user_id 参数时才添加
            tool = self.tool_registry.get_tool(tool_name)
            if tool:
                tool_param_names = [p["name"] for p in tool.params]
                if "user_id" in tool_param_names and "user_id" not in params and user_id:
                    params["user_id"] = user_id
            
            if not tool:
                observation = f"未知工具: {tool_name}"
                if self.verbose:
                    print(f"❌ {observation}")
            else:
                if self.verbose:
                    print(f"🔧 调用工具: {tool_name}({params})")
                
                result = tool.call(**params)
                observation = json.dumps(result, ensure_ascii=False)
                
                if self.verbose:
                    print(f"👁️ 观察结果: {observation[:100]}...")
            
            # 记录观察
            observations.append({
                "tool_name": tool_name,
                "params": params,
                "result": observation,
                "timestamp": datetime.now().isoformat()
            })
            
            # 更新思考历史
            thought_history.append(f"思考 {iterations}: {response}")
            thought_history.append(f"观察 {iterations}: {observation[:100]}...")
        
        # 达到最大迭代次数，总结回答
        if self.verbose:
            print("⏱️ 达到最大迭代次数，总结回答")
        return self._summarize(observations, user_input)
    
    def chat(self, user_input: str, user_id: str = "default_user") -> str:
        """简单聊天接口"""
        return self.run(user_input, user_id)


# ==================== 记忆系统 ====================
class AgentMemory:
    """Agent记忆系统"""
    
    def __init__(self):
        self.short_term = []  # 当前对话历史
        self.long_term = {}   # 长期记忆（用户画像等）
    
    def add_short_term(self, user_input: str, response: str):
        """添加短期记忆"""
        self.short_term.append({
            "user_input": user_input,
            "response": response,
            "timestamp": datetime.now().isoformat()
        })
        
        # 保持最近10条记录
        if len(self.short_term) > 10:
            self.short_term = self.short_term[-10:]
    
    def get_short_term(self) -> str:
        """获取短期记忆字符串"""
        history = []
        for item in self.short_term:
            history.append(f"用户: {item['user_input']}")
            history.append(f"助手: {item['response']}")
        return "\n".join(history)
    
    def update_long_term(self, key: str, value: Any):
        """更新长期记忆"""
        self.long_term[key] = value
    
    def get_long_term(self, key: str) -> Optional[Any]:
        """获取长期记忆"""
        return self.long_term.get(key)
    
    def clear(self):
        """清空记忆"""
        self.short_term = []
        self.long_term = {}


# ==================== 全局实例 ====================
_agent_instance = None


def get_agent() -> InterviewAgent:
    """获取Agent单例"""
    global _agent_instance
    if _agent_instance is None:
        _agent_instance = InterviewAgent()
    return _agent_instance


# ==================== 便捷函数 ====================
def agent_chat(user_input: str, user_id: str = "default_user") -> str:
    """便捷聊天函数"""
    agent = get_agent()
    return agent.chat(user_input, user_id)
