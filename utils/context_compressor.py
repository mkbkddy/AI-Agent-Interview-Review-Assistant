"""
上下文压缩策略模块
===============

实现基于语义压缩的上下文管理功能，包括：
1. 自动触发机制
2. 手动触发机制
3. 语义压缩算法
4. 核心信息提取
"""

import os
import json
import re
from datetime import datetime
from typing import List, Dict, Any, Optional, Tuple
from dataclasses import dataclass, field
from dotenv import load_dotenv

load_dotenv()


@dataclass
class CompressionConfig:
    """压缩配置"""
    # 自动触发阈值
    auto_trigger_tokens: int = int(os.getenv("CONTEXT_TOKEN_THRESHOLD", 3000))
    auto_trigger_messages: int = int(os.getenv("CONTEXT_MESSAGE_THRESHOLD", 20))
    
    # 压缩参数
    compression_ratio: float = 0.5  # 压缩到原始的50%
    min_messages_keep: int = 3      # 最少保留的消息数
    max_messages_keep: int = 10     # 最多保留的消息数
    
    # 核心信息保留
    preserve_first_messages: int = 2  # 保留开头消息数
    preserve_last_messages: int = 3   # 保留结尾消息数
    preserve_user_profile: bool = True  # 保留用户画像


@dataclass
class CompressedContext:
    """压缩后的上下文"""
    original_count: int = 0
    compressed_count: int = 0
    compressed_messages: List[Dict] = field(default_factory=list)
    summary: str = ""
    timestamp: str = ""
    compression_ratio: float = 0.0
    core_topics: List[str] = field(default_factory=list)


class SemanticCompressor:
    """语义压缩器"""
    
    def __init__(self, config: Optional[CompressionConfig] = None):
        self.config = config or CompressionConfig()
        self.client = None
        self._init_client()
    
    def _init_client(self):
        """初始化API客户端"""
        try:
            from openai import OpenAI
            api_key = os.getenv("DASHSCOPE_API_KEY")
            base_url = os.getenv("DASHSCOPE_BASE_URL", "https://dashscope.aliyuncs.com/compatible-mode/v1")
            
            if api_key:
                self.client = OpenAI(
                    api_key=api_key,
                    base_url=base_url
                )
        except Exception as e:
            print(f"Warning: Could not initialize OpenAI client: {e}")
    
    def count_tokens(self, text: str) -> int:
        """估算token数量（简化版）"""
        # 粗略估算：中文约2字符/token，英文约4字符/token
        chinese_chars = len(re.findall(r'[\u4e00-\u9fff]', text))
        other_chars = len(text) - chinese_chars
        return int(chinese_chars / 2 + other_chars / 4)
    
    def count_messages_tokens(self, messages: List[Dict]) -> int:
        """计算消息列表的总token数"""
        total = 0
        for msg in messages:
            total += self.count_tokens(msg.get("content", ""))
        return total
    
    def should_compress(self, messages: List[Dict]) -> Tuple[bool, str]:
        """
        判断是否需要压缩
        
        Returns:
            (should_compress, reason)
        """
        if not messages:
            return False, "消息列表为空"
        
        # 检查消息数量
        if len(messages) >= self.config.auto_trigger_messages:
            return True, f"消息数量达到阈值 ({len(messages)} >= {self.config.auto_trigger_messages})"
        
        # 检查token数量
        total_tokens = self.count_messages_tokens(messages)
        if total_tokens >= self.config.auto_trigger_tokens:
            return True, f"Token数量达到阈值 ({total_tokens} >= {self.config.auto_trigger_tokens})"
        
        return False, "未达到压缩阈值"
    
    def extract_core_info(self, messages: List[Dict]) -> Dict[str, Any]:
        """提取核心信息"""
        core = {
            "user_profile": {},
            "key_topics": [],
            "important_decisions": [],
            "unresolved_questions": [],
            "context_summary": ""
        }
        
        # 统计用户提到的关键信息
        user_messages = [m["content"] for m in messages if m.get("role") == "user"]
        assistant_messages = [m["content"] for m in messages if m.get("role") == "assistant"]
        
        # 提取用户意图和需求
        if user_messages:
            first_intent = self._extract_intent(user_messages[0])
            core["user_profile"]["first_intent"] = first_intent
        
        # 提取关键话题
        core["key_topics"] = self._extract_topics(messages)
        
        # 生成上下文摘要
        if len(messages) > 5:
            core["context_summary"] = self._generate_summary(messages)
        
        return core
    
    def _extract_intent(self, text: str) -> str:
        """提取用户意图"""
        intents = []
        
        if any(word in text for word in ["复盘", "总结", "分析"]):
            intents.append("面试复盘")
        if any(word in text for word in ["练习", "模拟", "面试"]):
            intents.append("面试练习")
        if any(word in text for word in ["技巧", "建议", "指导"]):
            intents.append("技巧咨询")
        if any(word in text for word in ["评估", "评分", "打分"]):
            intents.append("简历评估")
        
        return ", ".join(intents) if intents else "一般咨询"
    
    def _extract_topics(self, messages: List[Dict]) -> List[str]:
        """提取关键话题"""
        topics = set()
        
        topic_keywords = {
            "Java": ["Java", "java", "JVM"],
            "Spring": ["Spring", "Spring Boot", "SpringMVC"],
            "数据库": ["数据库", "MySQL", "Redis", "SQL"],
            "微服务": ["微服务", "Spring Cloud", "Dubbo"],
            "架构": ["架构", "设计模式", "分布式"],
            "面试": ["面试", "求职", "招聘"],
            "项目": ["项目", "经验", "工作"],
            "简历": ["简历", "CV", "自我介绍"]
        }
        
        all_text = " ".join([m.get("content", "") for m in messages])
        
        for topic, keywords in topic_keywords.items():
            if any(kw in all_text for kw in keywords):
                topics.add(topic)
        
        return list(topics)
    
    def _generate_summary(self, messages: List[Dict]) -> str:
        """生成上下文摘要（基于规则）"""
        summary_parts = []
        
        # 统计对话轮次
        user_count = sum(1 for m in messages if m.get("role") == "user")
        assistant_count = sum(1 for m in messages if m.get("role") == "assistant")
        
        summary_parts.append(f"对话共 {user_count} 轮用户提问，{assistant_count} 轮助手回答。")
        
        # 提取关键话题
        topics = self._extract_topics(messages)
        if topics:
            summary_parts.append(f"涉及话题：{', '.join(topics)}。")
        
        # 最近的问题
        recent_user = [m["content"] for m in messages if m.get("role") == "user"][-1]
        if recent_user:
            summary_parts.append(f"最近问题：{recent_user[:50]}...")
        
        return " ".join(summary_parts)
    
    def compress_with_llm(self, messages: List[Dict], core_info: Dict) -> CompressedContext:
        """使用LLM进行语义压缩"""
        if not self.client:
            return self.compress_with_rules(messages, core_info)
        
        try:
            # 构建压缩提示
            compress_prompt = f"""请将以下对话历史进行语义压缩，保留核心信息和关键上下文。

对话历史：
{self._format_messages(messages)}

已知上下文：
- 用户意图：{core_info.get('user_profile', {}).get('first_intent', '未知')}
- 关键话题：{', '.join(core_info.get('key_topics', []))}

请生成：
1. 压缩后的对话（保留核心内容和关键细节）
2. 对话摘要
3. 保留的关键话题列表

以JSON格式输出：
{{
    "compressed_messages": [...],
    "summary": "...",
    "key_topics": [...]
}}"""

            response = self.client.chat.completions.create(
                model=os.getenv("QWEN_MODEL", "qwen-max"),
                messages=[{"role": "user", "content": compress_prompt}],
                max_tokens=2000,
                temperature=0.3
            )
            
            result_text = response.choices[0].message.content
            
            # 解析结果
            try:
                # 尝试提取JSON
                json_match = re.search(r'\{.*\}', result_text, re.DOTALL)
                if json_match:
                    result = json.loads(json_match.group())
                    return CompressedContext(
                        original_count=len(messages),
                        compressed_count=len(result.get("compressed_messages", [])),
                        compressed_messages=result.get("compressed_messages", []),
                        summary=result.get("summary", ""),
                        timestamp=datetime.now().isoformat(),
                        compression_ratio=len(result.get("compressed_messages", [])) / len(messages),
                        core_topics=result.get("key_topics", [])
                    )
            except json.JSONDecodeError:
                pass
            
            # 如果解析失败，使用规则压缩
            return self.compress_with_rules(messages, core_info)
            
        except Exception as e:
            print(f"LLM compression failed: {e}, falling back to rules")
            return self.compress_with_rules(messages, core_info)
    
    def compress_with_rules(self, messages: List[Dict], core_info: Dict) -> CompressedContext:
        """使用规则进行压缩"""
        if not messages:
            return CompressedContext()
        
        # 计算要保留的消息
        keep_count = max(
            self.config.min_messages_keep,
            min(int(len(messages) * self.config.compression_ratio), self.config.max_messages_keep)
        )
        
        compressed = []
        
        # 1. 保留开头消息（建立上下文）
        compressed.extend(messages[:self.config.preserve_first_messages])
        
        # 2. 保留结尾消息（最新上下文）
        if len(messages) > self.config.preserve_first_messages + self.config.preserve_last_messages:
            # 中间部分按比例抽样
            middle_start = self.config.preserve_first_messages
            middle_end = len(messages) - self.config.preserve_last_messages
            middle_messages = messages[middle_start:middle_end]
            
            # 均匀抽样
            if middle_messages:
                step = max(1, len(middle_messages) // (keep_count - self.config.preserve_first_messages - self.config.preserve_last_messages))
                for i in range(0, len(middle_messages), step):
                    if len(compressed) < keep_count:
                        compressed.append(middle_messages[i])
        
        # 3. 添加结尾消息
        compressed.extend(messages[-self.config.preserve_last_messages:])
        
        # 去除重复（如果去重后数量减少）
        seen = set()
        unique_compressed = []
        for msg in compressed:
            msg_key = f"{msg.get('role')}:{msg.get('content', '')[:100]}"
            if msg_key not in seen:
                seen.add(msg_key)
                unique_compressed.append(msg)
        
        # 生成摘要
        summary = self._generate_summary(messages)
        
        return CompressedContext(
            original_count=len(messages),
            compressed_count=len(unique_compressed),
            compressed_messages=unique_compressed,
            summary=summary,
            timestamp=datetime.now().isoformat(),
            compression_ratio=len(unique_compressed) / len(messages),
            core_topics=core_info.get("key_topics", [])
        )
    
    def compress(self, messages: List[Dict], force_llm: bool = False) -> CompressedContext:
        """压缩对话历史"""
        if not messages:
            return CompressedContext()
        
        # 提取核心信息
        core_info = self.extract_core_info(messages)
        
        # 根据情况选择压缩方式
        if force_llm and self.client:
            return self.compress_with_llm(messages, core_info)
        else:
            return self.compress_with_rules(messages, core_info)
    
    def _format_messages(self, messages: List[Dict]) -> str:
        """格式化消息列表"""
        formatted = []
        for i, msg in enumerate(messages, 1):
            role = msg.get("role", "unknown")
            content = msg.get("content", "")
            formatted.append(f"{i}. [{role}] {content[:200]}..." if len(content) > 200 else f"{i}. [{role}] {content}")
        return "\n".join(formatted)


# ==================== 全局实例 ====================
_compressor_instance = None


def get_compressor() -> SemanticCompressor:
    """获取压缩器单例"""
    global _compressor_instance
    if _compressor_instance is None:
        _compressor_instance = SemanticCompressor()
    return _compressor_instance


# ==================== 便捷函数 ====================
def check_compression_needed(messages: List[Dict]) -> Tuple[bool, str]:
    """检查是否需要压缩"""
    compressor = get_compressor()
    return compressor.should_compress(messages)


def compress_context(messages: List[Dict], use_llm: bool = False) -> CompressedContext:
    """压缩上下文"""
    compressor = get_compressor()
    return compressor.compress(messages, force_llm=use_llm)
