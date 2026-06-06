#!/usr/bin/env python3
"""测试通义千问 API 是否能成功调用（使用 openai 库）"""

import os
import sys
import codecs
import traceback
from dotenv import load_dotenv
from openai import OpenAI

# 强制 Windows 终端使用 UTF-8 输出
if sys.platform == "win32":
    sys.stdout = codecs.getwriter("utf-8")(sys.stdout.detach())
    sys.stderr = codecs.getwriter("utf-8")(sys.stderr.detach())

DASHSCOPE_BASE_URL = "https://dashscope.aliyuncs.com/compatible-mode/v1"


def safe_api_key(raw: str) -> str:
    """确保 API key 只包含 ASCII 字符"""
    try:
        return raw.encode("ascii").decode("ascii")
    except UnicodeEncodeError:
        raise ValueError("API Key 包含非 ASCII 字符")


def safe_model(name: str) -> str:
    """确保模型名只包含 ASCII 字符"""
    try:
        return name.strip().encode("ascii").decode("ascii")
    except UnicodeEncodeError:
        return "qwen-plus"


def main():
    load_dotenv()

    api_key_raw = os.getenv("DASHSCOPE_API_KEY", "").strip()
    model_name = safe_model(os.getenv("QWEN_MODEL", "qwen-plus"))

    if not api_key_raw:
        print("ERROR: 未配置 DASHSCOPE_API_KEY")
        print("请在 .env 文件中设置 DASHSCOPE_API_KEY=你的API密钥")
        return False

    try:
        api_key = safe_api_key(api_key_raw)
    except ValueError as e:
        print(f"ERROR: {e}")
        return False

    print("=" * 60)
    print("        通义千问 API 测试工具 (openai 库)")
    print("=" * 60)
    print()
    print(f"  - API Key (前10位): {api_key[:10]}...")
    print(f"  - 模型: {model_name}")
    print(f"  - Base URL: {DASHSCOPE_BASE_URL}")
    print()
    print("正在调用通义千问 API...")
    print()

    try:
        client = OpenAI(api_key=api_key, base_url=DASHSCOPE_BASE_URL)

        response = client.chat.completions.create(
            model=model_name,
            messages=[{"role": "user", "content": "Hello, please reply briefly in English."}],
            max_tokens=100,
        )

        reply = response.choices[0].message.content or ""
        print("SUCCESS: API 调用成功！")
        print("模型回复：")
        print("-" * 50)
        print(reply)
        print("-" * 50)
        print()
        print("测试通过！大模型 API 可以正常调用")
        return True

    except Exception as e:
        print(f"ERROR: 调用失败: {type(e).__name__}: {e}")
        print()
        print("详细错误信息：")
        traceback.print_exc()
        return False


if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)
