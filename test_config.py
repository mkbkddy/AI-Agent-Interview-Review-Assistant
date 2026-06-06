#!/usr/bin/env python3
"""验证配置文件是否正确加载"""

import os
import sys

# 强制 Windows 终端使用 UTF-8 输出
if sys.platform == "win32":
    sys.stdout.reconfigure(encoding="utf-8")
    sys.stderr.reconfigure(encoding="utf-8")

from dotenv import load_dotenv

# 加载配置
load_dotenv()

def test_config_loading():
    """测试所有环境变量是否正确加载"""
    print("="*60)
    print("         配置文件加载验证")
    print("="*60)
    print()
    
    config_items = [
        ("SUPABASE_URL", "Supabase 数据库地址"),
        ("SUPABASE_KEY", "Supabase API Key"),
        ("SUPABASE_SERVICE_ROLE_KEY", "Supabase Service Role Key"),
        ("DASHSCOPE_API_KEY", "阿里千问 API Key"),
        ("QWEN_MODEL", "千问模型名称"),
        ("QWEN_VL_MODEL", "千问视觉模型名称"),
        ("EMAIL_SENDER", "邮件发送者"),
        ("EMAIL_AUTH_CODE", "邮件授权码"),
        ("REDIS_HOST", "Redis 主机"),
        ("REDIS_PORT", "Redis 端口"),
        ("APP_VERSION", "应用版本"),
        ("DEBUG", "调试模式"),
    ]
    
    success_count = 0
    warning_count = 0
    
    for key, description in config_items:
        value = os.getenv(key)
        
        if value is None:
            print(f"[FAIL] [{key}] {description}: 未设置")
            warning_count += 1
        elif value == "":
            print(f"[WARN] [{key}] {description}: 为空")
            warning_count += 1
        else:
            # 对于敏感信息，只显示前10位
            display_value = value[:10] + "..." if len(value) > 10 else value
            print(f"[OK]  [{key}] {description}: {display_value}")
            success_count += 1
    
    print()
    print("="*60)
    print(f"结果: {success_count} 项配置已设置, {warning_count} 项需要关注")
    print("="*60)
    
    # 验证关键配置
    critical_keys = ["DASHSCOPE_API_KEY", "QWEN_MODEL"]
    for key in critical_keys:
        if not os.getenv(key):
            print(f"\n[FAIL] 关键配置 [{key}] 未设置，应用可能无法正常运行！")
            return False
    
    print("\n[OK] 配置验证通过！")
    return True

def test_module_imports():
    """测试各个模块是否能正确导入并读取配置"""
    print("\n" + "="*60)
    print("         模块导入测试")
    print("="*60)
    
    modules_to_test = [
        ("utils.rag_engine", ["DASHSCOPE_API_KEY", "QWEN_MODEL", "QWEN_VL_MODEL"]),
        ("utils.triple_tier_storage", ["SUPABASE_URL", "SUPABASE_KEY"]),
        ("utils.knowledge_manager", []),
    ]
    
    all_passed = True
    
    for module_name, expected_vars in modules_to_test:
        try:
            module = __import__(module_name, fromlist=[""])
            print(f"\n[OK] 模块 {module_name} 导入成功")
            
            for var_name in expected_vars:
                if hasattr(module, var_name):
                    value = getattr(module, var_name)
                    print(f"   - {var_name}: {'已设置' if value else '为空'}")
                else:
                    print(f"   - {var_name}: 未定义")
                    
        except Exception as e:
            print(f"[FAIL] 模块 {module_name} 导入失败: {e}")
            all_passed = False
    
    if all_passed:
        print("\n[OK] 所有模块导入测试通过！")
    else:
        print("\n[FAIL] 部分模块导入失败")
    
    return all_passed

if __name__ == "__main__":
    config_ok = test_config_loading()
    modules_ok = test_module_imports()
    
    if config_ok and modules_ok:
        print("\n所有验证通过！")
        sys.exit(0)
    else:
        print("\n部分验证未通过，请检查配置")
        sys.exit(1)
