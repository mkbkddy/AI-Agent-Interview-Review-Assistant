import os
import sys
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from utils.rag_engine import (
    _validate_image_file,
    _get_mime_type,
    process_image_with_vision,
    process_jd_to_context,
    SUPPORTED_IMAGE_FORMATS,
    MAX_IMAGE_SIZE_MB,
    MAX_IMAGE_SIZE_BYTES
)


def test_image_validation():
    """测试图片验证功能"""
    print("=" * 60)
    print("📷 测试图片验证功能")
    print("=" * 60)

    # 测试不存在的文件
    print("\n1. 测试不存在的文件:")
    valid, info = _validate_image_file("nonexistent.jpg")
    print(f"   结果: {'✅ 有效' if valid else '❌ 无效'}")
    print(f"   信息: {info}")

    # 测试空文件
    print("\n2. 测试空文件:")
    with open("test_empty.jpg", "wb") as f:
        f.write(b"")
    valid, info = _validate_image_file("test_empty.jpg")
    print(f"   结果: {'✅ 有效' if valid else '❌ 无效'}")
    print(f"   信息: {info}")
    os.remove("test_empty.jpg")

    # 测试支持的格式列表
    print("\n3. 支持的图片格式:")
    print(f"   {', '.join(sorted(SUPPORTED_IMAGE_FORMATS))}")

    # 测试文件大小限制
    print("\n4. 文件大小限制:")
    print(f"   最大支持: {MAX_IMAGE_SIZE_MB}MB ({MAX_IMAGE_SIZE_BYTES} bytes)")

    print("\n✅ 图片验证测试完成！")


def test_mime_types():
    """测试 MIME 类型获取"""
    print("\n" + "=" * 60)
    print("📄 测试 MIME 类型获取")
    print("=" * 60)

    test_cases = [
        ("test.jpg", "image/jpeg"),
        ("test.jpeg", "image/jpeg"),
        ("test.png", "image/png"),
        ("test.webp", "image/webp"),
        ("test.bmp", "image/bmp"),
        ("test.gif", "image/gif"),
        ("test.txt", "image/jpeg"),  # 未知格式默认返回 jpeg
    ]

    for file_path, expected in test_cases:
        result = _get_mime_type(file_path)
        status = "✅" if result == expected else "❌"
        print(f"   {status} {file_path:15} -> {result} (预期: {expected})")

    print("\n✅ MIME 类型测试完成！")


def test_real_image_parsing():
    """测试真实图片解析（需要有测试图片）"""
    print("\n" + "=" * 60)
    print("🔍 测试真实图片解析")
    print("=" * 60)

    # 查找测试图片
    test_images = []
    for ext in SUPPORTED_IMAGE_FORMATS:
        pattern = f"*.{ext}"
        for f in os.listdir("."):
            if f.lower().endswith(f".{ext}"):
                test_images.append(f)
                break  # 每种格式取一个

    if not test_images:
        print("   ⚠️ 未找到测试图片，跳过此测试")
        print("   💡 请在项目目录下放置一些测试图片（JPG/PNG等）")
        return

    print(f"\n找到 {len(test_images)} 张测试图片:")
    for img_path in test_images:
        print(f"\n   📸 测试: {img_path}")

        # 验证图片
        valid, info = _validate_image_file(img_path)
        print(f"   验证结果: {'✅ 通过' if valid else '❌ 失败'}")
        print(f"   图片信息: {info}")

        # 解析图片
        if valid:
            print("   正在解析...")
            result = process_jd_to_context(img_path)
            if result.startswith("图片解析失败") or result.startswith("图片验证失败"):
                print(f"   ❌ 解析失败: {result}")
            else:
                print(f"   ✅ 解析成功！提取到 {len(result)} 个字符")
                print(f"   内容预览: {result[:100]}...")

    print("\n✅ 真实图片解析测试完成！")


def test_api_key_check():
    """测试 API Key 检查"""
    from dotenv import load_dotenv
    load_dotenv()

    print("\n" + "=" * 60)
    print("🔑 测试 API Key 配置")
    print("=" * 60)

    api_key = os.getenv("DASHSCOPE_API_KEY", "").strip()
    if api_key:
        print(f"   ✅ API Key 已配置 ({len(api_key)} 字符)")
        print(f"   密钥前8位: {api_key[:8]}...")
    else:
        print("   ❌ 未配置 API Key")
        print("   💡 请在 .env 文件中设置 DASHSCOPE_API_KEY")

    model_name = os.getenv("QWEN_VL_MODEL", "qwen-vl-max")
    print(f"   🤖 视觉模型: {model_name}")

    print("\n✅ API Key 检查完成！")


if __name__ == "__main__":
    print("=" * 60)
    print("🎯 AI 面试复盘助手 - 图片解析功能测试")
    print("=" * 60)

    test_image_validation()
    test_mime_types()
    test_api_key_check()
    test_real_image_parsing()

    print("\n" + "=" * 60)
    print("🎉 所有测试完成！")
    print("=" * 60)