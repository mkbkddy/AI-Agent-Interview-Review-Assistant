#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
测试上下文压缩功能
"""
from utils.context_compressor import SemanticCompressor, check_compression_needed, compress_context

def test_compression():
    """测试压缩功能"""
    print("=" * 60)
    print("测试上下文压缩功能")
    print("=" * 60)
    
    # 创建压缩器实例
    compressor = SemanticCompressor()
    
    # 测试消息列表
    messages = [
        {"role": "assistant", "content": "我是你的 AI 面试导师，你可以针对本次面试复盘向我提问，或让我帮你练习特定环节。"},
        {"role": "user", "content": "我想了解Java中的Spring Boot框架"},
        {"role": "assistant", "content": "Spring Boot是Spring框架的扩展，它简化了Spring应用的创建和部署过程。主要特点包括：自动配置、内嵌服务器、starter依赖等。"},
        {"role": "user", "content": "那Spring MVC和Spring Boot有什么区别？"},
        {"role": "assistant", "content": "Spring MVC是Spring框架的一个模块，专注于Web层的MVC模式。而Spring Boot是对Spring框架的封装和扩展，提供了自动配置、嵌入式服务器等特性来简化开发。"},
        {"role": "user", "content": "能详细讲讲Spring Boot的启动流程吗？"},
        {"role": "assistant", "content": "Spring Boot的启动流程包括：1. 创建SpringApplication实例，推断应用类型；2. 运行run方法，准备Environment；3. 创建ApplicationContext；4. 刷新上下文（关键步骤）；5. 执行CommandLineRunner等扩展点。"},
        {"role": "user", "content": "什么是Spring Boot的自动配置原理？"},
        {"role": "assistant", "content": "Spring Boot的自动配置原理基于@EnableAutoConfiguration注解，通过@SpringBootApplication组合实现。它会扫描classpath下的META-INF/spring.factories文件，加载AutoConfiguration类，并根据条件注解(@Conditional)来决定是否生效。"},
        {"role": "user", "content": "Spring Boot中的bean作用域有哪些？"},
        {"role": "assistant", "content": "Spring Boot中的bean作用域包括：singleton（单例，默认）、prototype（原型，每次获取创建新实例）、request（请求作用域）、session（会话作用域）、application（应用作用域）和websocket（WebSocket作用域）。"},
        {"role": "user", "content": "能讲讲Spring的事务管理吗？"},
        {"role": "assistant", "content": "Spring事务管理支持编程式和声明式两种方式。声明式事务通过@Transactional注解实现，支持propagation（传播行为）、isolation（隔离级别）、rollbackFor（回滚条件）等属性配置。"},
        {"role": "user", "content": "什么是Spring Boot的拦截器和过滤器？"},
        {"role": "assistant", "content": "Filter是Servlet规范的一部分，在Spring Boot中通过@WebFilter注解或FilterRegistrationBean注册。Interceptor是Spring MVC的一部分，通过实现HandlerInterceptor接口并在WebMvcConfigurer中注册。执行顺序：Filter → Interceptor → Controller。"},
    ]
    
    print(f"\n原始消息数量: {len(messages)}")
    
    # 测试token计数
    total_tokens = compressor.count_messages_tokens(messages)
    print(f"估算总Token数: {total_tokens}")
    
    # 测试压缩检查
    print("\n" + "-" * 40)
    print("测试压缩检查")
    should_compress, reason = compressor.should_compress(messages)
    print(f"是否需要压缩: {should_compress}")
    print(f"原因: {reason}")
    
    # 测试压缩
    print("\n" + "-" * 40)
    print("测试压缩执行")
    compressed = compressor.compress(messages)
    
    print(f"\n压缩结果:")
    print(f"  原始消息数: {compressed.original_count}")
    print(f"  压缩后消息数: {compressed.compressed_count}")
    print(f"  压缩率: {compressed.compression_ratio:.1%}")
    print(f"  核心话题: {', '.join(compressed.core_topics)}")
    print(f"  摘要: {compressed.summary}")
    
    # 显示压缩后的消息
    print("\n压缩后的消息列表:")
    for i, msg in enumerate(compressed.compressed_messages, 1):
        role = msg.get("role", "unknown")
        content = msg.get("content", "")[:60]
        print(f"  {i}. [{role}] {content}...")
    
    # 测试检查函数
    print("\n" + "-" * 40)
    print("测试便捷函数")
    should_compress2, reason2 = check_compression_needed(messages)
    print(f"check_compression_needed: {should_compress2} - {reason2}")
    
    compressed2 = compress_context(messages[:6])
    print(f"compress_context: {compressed2.original_count} -> {compressed2.compressed_count}")


def test_core_info_extraction():
    """测试核心信息提取"""
    print("\n" + "=" * 60)
    print("测试核心信息提取")
    print("=" * 60)
    
    compressor = SemanticCompressor()
    
    messages = [
        {"role": "user", "content": "我是Java后端开发工程师，有3年经验"},
        {"role": "assistant", "content": "很高兴认识你！请问你想练习哪种类型的面试？"},
        {"role": "user", "content": "我想准备阿里的Java高级工程师面试"},
        {"role": "assistant", "content": "好的，阿里Java高级工程师面试会涉及：JVM、并发编程、分布式、微服务、数据库优化等。"},
        {"role": "user", "content": "能详细讲讲JVM垃圾回收吗？"},
        {"role": "assistant", "content": "JVM垃圾回收主要包括：年轻代的Minor GC和老年代的Full GC。常用算法有：标记-清除、标记-整理、复制算法。G1和ZGC是较新的垃圾收集器。"},
    ]
    
    core_info = compressor.extract_core_info(messages)
    
    print("\n提取的核心信息:")
    print(f"  用户画像: {core_info['user_profile']}")
    print(f"  关键话题: {', '.join(core_info['key_topics'])}")
    print(f"  上下文摘要: {core_info['context_summary']}")


if __name__ == "__main__":
    test_compression()
    test_core_info_extraction()
