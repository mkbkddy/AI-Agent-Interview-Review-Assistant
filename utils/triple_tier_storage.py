"""
三级数据存储与读取系统
==================

本模块实现了三级数据存储与读取机制，包括：
1. 第一优先级：Supabase 云端数据库
2. 第二优先级：Redis 缓存
3. 第三优先级：本地存储

功能特性：
- 自动故障转移和数据回退
- 数据一致性保障与同步机制
- 性能监控与日志记录
- 完整的错误处理
"""

import os
import json
import time
import logging
import traceback
from datetime import datetime
from typing import Optional, Any, Dict, List, Tuple
from dotenv import load_dotenv

load_dotenv()

# ============================================================================
# 配置常量
# ============================================================================
LOCAL_DATA_DIR = "./data"
LOCAL_DATA_FILE = os.path.join(LOCAL_DATA_DIR, "interviews.json")
REDIS_CACHE_FILE = os.path.join(LOCAL_DATA_DIR, "redis_cache.json")

# 日志配置
LOG_FILE = os.path.join(LOCAL_DATA_DIR, "storage.log")

# ============================================================================
# 性能监控与日志系统
# ============================================================================
class PerformanceMonitor:
    """性能监控类，记录各层级存储的访问频率、响应时间及失败次数"""
    
    def __init__(self):
        self.stats = {
            "supabase": {"success": 0, "failure": 0, "total_time": 0.0, "last_error": None},
            "redis": {"success": 0, "failure": 0, "total_time": 0.0, "last_error": None},
            "local": {"success": 0, "failure": 0, "total_time": 0.0, "last_error": None}
        }
        self._setup_logger()
    
    def _setup_logger(self):
        """设置日志记录器"""
        os.makedirs(LOCAL_DATA_DIR, exist_ok=True)
        
        self.logger = logging.getLogger("DataStorage")
        self.logger.setLevel(logging.INFO)
        
        # 文件处理器
        fh = logging.FileHandler(LOG_FILE, encoding="utf-8")
        fh.setLevel(logging.INFO)
        
        # 控制台处理器
        ch = logging.StreamHandler()
        ch.setLevel(logging.INFO)
        
        # 格式化
        formatter = logging.Formatter(
            "%(asctime)s - %(levelname)s - %(message)s",
            datefmt="%Y-%m-%d %H:%M:%S"
        )
        fh.setFormatter(formatter)
        ch.setFormatter(formatter)
        
        self.logger.addHandler(fh)
        self.logger.addHandler(ch)
    
    def record_access(self, tier: str, success: bool, duration: float, error: str = None):
        """
        记录访问结果
        
        Args:
            tier: 存储层级 (supabase/redis/local)
            success: 是否成功
            duration: 响应时间（秒）
            error: 错误信息（如果有）
        """
        if tier not in self.stats:
            return
        
        if success:
            self.stats[tier]["success"] += 1
        else:
            self.stats[tier]["failure"] += 1
            self.stats[tier]["last_error"] = error
        
        self.stats[tier]["total_time"] += duration
        
        # 记录日志
        status = "SUCCESS" if success else "FAILURE"
        msg = f"[{tier.upper()}] {status} - Duration: {duration:.4f}s"
        if error:
            msg += f" - Error: {error}"
        
        if success:
            self.logger.info(msg)
        else:
            self.logger.error(msg)
    
    def get_stats(self) -> Dict:
        """获取性能统计信息"""
        return self.stats.copy()
    
    def print_stats(self):
        """打印性能统计信息"""
        print("\n" + "="*60)
        print("📊 数据存储性能统计")
        print("="*60)
        
        for tier, data in self.stats.items():
            total = data["success"] + data["failure"]
            if total > 0:
                avg_time = data["total_time"] / total
                success_rate = data["success"] / total * 100
                print(f"\n【{tier.upper()}】")
                print(f"  总访问次数: {total}")
                print(f"  成功次数: {data['success']} ({success_rate:.1f}%)")
                print(f"  失败次数: {data['failure']}")
                print(f"  平均响应时间: {avg_time:.4f}秒")
                if data["last_error"]:
                    print(f"  最近错误: {data['last_error'][:50]}...")
        
        print("\n" + "="*60)


# 全局性能监控器
monitor = PerformanceMonitor()


# ============================================================================
# 第一层级：Supabase 云端数据库
# ============================================================================
class SupabaseStorage:
    """
    Supabase 云端数据库存储类
    
    功能：
    - 连接管理
    - 数据CRUD操作
    - 健康检查
    """
    
    def __init__(self):
        self.client: Optional[Any] = None
        self.connected = False
        self._connect()
    
    def _connect(self):
        """建立 Supabase 连接"""
        start_time = time.time()
        try:
            from supabase import create_client, Client
            
            url = os.getenv("SUPABASE_URL", "")
            key = os.getenv("SUPABASE_KEY", "")
            
            if not url or not key:
                raise ValueError("Supabase 配置未填写")
            
            self.client = create_client(url, key)
            
            # 测试连接
            self.client.table("interviews").select("*").limit(1).execute()
            
            self.connected = True
            duration = time.time() - start_time
            monitor.record_access("supabase", True, duration)
            print("☁️ Supabase 连接成功")
            
        except Exception as e:
            self.connected = False
            duration = time.time() - start_time
            monitor.record_access("supabase", False, duration, str(e))
            print(f"⚠️ Supabase 连接失败: {e}")
    
    def is_connected(self) -> bool:
        """检查连接状态"""
        if not self.connected or not self.client:
            return False
        
        try:
            self.client.table("interviews").select("id").limit(1).execute()
            return True
        except:
            self.connected = False
            return False
    
    def save(self, data: dict) -> bool:
        """
        保存数据到 Supabase
        
        Args:
            data: 要保存的数据字典
        
        Returns:
            bool: 是否保存成功
        """
        if not self.is_connected():
            return False
        
        start_time = time.time()
        try:
            self.client.table("interviews").insert(data).execute()
            duration = time.time() - start_time
            monitor.record_access("supabase", True, duration)
            return True
        except Exception as e:
            duration = time.time() - start_time
            monitor.record_access("supabase", False, duration, str(e))
            return False
    
    def get_last(self, user_id: str) -> Optional[dict]:
        """获取用户最近一次记录"""
        if not self.is_connected():
            return None
        
        start_time = time.time()
        try:
            response = self.client.table("interviews") \
                .select("avg_wpm, scores, created_at") \
                .eq("user_id", user_id) \
                .order("created_at", desc=True) \
                .limit(1) \
                .execute()
            
            duration = time.time() - start_time
            monitor.record_access("supabase", True, duration)
            
            return response.data[0] if response.data else None
        except Exception as e:
            duration = time.time() - start_time
            monitor.record_access("supabase", False, duration, str(e))
            return None
    
    def get_all(self, user_id: str) -> List:
        """获取用户所有历史记录"""
        if not self.is_connected():
            return []
        
        start_time = time.time()
        try:
            response = self.client.table("interviews") \
                .select("created_at, avg_wpm, scores") \
                .eq("user_id", user_id) \
                .order("created_at", desc=False) \
                .execute()
            
            duration = time.time() - start_time
            monitor.record_access("supabase", True, duration)
            
            return [(d["created_at"], d["avg_wpm"], d["scores"]) for d in response.data]
        except Exception as e:
            duration = time.time() - start_time
            monitor.record_access("supabase", False, duration, str(e))
            return []
    
    def get_fragments(self, user_id: str, limit: int = 2) -> str:
        """获取历史转录片段"""
        if not self.is_connected():
            return ""
        
        start_time = time.time()
        try:
            response = self.client.table("interviews") \
                .select("transcript, created_at") \
                .eq("user_id", user_id) \
                .order("created_at", desc=True) \
                .limit(limit) \
                .execute()
            
            duration = time.time() - start_time
            monitor.record_access("supabase", True, duration)
            
            if not response.data:
                return ""
            
            fragments = []
            for item in response.data:
                date_str = item["created_at"][:10]
                text_snippet = item["transcript"][:400]
                fragments.append(f"--- 历史面试 ({date_str}) ---\n{text_snippet}...")
            
            return "\n\n".join(fragments)
        except Exception as e:
            duration = time.time() - start_time
            monitor.record_access("supabase", False, duration, str(e))
            return ""


# ============================================================================
# 第二层级：Redis 缓存
# ============================================================================
class RedisStorage:
    """
    Redis 缓存存储类
    
    注意：这是一个模拟实现，使用本地 JSON 文件模拟 Redis 行为。
    如需使用真实 Redis，请安装 redis-py 并修改本类实现。
    
    功能：
    - 缓存读写
    - TTL 管理（模拟）
    - 故障检测
    """
    
    def __init__(self):
        self.connected = False
        self.cache: Dict = {}
        self._connect()
    
    def _connect(self):
        """建立 Redis 连接（模拟）"""
        start_time = time.time()
        try:
            # 尝试导入真实 Redis（如果已安装）
            import redis
            redis_host = os.getenv("REDIS_HOST", "localhost")
            redis_port = int(os.getenv("REDIS_PORT", 6379))
            redis_password = os.getenv("REDIS_PASSWORD", None)
            
            self.real_redis = redis.Redis(
                host=redis_host,
                port=redis_port,
                password=redis_password,
                decode_responses=True,
                socket_connect_timeout=5
            )
            # 测试连接
            self.real_redis.ping()
            self.connected = True
            self.use_real_redis = True
            print("🔴 Redis 连接成功（真实Redis）")
        except ImportError:
            # 未安装 redis，使用模拟实现
            self.use_real_redis = False
            self._load_cache_from_file()
            self.connected = True
            print("🔴 Redis 使用模拟实现（本地缓存）")
        except Exception as e:
            # 连接失败，使用模拟实现
            self.use_real_redis = False
            self._load_cache_from_file()
            self.connected = True
            print(f"🔴 Redis 连接失败，使用本地缓存模拟: {e}")
        
        duration = time.time() - start_time
        monitor.record_access("redis", True, duration)
    
    def _load_cache_from_file(self):
        """从文件加载缓存"""
        try:
            if os.path.exists(REDIS_CACHE_FILE):
                with open(REDIS_CACHE_FILE, "r", encoding="utf-8") as f:
                    self.cache = json.load(f)
            else:
                self.cache = {}
        except:
            self.cache = {}
    
    def _save_cache_to_file(self):
        """保存缓存到文件"""
        try:
            os.makedirs(LOCAL_DATA_DIR, exist_ok=True)
            with open(REDIS_CACHE_FILE, "w", encoding="utf-8") as f:
                json.dump(self.cache, f, ensure_ascii=False, indent=2)
        except:
            pass
    
    def is_connected(self) -> bool:
        """检查连接状态"""
        if self.use_real_redis:
            try:
                return self.real_redis.ping()
            except:
                return False
        return True
    
    def set(self, key: str, value: Any, ttl: int = 3600) -> bool:
        """
        设置缓存
        
        Args:
            key: 缓存键
            value: 缓存值
            ttl: 过期时间（秒）
        
        Returns:
            bool: 是否设置成功
        """
        start_time = time.time()
        try:
            if self.use_real_redis:
                self.real_redis.setex(key, ttl, json.dumps(value, ensure_ascii=False))
            else:
                # 模拟实现
                self.cache[key] = {
                    "value": value,
                    "expires_at": time.time() + ttl if ttl > 0 else None
                }
                self._save_cache_to_file()
            
            duration = time.time() - start_time
            monitor.record_access("redis", True, duration)
            return True
        except Exception as e:
            duration = time.time() - start_time
            monitor.record_access("redis", False, duration, str(e))
            return False
    
    def get(self, key: str) -> Optional[Any]:
        """
        获取缓存
        
        Args:
            key: 缓存键
        
        Returns:
            缓存值，如果不存在或过期返回 None
        """
        start_time = time.time()
        try:
            if self.use_real_redis:
                data = self.real_redis.get(key)
                if data:
                    return json.loads(data)
                return None
            else:
                # 模拟实现
                if key in self.cache:
                    entry = self.cache[key]
                    expires_at = entry.get("expires_at")
                    
                    # 检查是否过期
                    if expires_at is None or time.time() < expires_at:
                        return entry["value"]
                    else:
                        # 已过期，删除
                        del self.cache[key]
                        self._save_cache_to_file()
                
                return None
        except Exception as e:
            duration = time.time() - start_time
            monitor.record_access("redis", False, duration, str(e))
            return None
    
    def delete(self, key: str) -> bool:
        """删除缓存"""
        start_time = time.time()
        try:
            if self.use_real_redis:
                self.real_redis.delete(key)
            else:
                if key in self.cache:
                    del self.cache[key]
                    self._save_cache_to_file()
            
            duration = time.time() - start_time
            monitor.record_access("redis", True, duration)
            return True
        except Exception as e:
            duration = time.time() - start_time
            monitor.record_access("redis", False, duration, str(e))
            return False
    
    def save_interview(self, user_id: str, data: dict) -> bool:
        """保存面试数据到缓存"""
        cache_key = f"interview:{user_id}"
        return self.set(cache_key, data, ttl=86400)  # 24小时
    
    def get_interviews(self, user_id: str) -> Optional[List]:
        """获取缓存的面试数据列表"""
        cache_key = f"interviews_list:{user_id}"
        return self.get(cache_key)
    
    def append_interview(self, user_id: str, data: dict) -> bool:
        """追加面试数据到缓存列表"""
        cache_key = f"interviews_list:{user_id}"
        interviews = self.get(cache_key) or []
        interviews.append(data)
        return self.set(cache_key, interviews, ttl=86400)


# ============================================================================
# 第三层级：本地存储
# ============================================================================
class LocalStorage:
    """
    本地文件存储类
    
    功能：
    - JSON 文件读写
    - 目录管理
    """
    
    def __init__(self):
        self._ensure_local_dir()
    
    def _ensure_local_dir(self):
        """确保本地数据目录存在"""
        os.makedirs(LOCAL_DATA_DIR, exist_ok=True)
    
    def _load_data(self) -> dict:
        """加载本地 JSON 数据"""
        self._ensure_local_dir()
        if os.path.exists(LOCAL_DATA_FILE):
            try:
                with open(LOCAL_DATA_FILE, "r", encoding="utf-8") as f:
                    return json.load(f)
            except Exception as e:
                monitor.logger.error(f"加载本地数据失败: {e}")
                return {}
        return {}
    
    def _save_data(self, data: dict) -> bool:
        """保存数据到本地 JSON 文件"""
        self._ensure_local_dir()
        try:
            with open(LOCAL_DATA_FILE, "w", encoding="utf-8") as f:
                json.dump(data, f, ensure_ascii=False, indent=2)
            return True
        except Exception as e:
            monitor.logger.error(f"保存本地数据失败: {e}")
            return False
    
    def save(self, user_id: str, data: dict) -> bool:
        """保存面试数据"""
        start_time = time.time()
        try:
            local_data = self._load_data()
            if user_id not in local_data:
                local_data[user_id] = []
            local_data[user_id].append(data)
            self._save_data(local_data)
            
            duration = time.time() - start_time
            monitor.record_access("local", True, duration)
            return True
        except Exception as e:
            duration = time.time() - start_time
            monitor.record_access("local", False, duration, str(e))
            return False
    
    def get_last(self, user_id: str) -> Optional[dict]:
        """获取用户最近一次记录"""
        start_time = time.time()
        try:
            local_data = self._load_data()
            if user_id in local_data and local_data[user_id]:
                records = sorted(
                    local_data[user_id],
                    key=lambda x: x.get("created_at", ""),
                    reverse=True
                )
                duration = time.time() - start_time
                monitor.record_access("local", True, duration)
                return records[0] if records else None
            
            duration = time.time() - start_time
            monitor.record_access("local", True, duration)
            return None
        except Exception as e:
            duration = time.time() - start_time
            monitor.record_access("local", False, duration, str(e))
            return None
    
    def get_all(self, user_id: str) -> List:
        """获取用户所有历史记录"""
        start_time = time.time()
        try:
            local_data = self._load_data()
            if user_id in local_data and local_data[user_id]:
                records = sorted(
                    local_data[user_id],
                    key=lambda x: x.get("created_at", "")
                )
                duration = time.time() - start_time
                monitor.record_access("local", True, duration)
                return [(r.get("created_at", ""), r.get("avg_wpm", 0), r.get("scores", {})) for r in records]
            
            duration = time.time() - start_time
            monitor.record_access("local", True, duration)
            return []
        except Exception as e:
            duration = time.time() - start_time
            monitor.record_access("local", False, duration, str(e))
            return []
    
    def get_fragments(self, user_id: str, limit: int = 2) -> str:
        """获取历史转录片段"""
        start_time = time.time()
        try:
            local_data = self._load_data()
            if user_id in local_data and local_data[user_id]:
                records = sorted(
                    local_data[user_id],
                    key=lambda x: x.get("created_at", ""),
                    reverse=True
                )[:limit]
                
                fragments = []
                for item in records:
                    date_str = item.get("created_at", "")[:10]
                    text_snippet = item.get("transcript", "")[:400]
                    fragments.append(f"--- 历史面试 ({date_str}) ---\n{text_snippet}...")
                
                duration = time.time() - start_time
                monitor.record_access("local", True, duration)
                return "\n\n".join(fragments) if fragments else ""
            
            duration = time.time() - start_time
            monitor.record_access("local", True, duration)
            return ""
        except Exception as e:
            duration = time.time() - start_time
            monitor.record_access("local", False, duration, str(e))
            return ""


# ============================================================================
# 三级存储管理器
# ============================================================================
class TripleTierStorage:
    """
    三级数据存储管理器
    
    数据访问优先级：
    1. Supabase 云端数据库（第一优先级）
    2. Redis 缓存（第二优先级）
    3. 本地存储（第三优先级）
    
    功能：
    - 自动故障转移
    - 数据同步
    - 性能监控
    """
    
    def __init__(self):
        print("\n" + "="*60)
        print("🔄 初始化三级数据存储系统...")
        print("="*60)
        
        # 初始化各层级存储
        self.supabase = SupabaseStorage()
        self.redis = RedisStorage()
        self.local = LocalStorage()
        
        # 数据同步状态
        self.pending_sync = []  # 待同步到云端的数据
        self.last_sync_time = None
        
        print("\n📊 存储层级状态：")
        print(f"  ☁️ Supabase: {'已连接' if self.supabase.is_connected() else '未连接'}")
        print(f"  🔴 Redis: {'已连接' if self.redis.is_connected() else '未连接'}")
        print(f"  📁 Local: 已就绪")
        print("="*60 + "\n")
    
    def _load_pending_sync(self):
        """加载待同步数据列表"""
        sync_file = os.path.join(LOCAL_DATA_DIR, "pending_sync.json")
        try:
            if os.path.exists(sync_file):
                with open(sync_file, "r", encoding="utf-8") as f:
                    self.pending_sync = json.load(f)
        except:
            self.pending_sync = []
    
    def _save_pending_sync(self):
        """保存待同步数据列表"""
        sync_file = os.path.join(LOCAL_DATA_DIR, "pending_sync.json")
        try:
            with open(sync_file, "w", encoding="utf-8") as f:
                json.dump(self.pending_sync, f, ensure_ascii=False, indent=2)
        except:
            pass
    
    def sync_to_cloud(self) -> Tuple[int, int]:
        """
        同步本地和缓存数据到云端
        
        Returns:
            Tuple[成功数, 失败数]
        """
        if not self.supabase.is_connected():
            print("⚠️ Supabase 未连接，跳过同步")
            return (0, 0)
        
        self._load_pending_sync()
        
        success_count = 0
        fail_count = 0
        
        for item in self.pending_sync[:]:  # 使用切片避免迭代中修改列表
            try:
                if self.supabase.save(item["data"]):
                    self.pending_sync.remove(item)
                    success_count += 1
                    print(f"✅ 同步成功: {item['user_id']} - {item['data'].get('created_at', '')[:10]}")
            except Exception as e:
                fail_count += 1
                print(f"❌ 同步失败: {e}")
        
        self._save_pending_sync()
        self.last_sync_time = datetime.now()
        
        return (success_count, fail_count)
    
    def check_and_sync(self) -> bool:
        """
        检查并同步数据
        
        当 Supabase 恢复连接时，自动同步待同步数据
        
        Returns:
            bool: 是否执行了同步
        """
        if self.supabase.is_connected() and self.pending_sync:
            print("\n🔄 检测到 Supabase 已恢复连接，开始同步数据...")
            success, fail = self.sync_to_cloud()
            if success > 0:
                print(f"📤 同步完成: 成功 {success} 条, 失败 {fail} 条")
                return True
        return False
    
    def save_interview(
        self,
        user_id: str,
        avg_wpm: float,
        scores: dict,
        report: str,
        transcript: str
    ) -> dict:
        """
        保存面试数据
        
        数据流动：
        1. 优先尝试保存到 Supabase
        2. 如果失败，保存到 Redis 缓存
        3. 如果 Redis 也失败，保存到本地
        
        如果保存到次级存储，会将数据加入待同步队列
        
        Args:
            user_id: 用户ID
            avg_wpm: 平均语速
            scores: 评分
            report: 报告
            transcript: 转录文本
        
        Returns:
            dict: 保存结果信息
        """
        data = {
            "user_id": user_id,
            "avg_wpm": avg_wpm,
            "scores": scores,
            "report": report,
            "transcript": transcript,
            "created_at": datetime.now().isoformat()
        }
        
        result = {
            "success": False,
            "tier": None,
            "message": "",
            "data": data
        }
        
        # 第一优先级：Supabase
        if self.supabase.is_connected():
            if self.supabase.save(data):
                result["success"] = True
                result["tier"] = "supabase"
                result["message"] = "数据已保存到云端"
                print(f"✅ 保存成功 [Supabase]: {user_id}")
                
                # 检查并同步待同步数据
                self.check_and_sync()
                return result
            else:
                print("⚠️ Supabase 保存失败，尝试降级...")
        
        # 第二优先级：Redis
        if self.redis.is_connected():
            if self.redis.append_interview(user_id, data):
                self.pending_sync.append({
                    "user_id": user_id,
                    "data": data,
                    "timestamp": time.time()
                })
                self._save_pending_sync()
                
                result["success"] = True
                result["tier"] = "redis"
                result["message"] = "数据已保存到缓存，待网络恢复后同步到云端"
                print(f"🔴 保存成功 [Redis]: {user_id}")
                return result
        
        # 第三优先级：本地
        if self.local.save(user_id, data):
            self.pending_sync.append({
                "user_id": user_id,
                "data": data,
                "timestamp": time.time()
            })
            self._save_pending_sync()
            
            result["success"] = True
            result["tier"] = "local"
            result["message"] = "数据已保存到本地，待网络恢复后同步到云端"
            print(f"📁 保存成功 [Local]: {user_id}")
            return result
        
        # 所有层级都失败
        result["message"] = "所有存储层级都失败"
        print("❌ 保存失败 [所有层级]")
        return result
    
    def get_last_interview(self, user_id: str) -> Optional[dict]:
        """
        获取用户最近一次面试记录
        
        优先级：Supabase > Redis > Local
        """
        # 第一优先级：Supabase
        if self.supabase.is_connected():
            result = self.supabase.get_last(user_id)
            if result:
                print(f"📤 获取成功 [Supabase]: {user_id}")
                return result
        
        # 第二优先级：Redis
        if self.redis.is_connected():
            result = self.redis.get_interviews(user_id)
            if result:
                print(f"🔴 获取成功 [Redis]: {user_id}")
                # 返回最新的一条
                sorted_result = sorted(result, key=lambda x: x.get("created_at", ""), reverse=True)
                return sorted_result[0] if sorted_result else None
        
        # 第三优先级：本地
        result = self.local.get_last(user_id)
        if result:
            print(f"📁 获取成功 [Local]: {user_id}")
        return result
    
    def get_all_history(self, user_id: str) -> List:
        """
        获取用户所有历史记录
        
        优先级：Supabase > Redis > Local
        """
        # 第一优先级：Supabase
        if self.supabase.is_connected():
            result = self.supabase.get_all(user_id)
            if result:
                print(f"📤 获取历史成功 [Supabase]: {user_id} - {len(result)} 条记录")
                return result
        
        # 第二优先级：Redis
        if self.redis.is_connected():
            result = self.redis.get_interviews(user_id)
            if result:
                print(f"🔴 获取历史成功 [Redis]: {user_id} - {len(result)} 条记录")
                return [(r.get("created_at", ""), r.get("avg_wpm", 0), r.get("scores", {})) for r in result]
        
        # 第三优先级：本地
        result = self.local.get_all(user_id)
        if result:
            print(f"📁 获取历史成功 [Local]: {user_id} - {len(result)} 条记录")
        return result
    
    def get_history_fragments(self, user_id: str, limit: int = 2) -> str:
        """获取历史转录片段"""
        # 第一优先级：Supabase
        if self.supabase.is_connected():
            result = self.supabase.get_fragments(user_id, limit)
            if result:
                return result
        
        # 第二优先级：Redis
        if self.redis.is_connected():
            result = self.redis.get_interviews(user_id)
            if result:
                sorted_result = sorted(result, key=lambda x: x.get("created_at", ""), reverse=True)[:limit]
                fragments = []
                for item in sorted_result:
                    date_str = item.get("created_at", "")[:10]
                    text_snippet = item.get("transcript", "")[:400]
                    fragments.append(f"--- 历史面试 ({date_str}) ---\n{text_snippet}...")
                return "\n\n".join(fragments)
        
        # 第三优先级：本地
        return self.local.get_fragments(user_id, limit)
    
    def get_user_profile(self, user_id: str) -> str:
        """提取用户的长期画像"""
        history = self.get_all_history(user_id)
        
        if not history or len(history) < 2:
            return "该用户为新用户，暂无长期历史数据。"
        
        recent_records = history[-5:]
        recent_scores = [
            d[2].get("技术深度", 0) if isinstance(d[2], dict) else 0
            for d in recent_records
        ]
        
        avg_score = sum(recent_scores) / len(recent_scores)
        
        return f"用户近期 {len(recent_scores)} 次面试技术平均分：{avg_score:.1f}。历史关注点：语速稳定性、技术表达的连贯性。"
    
    def get_stats(self) -> dict:
        """获取性能统计信息"""
        return monitor.get_stats()
    
    def print_stats(self):
        """打印性能统计"""
        monitor.print_stats()
    
    def get_status(self) -> dict:
        """获取存储系统状态"""
        return {
            "supabase": {
                "connected": self.supabase.is_connected(),
                "type": "cloud"
            },
            "redis": {
                "connected": self.redis.is_connected(),
                "type": "cache",
                "use_real": getattr(self.redis, 'use_real_redis', False)
            },
            "local": {
                "connected": True,
                "type": "local"
            },
            "pending_sync": len(self.pending_sync),
            "last_sync": self.last_sync_time.isoformat() if self.last_sync_time else None
        }


# ============================================================================
# 全局存储管理器实例
# ============================================================================
_storage_instance: Optional[TripleTierStorage] = None


def get_storage() -> TripleTierStorage:
    """获取存储管理器单例"""
    global _storage_instance
    if _storage_instance is None:
        _storage_instance = TripleTierStorage()
    return _storage_instance


def init_db():
    """初始化数据库（兼容旧接口）"""
    storage = get_storage()
    print(f"\n数据存储初始化完成")
    print(f"存储层级: Supabase={storage.supabase.is_connected()}, Redis={storage.redis.is_connected()}, Local=True")


def save_interview_result(avg_wpm, scores, report, transcript, user_id="default_user"):
    """保存面试结果（兼容旧接口）"""
    storage = get_storage()
    return storage.save_interview(user_id, avg_wpm, scores, report, transcript)


def get_last_interview(user_id="default_user"):
    """获取最近面试（兼容旧接口）"""
    storage = get_storage()
    return storage.get_last_interview(user_id)


def get_all_history(user_id="default_user"):
    """获取所有历史（兼容旧接口）"""
    storage = get_storage()
    return storage.get_all_history(user_id)


def get_user_profile(user_id="default_user"):
    """获取用户画像（兼容旧接口）"""
    storage = get_storage()
    return storage.get_user_profile(user_id)


def get_history_fragments(user_id="default_user", limit=2):
    """获取历史片段（兼容旧接口）"""
    storage = get_storage()
    return storage.get_history_fragments(user_id, limit)


def get_use_cloud() -> bool:
    """检查是否使用云端（兼容旧接口）"""
    storage = get_storage()
    return storage.supabase.is_connected()


def sync_to_cloud():
    """手动触发数据同步（新增）"""
    storage = get_storage()
    return storage.sync_to_cloud()


def print_storage_stats():
    """打印存储统计（新增）"""
    storage = get_storage()
    storage.print_stats()


def get_storage_status() -> dict:
    """获取存储状态（新增）"""
    storage = get_storage()
    return storage.get_status()
