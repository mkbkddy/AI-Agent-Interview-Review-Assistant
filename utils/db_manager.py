import os
import json
from datetime import datetime
from dotenv import load_dotenv
from langchain_community.query_constructors import supabase
from supabase import create_client, Client
load_dotenv()

LOCAL_DATA_DIR = "./data"
LOCAL_DATA_FILE = os.path.join(LOCAL_DATA_DIR, "interviews.json")

# 尝试连接云端数据库
def get_use_cloud() -> bool:
    try:
        url: str = os.getenv("SUPABASE_URL", "")
        key: str = os.getenv("SUPABASE_KEY", "")
        
        if url and key:
            try:
                supabase: Client = create_client(url, key)
                # 测试连接
                supabase.table("interviews").select("*").limit(1).execute()
                USE_CLOUD = True
                print("☁️ 使用 Supabase 云端数据库模式")
            except Exception as e:
                print(f"⚠️ 无法连接到 Supabase，切换到本地存储模式: {e}")
                USE_CLOUD = False
        else:
            print("⚠️ 未配置 Supabase，使用本地存储模式")
            USE_CLOUD = False
    except Exception as e:
        print(f"⚠️ Supabase 库未安装或初始化失败，使用本地存储模式: {e}")
        USE_CLOUD = False
    return USE_CLOUD

USE_CLOUD = get_use_cloud()

def _ensure_local_dir():
    """确保本地数据目录存在"""
    os.makedirs(LOCAL_DATA_DIR, exist_ok=True)


def _load_local_data():
    """加载本地 JSON 数据"""
    _ensure_local_dir()
    if os.path.exists(LOCAL_DATA_FILE):
        try:
            with open(LOCAL_DATA_FILE, "r", encoding="utf-8") as f:
                return json.load(f)
        except Exception:
            return {}
    return {}


def _save_local_data(data):
    """保存数据到本地 JSON 文件"""
    _ensure_local_dir()
    with open(LOCAL_DATA_FILE, "w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=2)


def init_db():
    """初始化数据库（本地模式）"""
    if USE_CLOUD:
        print("☁️ 使用 Supabase 云端数据库模式")
    else:
        print("📂 使用本地文件存储模式")
        _ensure_local_dir()


def save_interview_result(avg_wpm, scores, report, transcript, user_id="default_user", USE_CLOUD=None):
    """保存面试数据（云端优先，失败则本地存储）"""
    data = {
        "user_id": user_id,
        "avg_wpm": avg_wpm,
        "scores": scores,
        "report": report,
        "transcript": transcript,
        "created_at": datetime.now().isoformat()
    }

    if USE_CLOUD:
        try:
            response = supabase.table("interviews").insert(data).execute()
            print("✅ 数据已保存到云端")
            return response
        except Exception as e:
            print(f"❌ Supabase 写入失败，切换到本地存储: {e}")
            USE_CLOUD = False

    # 本地存储
    local_data = _load_local_data()
    if user_id not in local_data:
        local_data[user_id] = []
    local_data[user_id].append(data)
    _save_local_data(local_data)
    print("✅ 数据已保存到本地")
    return {"data": [data]}


def get_last_interview(user_id="default_user", USE_CLOUD=None):
    """获取该用户最近一次记录"""
    if USE_CLOUD:
        try:
            response = supabase.table("interviews") \
                .select("avg_wpm, scores, created_at") \
                .eq("user_id", user_id) \
                .order("created_at", desc=True) \
                .limit(1) \
                .execute()
            return response.data[0] if response.data else None
        except Exception:
            USE_CLOUD = False

    # 本地存储
    local_data = _load_local_data()
    if user_id in local_data and local_data[user_id]:
        # 按时间排序取最近的
        records = sorted(local_data[user_id], key=lambda x: x["created_at"], reverse=True)
        return records[0] if records else None
    return None


def get_all_history(user_id="default_user", USE_CLOUD: bool = False):
    """获取所有历史，用于画成长曲线"""
    if USE_CLOUD:
        try:
            response = supabase.table("interviews") \
                .select("created_at, avg_wpm, scores") \
                .eq("user_id", user_id) \
                .order("created_at", desc=False) \
                .execute()
            return [(d["created_at"], d["avg_wpm"], d["scores"]) for d in response.data]
        except Exception:
            USE_CLOUD = False

    # 本地存储
    local_data = _load_local_data()
    if user_id in local_data and local_data[user_id]:
        records = sorted(local_data[user_id], key=lambda x: x["created_at"])
        return [(r["created_at"], r["avg_wpm"], r["scores"]) for r in records]
    return []


def get_user_profile(user_id="default_user"):
    """提取用户的长期画像"""
    history = get_all_history(user_id)

    if not history or len(history) < 2:
        return "该用户为新用户，暂无长期历史数据。"

    recent_records = history[-5:]
    recent_scores = [
        d[2].get("技术深度", 0) if isinstance(d[2], dict) else 0
        for d in recent_records
    ]

    avg_score = sum(recent_scores) / len(recent_scores)

    return f"用户近期 {len(recent_scores)} 次面试技术平均分：{avg_score:.1f}。历史关注点：语速稳定性、技术表达的连贯性。"


def get_history_fragments(user_id="default_user", limit=2, USE_CLOUD=None):
    """获取最近 N 次面试的转录文本片段"""
    if USE_CLOUD:
        try:
            response = supabase.table("interviews") \
                .select("transcript, created_at") \
                .eq("user_id", user_id) \
                .order("created_at", desc=True) \
                .limit(limit) \
                .execute()

            if not response.data:
                return "（暂无历史面试片段）"

            fragments = []
            for item in response.data:
                date_str = item["created_at"][:10]
                text_snippet = item["transcript"][:400]
                fragments.append(f"--- 历史面试记录 ({date_str}) ---\n{text_snippet}...")

            return "\n\n".join(fragments)
        except Exception:
            USE_CLOUD = False

    # 本地存储
    local_data = _load_local_data()
    if user_id in local_data and local_data[user_id]:
        records = sorted(local_data[user_id], key=lambda x: x["created_at"], reverse=True)[:limit]
        fragments = []
        for item in records:
            date_str = item["created_at"][:10]
            text_snippet = item.get("transcript", "")[:400]
            fragments.append(f"--- 历史面试记录 ({date_str}) ---\n{text_snippet}...")
        return "\n\n".join(fragments) if fragments else "（暂无历史面试片段）"
    return "（暂无历史面试片段）"
