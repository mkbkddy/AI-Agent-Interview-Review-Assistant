import streamlit as st
from dotenv import load_dotenv
load_dotenv()

import sys
import codecs
import os

# 禁用 TorchCodec（避免 FFmpeg DLL 加载问题）
os.environ["TORCH_CODEC_AVAILABLE"] = "0"

if sys.platform == "win32":
    sys.stdout.reconfigure(encoding="utf-8")
    sys.stderr.reconfigure(encoding="utf-8")

import whisperx
import json
import re
import uuid
import traceback
import numpy as np
import pandas as pd
from opencc import OpenCC
from openai import OpenAI, api_key

from utils.triple_tier_storage import (
    init_db,
    save_interview_result,
    get_last_interview,
    get_all_history,
    get_user_profile,
    get_history_fragments,
    get_use_cloud,
    sync_to_cloud,
    print_storage_stats,
    get_storage_status
)
from utils.metrics import calculate_wpm, generate_radar_chart
from utils.rag_engine import process_jd_to_context
from utils.knowledge_manager import (
    init_knowledge_base,
    upload_knowledge_file,
    get_knowledge_list,
    get_knowledge_content,
    delete_knowledge_file,
    search_knowledge,
    get_knowledge_stats,
    match_knowledge_with_interview,
    generate_prompt_from_knowledge,
    get_relevant_knowledge_summary
)
from utils.agent import agent_chat, get_agent
from utils.context_compressor import check_compression_needed, compress_context, get_compressor

cc = OpenCC("t2s")


# 从环境变量加载配置（请在 .env 文件中设置）
DASHSCOPE_API_KEY = os.getenv("DASHSCOPE_API_KEY", "")
QWEN_MODEL = os.getenv("QWEN_MODEL", "qwen-max")
QWEN_VL_MODEL = os.getenv("QWEN_VL_MODEL", "qwen-vl-max")
DASHSCOPE_BASE_URL = os.getenv("DASHSCOPE_BASE_URL", "https://dashscope.aliyuncs.com/compatible-mode/v1")


def create_openai_client():
    """创建通义千问 OpenAI 兼容客户端，确保所有参数为 ASCII"""
    api_key_raw = DASHSCOPE_API_KEY
    if not api_key_raw:
        return None, "未配置 DASHSCOPE_API_KEY"

    # 确保 API key 为纯 ASCII
    try:
        api_key = api_key_raw.encode("ascii").decode("ascii")
    except UnicodeEncodeError:
        return None, "API Key 包含非 ASCII 字符"

    try:
        client = OpenAI(api_key=api_key, base_url=DASHSCOPE_BASE_URL)
        return client, None
    except Exception as e:
        return None, f"创建客户端失败: {e}"


def safe_model_name(name: str) -> str:
    """确保模型名为纯 ASCII"""
    name = name.strip()
    try:
        return name.encode("ascii").decode("ascii")
    except UnicodeEncodeError:
        return QWEN_MODEL


def load_audio_file(file_path: str):
    """加载音频文件为 numpy 数组"""
    try:
        with open(file_path, "rb") as f:
            header = f.read(12)

        if header.startswith(b"RIFF"):
            try:
                from scipy.io import wavfile
                sample_rate, data = wavfile.read(file_path)
                if len(data.shape) > 1:
                    data = data.mean(axis=1)
                data = data.astype(np.float32)
                if data.max() > 1.0:
                    data = data / 32768.0
                if sample_rate != 16000:
                    import scipy.signal
                    num_samples = int(len(data) * 16000 / sample_rate)
                    data = scipy.signal.resample(data, num_samples)
                return data
            except Exception:
                pass

        # 通用：用 librosa（如果安装了）或者直接返回错误
        try:
            import librosa
            audio, _ = librosa.load(file_path, sr=16000, mono=True)
            return audio.astype(np.float32)
        except Exception:
            raise RuntimeError(
                "无法加载音频。请尝试将文件转换为 WAV 格式（16kHz, 单声道）"
            )
    except Exception as e:
        raise RuntimeError(f"加载音频失败: {e}")


st.set_page_config(page_title="AI 面试复盘助手", layout="wide", page_icon="🎤")
init_db()

current_user_id = "user_01"

if "report" not in st.session_state:
    st.session_state.report = None
if "transcript" not in st.session_state:
    st.session_state.transcript = None
if "df_wpm" not in st.session_state:
    st.session_state.df_wpm = None
if "scores" not in st.session_state:
    st.session_state.scores = None
if "jd_context" not in st.session_state:
    st.session_state.jd_context = "通用面试评价标准"
if "cv_context" not in st.session_state:
    st.session_state.cv_context = "暂无简历信息"
if "messages" not in st.session_state:
    st.session_state.messages = [
        {"role": "assistant", "content": "我是你的 AI 面试导师，你可以针对本次面试复盘向我提问，或让我帮你练习特定环节。"}
    ]
if "cv_score_result" not in st.session_state:
    st.session_state.cv_score_result = None

# 上下文压缩相关状态
if "compression_enabled" not in st.session_state:
    st.session_state.compression_enabled = True  # 默认启用自动压缩
if "last_compression_info" not in st.session_state:
    st.session_state.last_compression_info = None  # 上次压缩信息
if "auto_compression_warning_shown" not in st.session_state:
    st.session_state.auto_compression_warning_shown = False  # 是否已显示自动压缩提示



def evaluate_cv(jd_context: str, cv_context: str) -> str:
    """基于 JD 对简历进行多维度评分"""
    client, client_error = create_openai_client()
    if client_error:
        return f"❌ {client_error}"

    prompt = f"""
    请你作为一位资深的技术招聘专家，根据以下岗位描述（JD）对简历进行全面评估。

    【岗位描述（JD）】：
    {jd_context}

    【个人简历（CV）】：
    {cv_context}

    请从以下维度进行评估：
    1. **岗位匹配度**：整体匹配程度
    2. **技能相关性**：技能与岗位要求的吻合度
    3. **经验符合度**：工作经验与岗位要求的匹配度
    4. **学历背景**：教育背景与岗位要求的匹配度
    5. **项目经历**：项目经验与岗位的相关性

    评分标准（每项 1-10 分）：
    - 1-3分：基本不符合要求
    - 4-6分：部分符合要求
    - 7-8分：比较符合要求
    - 9-10分：非常符合要求

    请输出：
    1. 各项评分（JSON格式）：{{"岗位匹配度": X, "技能相关性": X, "经验符合度": X, "学历背景": X, "项目经历": X}}
    2. 综合评价：对简历的总体评价
    3. 优势分析：简历中的亮点
    4. 改进建议：针对岗位要求，简历需要改进的地方
    """

    try:
        response = client.chat.completions.create(
            model=model_name,
            messages=[{"role": "user", "content": prompt}],
            max_tokens=2000,
        )
        return response.choices[0].message.content
    except Exception as e:
        return f"❌ 简历评分失败: {type(e).__name__}: {e}"


def extract_scores(text: str):
    try:
        match = re.search(r"Scores:\s*(\{.*?\})", text, re.DOTALL)
        if match:
            return json.loads(match.group(1))
    except Exception:
        pass
    return None

# 连接云端数据库
USE_CLOUD = get_use_cloud()

api_key = DASHSCOPE_API_KEY
model_name = QWEN_MODEL

with st.sidebar:
    st.header("⚙️ 设置")
    if api_key:
        st.success("✅ 通义千问 API Key 已配置")
    else:
        st.error("❌ 请配置通义千问 API Key")
        st.info("请在 .env 文件中设置 DASHSCOPE_API_KEY")

    st.write(f"**当前模型**: {model_name}")
    st.write(f"**Base URL**: {DASHSCOPE_BASE_URL}")

    model_size = st.selectbox("Whisper 模型", ["tiny", "base", "small"], index=0)
    custom_model_path = st.text_input("自定义模型路径（可选）", value="")

    st.divider()
    st.header("🎯 岗位针对性增强")
    jd_file = st.file_uploader(
        "上传目标岗位 JD (图片/PDF/TXT)",
        type=["pdf", "txt", "png", "jpg", "jpeg"],
    )
    st.header("📄 个人背景增强")
    cv_file = st.file_uploader(
        "上传个人简历 (图片/PDF/TXT)",
        type=["pdf", "txt", "png", "jpg", "jpeg"],
    )

    st.divider()
    st.header("📚 专业知识库")
    st.info("上传专业知识文档，用于面试复盘时参考\n\n💡 系统会自动检测重复文件，避免重复上传")
    
    knowledge_file = st.file_uploader(
        "上传知识库文件 (PDF/TXT/MD/Word)",
        type=["pdf", "txt", "md", "doc", "docx"],
        key="knowledge_uploader"
    )
    
    if knowledge_file:
        with st.spinner("正在上传知识库文件..."):
            result = upload_knowledge_file(knowledge_file.name, knowledge_file.getvalue())
            if result["success"]:
                if result.get("is_duplicate", False):
                    st.info(f"ℹ️ {result['message']}")
                    st.info("💾 使用已缓存的文件内容，无需重复上传")
                else:
                    st.success(f"✅ {result['message']}")
                    st.write(f"文件名: {result['file_info']['original_name']}")
                    st.write(f"大小: {result['file_info']['size'] / 1024:.1f} KB")
            else:
                st.error(f"❌ {result['message']}")
    
    # 知识库文件列表
    knowledge_list = get_knowledge_list()
    if knowledge_list:
        st.write(f"已上传 {len(knowledge_list)} 个知识库文件:")
        for idx, file_info in enumerate(knowledge_list):
            col1, col2 = st.columns([3, 1])
            with col1:
                st.write(f"{idx+1}. {file_info['original_name']}")
            with col2:
                if st.button("删除", key=f"del_knowledge_{file_info['file_id']}"):
                    delete_knowledge_file(file_info["file_id"])
                    st.rerun()


st.title("🎤 AI 面试复盘助手")

uploaded_file = st.file_uploader(
    "选择面试音频文件", type=["wav", "mp3", "m4a", "flac"]
)

# 简历评分功能区域
st.divider()
st.subheader("📝 简历评分（可选）")
st.info("💡 上传 JD 和简历后，可先进行简历评分，评估匹配度后再进行面试复盘")

jd_uploaded = jd_file is not None
cv_uploaded = cv_file is not None

col1, col2 = st.columns(2)
with col1:
    if jd_uploaded:
        st.success(f"✅ JD 文件已上传: {jd_file.name}")
    else:
        st.warning("⚠️ 请上传岗位 JD")

with col2:
    if cv_uploaded:
        st.success(f"✅ 简历文件已上传: {cv_file.name}")
    else:
        st.warning("⚠️ 请上传个人简历")

if jd_uploaded and cv_uploaded:
    if st.button("🎯 开始简历评分", type="primary", key="cv_score_btn"):
        with st.spinner("正在分析简历与岗位匹配度..."):
            # 解析 JD
            ext = jd_file.name.split(".")[-1]
            temp_jd = f"temp_jd_{uuid.uuid4().hex}.{ext}"
            with open(temp_jd, "wb") as f:
                f.write(jd_file.getvalue())
            jd_context = process_jd_to_context(temp_jd)
            os.remove(temp_jd)

            # 解析 CV
            ext = cv_file.name.split(".")[-1]
            temp_cv = f"temp_cv_{uuid.uuid4().hex}.{ext}"
            with open(temp_cv, "wb") as f:
                f.write(cv_file.getvalue())
            cv_context = process_jd_to_context(temp_cv)
            os.remove(temp_cv)

            print(jd_context)
            print("="*30)
            print(cv_context)

            # 保存到 session state
            st.session_state.jd_context = jd_context
            st.session_state.cv_context = cv_context

            # 评估简历
            result = evaluate_cv(jd_context, cv_context)
            st.session_state.cv_score_result = result

            # 显示结果
            st.success("✅ 简历评分完成！")

# 显示简历评分结果
if st.session_state.cv_score_result:
    st.divider()
    st.subheader("📊 简历评分结果")
    result = st.session_state.cv_score_result
    
    if result.startswith("❌"):
        st.error(result)
    else:
        st.markdown(result)

        # 尝试提取评分并展示为图表
        try:
            import re
            json_match = re.search(r"\{.*?\}", result)
            if json_match:
                scores = json.loads(json_match.group())
                if isinstance(scores, dict):
                    st.subheader("📈 评分可视化")
                    df = pd.DataFrame.from_dict(scores, orient="index", columns=["评分"])
                    df["评分"] = df["评分"].astype(float)
                    st.bar_chart(df)
        except Exception as e:
            pass

tab_audio, tab_metrics, tab_report, tab_history, tab_knowledge, tab_assistant = st.tabs(
    ["🎵 音频处理", "📊 量化分析看板", "🧠 AI 深度诊断", "📈 成长轨迹", "📚 知识库", "💬 AI 助手"]
)

audio_path = "temp_audio.wav"

with tab_history:
    st.subheader("🚀 你的面试进化史")
    try:
        history_data = get_all_history(current_user_id)
        if not history_data:
            st.info("👋 欢迎新同学！你还没有历史面试数据，完成复盘后这里将生成成长曲线。")
        else:
            st.success(f"📊 已加载共计 {len(history_data)} 次面试记录。")
    except Exception as e:
        st.error(f"历史档案加载失败: {e}")

with tab_knowledge:
    st.subheader("📚 专业知识库管理")
    
    # 向量索引管理区域
    with st.expander("🧠 向量索引管理", expanded=False):
        st.write("构建向量索引可大幅提升语义搜索精度，支持更智能的内容检索")
        
        # 显示当前索引状态
        try:
            from utils.knowledge_manager import get_vector_index_stats
            stats = get_vector_index_stats()
            col1, col2, col3 = st.columns(3)
            with col1:
                st.metric("文档分块数", stats.get("total_chunks", 0))
            with col2:
                st.metric("索引文档数", stats.get("total_documents", 0))
            with col3:
                st.metric("向量维度", stats.get("vector_dim", 0))
            
            if stats.get("has_vector_model"):
                st.success("✅ 向量模型已加载，可执行精确向量搜索")
            else:
                st.warning("⚠️ 未检测到 sentence-transformers 模型，将使用基础匹配")
            
            # 构建/重建索引按钮
            col1, col2 = st.columns([1, 1])
            with col1:
                if st.button("🔄 构建向量索引", key="build_vector_index"):
                    with st.spinner("正在分析文档并构建向量索引，请稍候..."):
                        from utils.knowledge_manager import build_vector_index_for_knowledge
                        result = build_vector_index_for_knowledge()
                        if result["success"]:
                            st.success(f"✅ {result['message']}")
                            st.rerun()
                        else:
                            st.error(f"❌ {result['message']}")
            with col2:
                if st.button("📊 查看索引统计", key="view_index_stats"):
                    st.json(stats)
        except Exception as e:
            st.warning(f"向量索引功能暂不可用: {e}")
    
    st.divider()
    
    # 搜索功能（支持向量搜索和关键词搜索）
    col1, col2 = st.columns([2, 1])
    
    with col1:
        search_query = st.text_input("🔍 搜索知识库", "")
    
    with col2:
        search_mode = st.radio(
            "搜索模式",
            options=["智能混合搜索", "纯关键词搜索", "纯向量搜索"],
            horizontal=True,
            key="search_mode"
        )
    
    if search_query:
        with st.spinner("正在搜索..."):
            if search_mode == "智能混合搜索":
                # 使用智能混合检索策略
                try:
                    from utils.knowledge_manager import search_knowledge_smart
                    search_result = search_knowledge_smart(search_query, max_results=5)
                    
                    # 从返回结果中提取详细信息
                    results = search_result.get("results", [])
                    strategy_used = search_result.get("strategy_used", "unknown")
                    
                    # 生成搜索类型标签
                    strategy_labels = {
                        "vector_primary": "🎯 智能检索（向量优先）",
                        "hybrid_supplement": "🔄 智能检索（补充检索）",
                        "keyword_fallback": "📝 智能检索（文本匹配）"
                    }
                    search_type = strategy_labels.get(strategy_used, f"智能检索 ({strategy_used})")
                    
                    # 在界面上显示检索策略信息
                    if strategy_used in ["hybrid_supplement", "keyword_fallback"]:
                        metadata = search_result.get("search_metadata", {})
                        vector_sim = metadata.get("vector_max_similarity", 0)
                        st.info(f"ℹ️ 向量检索相似度: {vector_sim:.2%}，已自动切换到文本匹配补充检索")
                    
                except Exception as e:
                    print(f"智能检索失败: {e}")
                    import traceback
                    traceback.print_exc()
                    # 回退到关键词搜索
                    results = search_knowledge(search_query)
                    search_type = "关键词搜索（异常回退）"
                    
            elif search_mode == "纯向量搜索":
                try:
                    from utils.knowledge_manager import search_knowledge_by_vector
                    results = search_knowledge_by_vector(search_query, max_results=5, use_hybrid=False)
                    search_type = "🎯 纯向量搜索"
                except Exception as e:
                    st.error(f"向量搜索失败: {e}")
                    results = []
                    search_type = "❌ 向量搜索失败"
            else:
                results = search_knowledge(search_query)
                search_type = "📝 关键词搜索"
            
            if results:
                st.success(f"✅ {search_type} - 找到 {len(results)} 个匹配结果")
                for i, res in enumerate(results, 1):
                    st.write(f"**{i}. {res['file_name']}**")
                    
                    # 显示相似度/得分
                    similarity = res.get("similarity", res.get("score", 0))
                    col1, col2 = st.columns([1, 3])
                    with col1:
                        st.progress(min(similarity, 1.0))
                    with col2:
                        if similarity > 0:
                            st.write(f"相关性: {similarity:.2%}")
                    
                    st.write(f"预览: {res.get('preview', '')}...")
                    
                    # 显示匹配类型标签
                    tags = []
                    if res.get("search_type"):
                        tags.append(f"🔍 {res['search_type']}")
                    if res.get("matched_words"):
                        tags.append(f"关键词: {', '.join(res['matched_words'][:3])}")
                    if tags:
                        st.write(" ".join([f"`{t}`" for t in tags]))
                    
                    st.divider()
            else:
                st.info("❌ 未找到匹配的知识库内容")
    else:
        # 显示所有知识库文件
        knowledge_list = get_knowledge_list()
        if knowledge_list:
            stats = get_knowledge_stats()
            col1, col2, col3 = st.columns(3)
            with col1:
                st.metric("总文件数", stats["total_files"])
            with col2:
                st.metric("总大小", f"{stats['total_size'] / 1024 / 1024:.2f} MB")
            with col3:
                st.metric("格式种类", len(stats["formats"]))

            st.divider()
            st.subheader("文件列表")
            
            selected_file = st.selectbox(
                "选择文件查看内容",
                [f["original_name"] for f in knowledge_list],
                index=0 if knowledge_list else None
            )
            
            if selected_file:
                for f in knowledge_list:
                    if f["original_name"] == selected_file:
                        st.write(f"**文件名**: {f['original_name']}")
                        st.write(f"**上传时间**: {f['upload_time'][:19].replace('T', ' ')}")
                        st.write(f"**大小**: {f['size'] / 1024:.1f} KB")
                        
                        content = get_knowledge_content(f["file_id"])
                        if content.startswith("文件解析失败") or content.startswith("不支持"):
                            st.warning(content)
                        else:
                            st.subheader("内容预览")
                            st.text_area("", content, height=300)
                        break
        else:
            st.info("📭 知识库为空，请在侧边栏上传专业知识文档")

with tab_audio:
    if uploaded_file:
        ext = uploaded_file.name.split(".")[-1].lower()
        audio_path = f"temp_audio.{ext}"
        with open(audio_path, "wb") as f:
            f.write(uploaded_file.getbuffer())
        st.audio(audio_path)
    else:
        st.info("请先上传音频文件（支持 WAV, MP3, M4A, FLAC 格式）")

if st.button("🚀 开始一键复盘", type="primary"):
    if not api_key:
        st.error("请先在 .env 文件中配置通义千问 API Key！")
    elif not uploaded_file:
        st.error("请先上传面试音频！")
    else:
        with st.status("正在进行深度分析...", expanded=True) as status:
            # 1. 解析 JD
            jd_context = "通用面试评价标准"
            if jd_file:
                st.write(f"🔍 正在解析上传的 JD 文件: {jd_file.name}...")
                ext = jd_file.name.split(".")[-1]
                temp_jd = f"temp_{uuid.uuid4().hex}.{ext}"
                with open(temp_jd, "wb") as f:
                    f.write(jd_file.getvalue())

                jd_context = process_jd_to_context(temp_jd)
                os.remove(temp_jd)

                if not jd_context or jd_context == "通用面试评价标准":
                    st.warning("⚠️ 岗位 JD 解析结果似乎为空。")
                else:
                    st.info(f"✅ JD 解析成功！识别到约 {len(jd_context)} 个字符。")

            # 2. 解析简历
            cv_context = st.session_state.cv_context
            if cv_file:
                st.write(f"📄 正在解析上传的简历: {cv_file.name}...")
                ext = cv_file.name.split(".")[-1]
                temp_cv = f"temp_cv_{uuid.uuid4().hex}.{ext}"
                with open(temp_cv, "wb") as f:
                    f.write(cv_file.getvalue())

                cv_context = process_jd_to_context(temp_cv)
                os.remove(temp_cv)
                st.session_state.cv_context = cv_context
                st.info("✅ 简历解析成功！")

            # 3. 历史记录对比
            st.write("正在提取历史表现以进行对比分析...")
            comparison_context = ""

            try:
                last_record = get_last_interview(current_user_id)

                if last_record:
                    ls = last_record.get("scores", {})
                    last_date = str(last_record.get("created_at", ""))[:10]
                    st.success(f"✅ 已成功调取历史档案 (最近一次: {last_date})")
                    comparison_context = (
                        f"\n【该用户历史表现】：上次语速 {last_record.get('avg_wpm')} WPM，"
                        f"技术评分 {ls.get('技术深度', 'N/A')}。"
                    )
                else:
                    st.write("ℹ️ 暂无历史记录，本次将作为您的首次面试档案。")
            except Exception as e:
                st.error(f"❌ 无法连接云端历史库: {e}")

            # 4. 语音转文字（WhisperX）
            st.write("正在转录音频（WhisperX）...")
            device = "cpu"

            # 禁用 TorchCodec（避免 FFmpeg DLL 加载问题）
            os.environ["TORCH_CODEC_AVAILABLE"] = "0"

            try:
                whisper_model_path = custom_model_path if custom_model_path else model_size
                whisper_model = whisperx.load_model(
                    whisper_model_path, device, compute_type="int8"
                )
            except Exception as e:
                st.error(f"❌ Whisper 模型加载失败: {e}")
                st.info(
                    """
                💡 手动下载模型步骤：
                1. 访问 https://huggingface.co/openai/whisper-tiny
                2. 下载所有文件到本地目录
                3. 在左侧"自定义模型路径"输入框中指定路径
                """
                )
                st.stop()

            try:
                audio_data = load_audio_file(audio_path)
            except Exception as e:
                st.error(f"❌ {e}")
                st.stop()

            result = whisper_model.transcribe(audio_data, batch_size=16, language="zh")

            st.write("正在校准时间戳并计算指标...")

            try:
                # 再次禁用 TorchCodec（确保生效）
                os.environ["TORCH_CODEC_AVAILABLE"] = "0"
                model_a, metadata = whisperx.load_align_model(
                    language_code="zh", device=device
                )
                aligned_result = whisperx.align(
                    result["segments"], model_a, metadata, audio_data, device
                )
            except Exception as e:
                st.warning(f"⚠️ 时间戳校准失败，将使用原始转录结果: {e}")
                aligned_result = result

            df_wpm = calculate_wpm(aligned_result["segments"])
            full_transcript = ""
            for segment in aligned_result["segments"]:
                full_transcript += f"[{segment['start']:.2f}s] {cc.convert(segment['text'])}\n"

            # 5. 调用大模型生成复盘
            st.write("正在呼叫通义千问生成定制化复盘报告...")

            client, client_error = create_openai_client()
            if client_error:
                st.error(f"❌ {client_error}")
                st.stop()

            # 匹配知识库内容
            knowledge_summary = ""
            knowledge_prompt = ""
            if knowledge_list:
                st.write("🔍 正在匹配知识库内容...")
                knowledge_summary = get_relevant_knowledge_summary(full_transcript)
                st.info(knowledge_summary)
                
                # 生成知识库提示词
                knowledge_prompt = generate_prompt_from_knowledge(full_transcript, jd_context)

            prompt = (
                f"你是一个资深面试官。请参考[岗位要求]、[个人简历]和[参考知识库]评估本次[面试文本]。"
                f"{comparison_context}\n\n"
                "任务：\n"
                "1. 诊断技术点与逻辑。\n"
                "2. 给出\"成长对比\"：对比历史表现（若有），分析是否有进步。\n"
                "3. 新增\"简历 vs 表现\"差异分析：对比[个人简历]，指出用户有哪些简历中的亮点在本次面试中被忽略了，或哪些表达与简历不符。\n"
                "4. 如果有[参考知识库]，请结合知识库中的相关内容进行分析和建议。\n"
                '5. 结尾包含 JSON 评分 Scores: {"技术深度": 分数, "逻辑表达": 分数, "自信度": 分数, "沟通技巧": 分数, "岗位匹配度": 分数}\n\n'
                f"[岗位要求]: {jd_context}\n"
                f"[个人简历]: {cv_context}\n"
                f"[面试文本]: {full_transcript}\n"
                f"[参考知识库]: {knowledge_summary}"
            )

            try:
                response = client.chat.completions.create(
                    model=model_name,
                    messages=[{"role": "user", "content": prompt}],
                )
                coach_feedback = response.choices[0].message.content
            except Exception as e:
                st.error(f"❌ AI 分析失败: {type(e).__name__}: {e}")
                with st.expander("详细错误"):
                    st.code(traceback.format_exc())
                st.stop()

            st.session_state.df_wpm = df_wpm
            st.session_state.scores = extract_scores(coach_feedback)
            st.session_state.report = coach_feedback
            st.session_state.transcript = full_transcript
            st.session_state.jd_context = jd_context

            try:
                save_interview_result(
                    avg_wpm=float(df_wpm["wpm"].mean()) if not df_wpm.empty else 0,
                    scores=st.session_state.scores,
                    report=coach_feedback,
                    transcript=full_transcript,
                    user_id=current_user_id,
                )
                st.toast("✅ 数据已保存", icon="💾")
            except Exception as e:
                st.error(f"数据保存失败: {e}")

            status.update(label="复盘完成！", state="complete", expanded=False)

if st.session_state.report:
    with tab_metrics:
        st.subheader("📊 面试表现量化看板")
        col1, col2 = st.columns(2)
        with col1:
            st.write("📈 语速波动曲线 (WPM)")
            st.line_chart(st.session_state.df_wpm.set_index("start")["wpm"])
            st.download_button(
                "📥 下载语速原始数据 (CSV)",
                data=st.session_state.df_wpm.to_csv(index=False).encode("utf-8"),
                file_name="wpm_data.csv",
            )

        with col2:
            st.write("🕸️ 能力画像雷达图")
            if st.session_state.scores:
                fig = generate_radar_chart(st.session_state.scores)
                st.plotly_chart(fig, width="stretch")
            else:
                st.warning("⚠️ AI 报告格式不符合预期，未能提取到量化评分。")

    with tab_report:
        st.subheader("🤖 AI 教练深度复盘报告")
        if "简历" in st.session_state.report:
            st.info("💡 本次报告已包含简历对比分析，请查看\"简历 vs 表现\"章节。")
        st.markdown(st.session_state.report)
        st.download_button(
            "📥 下载 Markdown 复盘报告",
            data=st.session_state.report,
            file_name="analysis.md",
        )
        st.divider()
        st.subheader("📝 详细转录文本")
        st.download_button(
            "📥 下载详细转录文本 (TXT)",
            data=st.session_state.transcript,
            file_name="transcript.txt",
        )
        st.text_area("内容", st.session_state.transcript, height=400)

with tab_history:
    st.subheader("🚀 你的面试进化史")
    
    # 显示存储状态
    status = get_storage_status()
    with st.expander("📊 存储系统状态"):
        st.json(status)
    
    # 手动同步按钮
    if st.button("🔄 同步数据到云端"):
        success, fail = sync_to_cloud()
        st.info(f"同步完成: 成功 {success} 条, 失败 {fail} 条")
    
    history_data = get_all_history(current_user_id)

    if not history_data:
        st.info("暂无历史数据，去完成你的第一次面试吧！")
    else:
        h_dates = [d[0] for d in history_data]
        h_wpms = [d[1] for d in history_data]

        st.write("📊 语速平稳度趋势")
        st.line_chart(pd.DataFrame({"语速 (WPM)": h_wpms}, index=h_dates))

        st.write("📈 核心维度得分走势")
        score_list = []
        for d in history_data:
            s_dict = d[2] if isinstance(d[2], dict) else (json.loads(d[2]) if d[2] else {})
            s_dict["日期"] = d[0]
            score_list.append(s_dict)

        st.line_chart(pd.DataFrame(score_list).set_index("日期"))

with tab_assistant:
    st.write("### 🤖 AI 面试教练 Agent")
    st.info("💡 我是具备\"思考-行动-观察\"循环能力的智能Agent，可以根据你的问题自动调用工具获取相关信息。")
    
    # ==================== 上下文压缩控制 ====================
    with st.expander("🗜️ 上下文压缩控制", expanded=False):
        col1, col2 = st.columns([1, 2])
        
        with col1:
            # 启用/禁用自动压缩
            st.session_state.compression_enabled = st.checkbox(
                "启用自动压缩",
                value=st.session_state.compression_enabled,
                help="当对话上下文过长时自动进行语义压缩"
            )
        
        with col2:
            # 手动触发压缩按钮
            if st.button("🔄 手动压缩上下文", type="secondary"):
                if len(st.session_state.messages) > 3:
                    with st.spinner("正在压缩上下文..."):
                        # 执行压缩
                        compressed = compress_context(st.session_state.messages, use_llm=False)
                        
                        if compressed.compressed_count < compressed.original_count:
                            st.session_state.messages = compressed.compressed_messages
                            st.session_state.last_compression_info = {
                                "original_count": compressed.original_count,
                                "compressed_count": compressed.compressed_count,
                                "compression_ratio": f"{compressed.compression_ratio:.1%}",
                                "timestamp": compressed.timestamp,
                                "summary": compressed.summary,
                                "core_topics": compressed.core_topics
                            }
                            st.success(f"✅ 上下文已压缩：{compressed.original_count} → {compressed.compressed_count} 条消息 (压缩率: {compressed.compression_ratio:.1%})")
                        else:
                            st.info("当前上下文无需压缩")
                else:
                    st.info("对话消息太少，无需压缩")
        
        # 显示上次压缩信息
        if st.session_state.last_compression_info:
            st.write("**📊 最近压缩信息：**")
            info = st.session_state.last_compression_info
            col1, col2, col3 = st.columns(3)
            with col1:
                st.metric("原始消息", info["original_count"])
            with col2:
                st.metric("压缩后", info["compressed_count"])
            with col3:
                st.metric("压缩率", info["compression_ratio"])
            
            if info.get("core_topics"):
                st.write(f"**核心话题：** {', '.join(info['core_topics'])}")
            if info.get("summary"):
                with st.expander("📝 上下文摘要", expanded=False):
                    st.info(info["summary"])
    
    # ==================== 显示对话历史 ====================
    # 显示对话历史
    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])

    # ==================== 用户输入处理 ====================
    if user_query := st.chat_input("有什么我可以帮助你的吗？"):
        # 添加用户消息
        st.session_state.messages.append({"role": "user", "content": user_query})
        with st.chat_message("user"):
            st.markdown(user_query)

        # ==================== 自动压缩检查 ====================
        # 检查是否需要自动压缩
        should_compress, reason = check_compression_needed(st.session_state.messages)
        
        if should_compress and st.session_state.compression_enabled:
            # 显示即将压缩的提示
            if not st.session_state.auto_compression_warning_shown:
                st.info(f"🔔 上下文已达到压缩阈值 ({reason})，将在本次对话后自动压缩...")
                st.session_state.auto_compression_warning_shown = True
        
        # ==================== Agent 响应 ====================
        with st.chat_message("assistant"):
            with st.spinner("🤔 正在思考并分析问题..."):
                try:
                    # 使用新的 Agent 系统
                    response = agent_chat(user_query, current_user_id)
                    st.markdown(response)
                    
                    # 添加助手响应
                    st.session_state.messages.append(
                        {"role": "assistant", "content": response}
                    )
                    
                    # ==================== 自动压缩执行 ====================
                    # 在对话后检查并执行自动压缩
                    if st.session_state.compression_enabled:
                        should_compress_after, reason_after = check_compression_needed(st.session_state.messages)
                        
                        if should_compress_after:
                            with st.spinner("🗜️ 正在自动压缩上下文..."):
                                # 执行压缩
                                compressed = compress_context(st.session_state.messages, use_llm=False)
                                
                                if compressed.compressed_count < compressed.original_count:
                                    st.session_state.messages = compressed.compressed_messages
                                    st.session_state.last_compression_info = {
                                        "original_count": compressed.original_count,
                                        "compressed_count": compressed.compressed_count,
                                        "compression_ratio": f"{compressed.compression_ratio:.1%}",
                                        "timestamp": compressed.timestamp,
                                        "summary": compressed.summary,
                                        "core_topics": compressed.core_topics
                                    }
                                    st.session_state.auto_compression_warning_shown = False
                                    st.rerun()  # 重新渲染以显示压缩后的状态
                                    
                except Exception as e:
                    st.error(f"❌ Agent 调用失败: {type(e).__name__}: {e}")
                    with st.expander("详细错误"):
                        st.code(traceback.format_exc())
