import os
import base64
import uuid
import mimetypes
from dotenv import load_dotenv
load_dotenv()

try:
    from langchain_community.document_loaders import TextLoader, PyPDFLoader
    LANGCHAIN_AVAILABLE = True
except ImportError:
    LANGCHAIN_AVAILABLE = False

try:
    from langchain_huggingface import HuggingFaceEmbeddings
    EMBEDDINGS_AVAILABLE = True
except ImportError:
    EMBEDDINGS_AVAILABLE = False

try:
    from langchain_community.vectorstores import Chroma
    CHROMADB_AVAILABLE = True
except ImportError:
    CHROMADB_AVAILABLE = False

try:
    from langchain_text_splitters import RecursiveCharacterTextSplitter
    TEXT_SPLITTER_AVAILABLE = True
except ImportError:
    TEXT_SPLITTER_AVAILABLE = False

try:
    from PIL import Image
    PIL_AVAILABLE = True
except ImportError:
    PIL_AVAILABLE = False

try:
    from openai import OpenAI
    OPENAI_AVAILABLE = True
except ImportError:
    OPENAI_AVAILABLE = False

DASHSCOPE_API_KEY = os.getenv("DASHSCOPE_API_KEY", "")
QWEN_MODEL = os.getenv("QWEN_MODEL", "qwen-max")
QWEN_VL_MODEL = os.getenv("QWEN_VL_MODEL", "qwen-vl-max")

EMBEDDING_MODEL = "sentence-transformers/all-MiniLM-L6-v2"
DASHSCOPE_BASE_URL = "https://dashscope.aliyuncs.com/compatible-mode/v1"

MAX_IMAGE_SIZE_MB = 20
MAX_IMAGE_SIZE_BYTES = MAX_IMAGE_SIZE_MB * 1024 * 1024
SUPPORTED_IMAGE_FORMATS = {"jpg", "jpeg", "png", "webp", "bmp", "gif"}
SUPPORTED_TEXT_FORMATS = {"pdf", "txt", "md", "doc", "docx"}


def _safe_ascii(text: str) -> str:
    """确保文本在作为消息参数时不会引发编码问题"""
    return text.strip()


def _get_client():
    """获取 OpenAI 兼容客户端，确保 API key 为纯 ASCII"""
    if not OPENAI_AVAILABLE:
        return None, "未安装 openai 库，请运行: pip install openai"

    api_key_raw = DASHSCOPE_API_KEY
    if not api_key_raw:
        return None, "未配置 DASHSCOPE_API_KEY，请在 .env 文件中设置"

    try:
        api_key = api_key_raw.encode("ascii").decode("ascii")
    except UnicodeEncodeError:
        return None, "API Key 包含非 ASCII 字符"

    try:
        client = OpenAI(api_key=api_key, base_url=DASHSCOPE_BASE_URL)
        return client, None
    except Exception as e:
        return None, f"创建客户端失败: {e}"


def _safe_model_name(name: str) -> str:
    """确保模型名为纯 ASCII"""
    name = name.strip()
    try:
        return name.encode("ascii").decode("ascii")
    except UnicodeEncodeError:
        return "qwen-plus"


def _validate_image_file(file_path: str) -> tuple[bool, str]:
    """验证图片文件是否有效，返回 (是否有效, 错误信息/格式信息)"""
    if not os.path.exists(file_path):
        return False, f"文件不存在: {file_path}"

    file_size = os.path.getsize(file_path)
    if file_size == 0:
        return False, "文件为空"

    if file_size > MAX_IMAGE_SIZE_BYTES:
        return False, f"文件过大 ({file_size / 1024 / 1024:.1f}MB)，最大支持 {MAX_IMAGE_SIZE_MB}MB"

    ext = file_path.split(".")[-1].lower()
    if ext not in SUPPORTED_IMAGE_FORMATS:
        return False, f"不支持的图片格式: {ext}，支持格式: {', '.join(sorted(SUPPORTED_IMAGE_FORMATS))}"

    if PIL_AVAILABLE:
        try:
            with Image.open(file_path) as img:
                img.verify()
                format_info = img.format
                width, height = img.size
                if width <= 0 or height <= 0:
                    return False, "图片尺寸无效"
                return True, f"{format_info} ({width}x{height})"
        except Exception as e:
            return False, f"图片损坏或无法解析: {e}"
    else:
        return True, f"{ext.upper()} 格式（建议安装 Pillow 以获得更好的图片验证: pip install Pillow）"


def _get_mime_type(file_path: str) -> str:
    """获取正确的 MIME 类型"""
    ext = file_path.split(".")[-1].lower()
    mime_map = {
        "jpg": "image/jpeg",
        "jpeg": "image/jpeg",
        "png": "image/png",
        "webp": "image/webp",
        "bmp": "image/bmp",
        "gif": "image/gif",
    }
    return mime_map.get(ext, "image/jpeg")


def process_image_with_vision(file_path: str) -> str:
    """使用视觉模型解析图片（JD/简历）"""
    valid, info = _validate_image_file(file_path)
    if not valid:
        return f"图片验证失败: {info}"

    model_name = QWEN_VL_MODEL

    client, error = _get_client()
    if error:
        return f"图片解析失败: {error}"

    try:
        with open(file_path, "rb") as image_file:
            image_bytes = image_file.read()

        base64_image = base64.b64encode(image_bytes).decode("utf-8")
        mime = _get_mime_type(file_path)

        response = client.chat.completions.create(
            model=model_name,
            messages=[
                {
                    "role": "user",
                    "content": [
                        {
                            "type": "text",
                            "text": "请提取这张图片中的所有文字内容。如果是岗位描述，重点提取：岗位名称、核心职责、任职要求、技术栈、学历/经验要求；如果是个人简历，重点提取：姓名、学历、工作经验、项目经历、技能栈。请以纯文本输出。",
                        },
                        {
                            "type": "image_url",
                            "image_url": {"url": f"data:{mime};base64,{base64_image}"},
                        },
                    ],
                }
            ],
            max_tokens=2000,
        )
        result = response.choices[0].message.content or ""
        if not result.strip():
            return "图片解析返回空内容，请检查图片是否清晰可读"
        return _safe_ascii(result)

    except Exception as e:
        error_msg = str(e)
        if "rate limit" in error_msg.lower() or "quota" in error_msg.lower():
            return "图片解析失败: API 调用频率过高或配额不足，请稍后重试"
        elif "timeout" in error_msg.lower():
            return "图片解析失败: 请求超时，请稍后重试"
        elif "invalid" in error_msg.lower() and "image" in error_msg.lower():
            return "图片解析失败: 图片格式不被模型支持，请尝试转换为 JPG 或 PNG 格式"
        else:
            return f"图片解析失败: {error_msg[:200]}"


def process_jd_to_context(file_path: str) -> str:
    """
    解析 JD / 简历文件为文本 context

    支持格式：
    - 图片 (png/jpg/jpeg/webp/bmp/gif)：调用视觉模型识别
    - PDF：使用 PyPDFLoader 解析
    - TXT/MD：使用 TextLoader 解析
    """
    if not os.path.exists(file_path):
        return f"文件解析失败: 文件不存在 - {file_path}"

    if os.path.getsize(file_path) == 0:
        return "文件解析失败: 文件为空"

    ext = file_path.split(".")[-1].lower()

    # 图片格式：用 VL 模型识别
    if ext in SUPPORTED_IMAGE_FORMATS:
        result = process_image_with_vision(file_path)
        if result.startswith("图片解析失败") or result.startswith("图片验证失败"):
            return f"{result}\n\n💡 建议：请确保图片格式为 JPG/PNG，文件大小不超过 {MAX_IMAGE_SIZE_MB}MB，并且内容清晰可读。"
        return result

    # 文本格式：用 RAG 解析
    if ext in SUPPORTED_TEXT_FORMATS:
        # 检查必要的依赖
        if not LANGCHAIN_AVAILABLE:
            return f"文件解析失败: 缺少 langchain 依赖，请安装: pip install langchain langchain-community"

        try:
            # 1. 根据文件类型加载
            if ext == "pdf":
                try:
                    loader = PyPDFLoader(file_path)
                    documents = loader.load()
                except Exception as pdf_error:
                    return f"PDF 解析失败: {pdf_error}\n\n💡 建议：请确保 PDF 文件未损坏，或安装 pypdf: pip install pypdf"
            else:
                try:
                    loader = TextLoader(file_path, encoding="utf-8")
                    documents = loader.load()
                except UnicodeDecodeError:
                    try:
                        loader = TextLoader(file_path, encoding="gbk")
                        documents = loader.load()
                    except Exception as txt_error:
                        return f"文本文件读取失败: {txt_error}"

            if not documents or all(not doc.page_content.strip() for doc in documents):
                return "文件解析失败: 文件内容为空"

            # 2. 文本切片（JD 通常不长，切成 500 字一块）
            if TEXT_SPLITTER_AVAILABLE:
                text_splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=50)
                docs = text_splitter.split_documents(documents)
            else:
                docs = documents

            # 3. 如果向量库可用，使用向量检索；否则直接返回全文
            if EMBEDDINGS_AVAILABLE and CHROMADB_AVAILABLE:
                try:
                    embeddings = HuggingFaceEmbeddings(model_name=EMBEDDING_MODEL)
                    vectorstore = Chroma.from_documents(
                        docs,
                        embeddings,
                        persist_directory=f"./chroma_{uuid.uuid4().hex[:8]}"
                    )
                    retriever = vectorstore.as_retriever(search_kwargs={"k": min(3, len(docs))})
                    relevant_docs = retriever.invoke("岗位职责、任职要求、技术栈、核心能力")
                    context = "\n".join([doc.page_content for doc in relevant_docs])
                except Exception as vector_error:
                    full_text = "\n".join([doc.page_content for doc in docs])
                    return full_text[:3000]
            else:
                full_text = "\n".join([doc.page_content for doc in docs])
                context = full_text[:3000]

            if not context.strip():
                return "文件解析失败: 未能提取有效内容"

            return context

        except Exception as e:
            return f"文件解析失败: {str(e)[:200]}"

    return f"不支持的文件格式: {ext}。支持格式: 图片 ({', '.join(sorted(SUPPORTED_IMAGE_FORMATS))}) / 文本 ({', '.join(sorted(SUPPORTED_TEXT_FORMATS))})"
