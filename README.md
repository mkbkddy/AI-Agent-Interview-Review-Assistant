# AI Interview Coach - AI 面试教练系统

> 面向求职场景的 AI 面试分析与训练系统。基于语音转录、结构化 Prompt 与多阶段分析机制，实现面试内容解析、技术诊断、追问训练与成长轨迹建模。

***

## 📖 目录

- [✨ 项目简介](#-项目简介)
- [🎯 核心功能](#-核心功能)
- [🛠️ 技术栈](#️-技术栈)
- [📁 项目结构](#-项目结构)
- [🚀 快速开始](#-快速开始)
- [⚙️ 配置说明](#️-配置说明)
- [🌐 API 端点一览](#-api-端点一览)
- [📚 功能模块详解](#-功能模块详解)

***

## ✨ 项目简介

AI Interview Coach 是一个基于 Streamlit 构建的智能面试分析与训练系统，帮助求职者：

- 📝 上传面试音频/文本进行 AI 分析
- 🎯 获取多维度的技术与表达能力诊断
- 🤖 进行 AI 追问训练与实时反馈
- 📊 追踪历史面试表现与成长轨迹
- 📚 管理专业知识库，支持智能检索

系统采用模块化设计，集成了向量检索、混合搜索、智能 Agent 与上下文压缩等先进技术。

***

## 🎯 核心功能

### 1. 多阶段 AI 面试分析流程

针对长文本面试转录场景，设计多阶段 AI 分析流程，对转录内容进行证据提取、技能归因与关键问题归因分析，并通过阶段拆分与结构化摘要降低长上下文分析中的信息干扰问题。

| 阶段     | 说明                                     |
| ------ | -------------------------------------- |
| 文件上传   | 支持 MP3/WAV/M4A/FLAC 等音频格式，进行 MIME 魔数校验 |
| ASR 转写 | 基于 WhisperX 进行语音识别与说话人分离               |
| 量化指标   | 统计语速 WPM、填充词密度、长停顿与沉默比等表达特征            |
| LLM 分析 | 基于结构化 Prompt 与双消息架构生成分析结果              |
| 深度诊断   | 执行证据提取、转录分段、能力画像与质量评估                  |
| 成长轨迹   | 对历史面试数据进行趋势分析与阶段性对比                    |

### 2. 语音表达能力分析

基于 ASR 与说话人分离实现面试音频结构化处理，并结合语速、停顿与填充词等时序特征构建表达能力指标体系，用于定位连续追问场景下的表达波动区间。

**关键指标：**

- 📊 WPM（Words Per Minute）语速统计
- 💬 填充词频率与分钟级分布
- ⏸️ 长停顿检测
- 🎚️ 沉默时间占比

### 3. 简历评分与 JD 匹配

支持上传职位描述（JD）与简历（CV），通过 AI 分析两者的匹配度，生成个性化的评分报告与改进建议。

- 自动解析 JD 中的技能要求
- 分析 CV 与 JD 的匹配程度
- 生成改进建议与提升方向

### 4. 智能知识库管理

构建专业知识库，支持多格式文档上传与智能检索：

- 📄 支持 PDF、TXT、DOCX 等格式
- 🔍 关键词搜索 + 向量搜索 + 混合搜索
- 🧠 基于语义的智能检索与排序
- 📋 MD5 内容去重，避免重复上传

### 5. AI Agent 对话系统

具备"思考-行动-观察"循环能力的智能 Agent，可以根据用户问题自动调用工具获取相关信息：

- 🔧 工具注册与动态调用机制
- 💭 多轮对话记忆与上下文管理
- 🗜️ 自动上下文压缩（智能减少 token 消耗）
- 🔄 手动触发压缩功能

### 6. 三级数据存储机制

设计"云端 → 缓存 → 本地"的三级存储架构，确保数据持久性与访问性能：

- ☁️ **Supabase (PostgreSQL)**: 云端持久化存储
- ⚡ **Redis**: 缓存加速（可选，未安装时自动降级为本地缓存）
- 💾 **本地 JSON**: 离线数据备份

### 7. 智能检索系统

- **向量检索**: 基于 sentence-transformers 的语义理解
- **关键词检索**: 基于中文分词的精确匹配
- **智能混合搜索**: 自动选择最优检索策略
- **段落级分块**: 保留完整语义单元

### 8. 上下文压缩策略

- 自动触发：对话长度或消息数超过阈值时自动压缩
- 手动触发：用户可主动压缩上下文
- 摘要保留：压缩后保留核心话题与摘要信息

***

## 🛠️ 技术栈

### 前端 / UI 框架

| 技术                                 | 用途             |
| ---------------------------------- | -------------- |
| [Streamlit](https://streamlit.io/) | 快速构建交互式 Web 应用 |
| Python 3.10+                       | 主编程语言          |

### AI / 大语言模型

| 技术                                                             | 用途        |
| -------------------------------------------------------------- | --------- |
| [Alibaba Qwen (通义千问)](https://help.aliyun.com/zh/model-studio) | 大语言模型推理   |
| [WhisperX](https://github.com/m-bain/whisperX)                 | 语音识别与字级对齐 |
| [sentence-transformers](https://www.sbert.net/)                | 文本向量编码    |

### 数据库与存储

| 技术                                | 用途             |
| --------------------------------- | -------------- |
| [Supabase](https://supabase.com/) | 云端数据持久化存储与历史查询 |
| [Redis](https://redis.io/)        | 缓存与限流（可选）      |
| 本地 JSON 文件                        | 本地数据备份与离线存储    |

### 数据处理与分析

| 技术                                         | 用途      |
| ------------------------------------------ | ------- |
| [Pandas](https://pandas.pydata.org/)       | 数据处理与分析 |
| [NumPy](https://numpy.org/)                | 数值计算    |
| [OpenCC](https://github.com/BYVoid/OpenCC) | 中文简繁转换  |
| [Pydub](https://github.com/jiaaro/pydub)   | 音频处理    |

### 检索与 RAG

| 技术                                                 | 用途         |
| -------------------------------------------------- | ---------- |
| [FAISS](https://github.com/facebookresearch/faiss) | 向量相似度搜索    |
| [LangChain](https://www.langchain.com/)            | RAG 框架与检索链 |
| jieba                                              | 中文分词       |

### 图像处理

| 技术                                   | 用途         |
| ------------------------------------ | ---------- |
| [OpenCV](https://opencv.org/)        | JD/简历图片预处理 |
| [Pillow](https://python-pillow.org/) | 图像读取与转换    |

### 文档解析

| 技术          | 用途        |
| ----------- | --------- |
| python-docx | Word 文档解析 |
| PyPDF2      | PDF 文档解析  |

***

## 📁 项目结构

```
AI-Interview-Coach-main/
├── app.py                          # Streamlit 主应用入口（UI + 核心逻辑）
├── run_interview.py                # 面试分析命令行工具
├── requirements.txt                # Python 依赖包清单
├── packages.txt                    # 系统依赖包（Debian/Ubuntu）
├── .env.example                    # 环境变量模板（复制为 .env 使用）
├── README.md                       # 项目说明文档
├──
├── utils/                          # 工具模块目录
│   ├── agent.py                    # AI Agent 模块（思考-行动-观察循环）
│   ├── knowledge_manager.py        # 知识库管理（上传/搜索/匹配）
│   ├── vector_index.py             # 向量索引与相似度搜索
│   ├── context_compressor.py       # 上下文压缩与管理
│   ├── triple_tier_storage.py      # 三级数据存储（云+缓存+本地）
│   ├── rag_engine.py               # RAG 引擎（JD/CV 处理）
│   ├── hybrid_search.py            # 混合搜索（向量 + 关键词）
│   ├── db_manager.py               # 数据库管理
│   └── metrics.py                  # 量化指标与图表生成
├──
├── knowledge_base/                 # 知识库文件存储目录（运行时自动创建）
│   ├── files/                      # 上传的原始文件
│   ├── index.json                  # 文件索引与元数据
│   └── vector_index/               # 向量索引数据（可选）
├──
├── data/                           # 本地数据目录（运行时自动创建）
│   ├── interviews/                 # 面试记录 JSON
│   └── user_data/                  # 用户数据
├──
└── test_*.py                       # 各类测试脚本（可选运行）
```

***

## 🚀 快速开始

### 步骤 1: 环境要求

- **Python**: 3.10 或更高版本
- **操作系统**: Windows / macOS / Linux
- **磁盘空间**: 至少 2GB（用于 WhisperX 模型）
- **网络**: 需要能访问阿里云 DashScope API

### 步骤 2: 克隆项目

```bash
git clone https://github.com/your-username/AI-Interview-Coach.git
cd AI-Interview-Coach
```

### 步骤 3: 安装系统依赖（Linux）

```bash
# Ubuntu/Debian 系统
sudo apt-get update
sudo apt-get install -y ffmpeg
```

> Windows/macOS 用户：FFmpeg 会通过 Python 的 pydub 自动处理，或手动安装：
>
> - Windows: `choco install ffmpeg` 或下载后添加到 PATH
> - macOS: `brew install ffmpeg`

### 步骤 4: 创建虚拟环境并安装 Python 依赖

```bash
# 创建虚拟环境（推荐）
python -m venv .venv

# 激活虚拟环境
# Windows (PowerShell):
.venv\Scripts\Activate.ps1
# Windows (CMD):
.venv\Scripts\activate.bat
# macOS/Linux:
source .venv/bin/activate

# 安装依赖
pip install -r requirements.txt
```

> 💡 **提示**: 首次安装 WhisperX 可能需要较长时间，因为需要下载模型文件。
>
> 💡 **提示**: 如果 `faiss-cpu` 安装失败，可以尝试安装 `faiss-cpu` 的预编译版本。

### 步骤 5: 配置环境变量

复制 `.env.example` 为 `.env` 并填入你的配置：

```bash
# 复制环境变量模板
cp .env.example .env
```

编辑 `.env` 文件：

```env
# ============== 阿里千问 API（必需） ==============
# 获取地址: https://dashscope.console.aliyun.com/
DASHSCOPE_API_KEY=your_dashscope_api_key_here
QWEN_MODEL=qwen-max
QWEN_VL_MODEL=qwen-vl-max
DASHSCOPE_BASE_URL=https://dashscope.aliyuncs.com/compatible-mode/v1

# ============== Supabase（可选，用于云端存储） ==============
# 获取地址: https://supabase.com/
SUPABASE_URL=
SUPABASE_KEY=
SUPABASE_SERVICE_ROLE_KEY=

# ============== Redis（可选，用于缓存加速） ==============
REDIS_HOST=localhost
REDIS_PORT=6379
REDIS_PASSWORD=

# ============== 应用配置 ==============
APP_VERSION=1.0.0
DEBUG=false
```

> **注意**: 最少只需要配置 `DASHSCOPE_API_KEY` 即可运行核心功能。

### 步骤 6: 启动应用

```bash
# 启动 Streamlit Web 应用
streamlit run app.py
```

启动后，浏览器会自动打开 `http://localhost:8501`

### 可选：命令行使用

```bash
# 直接运行面试分析（命令行模式）
python run_interview.py
```

***

## ⚙️ 配置说明

### 环境变量详解

| 环境变量                             | 必填 | 默认值                                                 | 说明                            |
| -------------------------------- | -- | --------------------------------------------------- | ----------------------------- |
| **DASHSCOPE\_API\_KEY**          | ✅  | —                                                   | 阿里云 DashScope API Key，用于大模型推理 |
| **QWEN\_MODEL**                  | ❌  | qwen-max                                            | 对话使用的千问模型名称                   |
| **QWEN\_VL\_MODEL**              | ❌  | qwen-vl-max                                         | 多模态（图片）模型名称                   |
| **DASHSCOPE\_BASE\_URL**         | ❌  | `https://dashscope.aliyuncs.com/compatible-mode/v1` | API 基础地址                      |
| **SUPABASE\_URL**                | ❌  | —                                                   | Supabase 项目 URL（启用云端存储）       |
| **SUPABASE\_KEY**                | ❌  | —                                                   | Supabase anon public key      |
| **SUPABASE\_SERVICE\_ROLE\_KEY** | ❌  | —                                                   | Supabase service\_role 密钥     |
| **REDIS\_HOST**                  | ❌  | localhost                                           | Redis 主机地址                    |
| **REDIS\_PORT**                  | ❌  | 6379                                                | Redis 端口                      |
| **REDIS\_PASSWORD**              | ❌  | —                                                   | Redis 密码                      |
| **APP\_VERSION**                 | ❌  | 1.0.0                                               | 应用版本号                         |
| **DEBUG**                        | ❌  | false                                               | 是否启用调试模式                      |

### 获取 API Key

#### 1. 阿里千问 API Key

1. 访问 [阿里云 DashScope 控制台](https://dashscope.console.aliyun.com/)
2. 注册/登录账号
3. 进入「API Key 管理」
4. 创建新的 API Key，复制到 `.env`

> 💡 新用户可以免费领取一定额度的 API 调用次数

#### 2. Supabase（可选）

1. 访问 [Supabase](https://supabase.com/) 并注册
2. 创建新项目
3. 在「Settings → API」中获取 `URL` 和 `anon key`

> 💡 不配置 Supabase 时，数据会自动保存在本地 JSON 文件中

#### 3. Redis（可选）

本地安装 Redis 即可，或使用云端 Redis 服务。未配置时自动使用本地文件缓存。

***

## 🌐 API 端点一览

> 本项目基于 Streamlit 构建，采用单页应用模式。以下为模块功能对应的 Streamlit Tab 与对应后端逻辑。

| 功能模块        | Tab 名称      | 主要功能               | 相关文件                                     |
| ----------- | ----------- | ------------------ | ---------------------------------------- |
| 📝 面试分析     | `上传音频 / 文本` | 上传面试音频或文本进行 AI 分析  | `app.py`, `utils/rag_engine.py`          |
| 📊 量化分析     | `量化指标`      | 显示语速、填充词、停顿等表达特征图表 | `app.py`, `utils/metrics.py`             |
| 🤖 AI 对话    | `AI 对话`     | 基于历史面试数据的智能问答      | `app.py`, `utils/agent.py`               |
| 📚 知识库      | `知识库管理`     | 上传/搜索专业知识文档        | `app.py`, `utils/knowledge_manager.py`   |
| 📋 简历评分     | `简历评分`      | JD 与 CV 匹配分析       | `app.py`, `utils/rag_engine.py`          |
| 🤖 AI Agent | `AI Agent`  | 思考-行动-观察循环的智能助手    | `app.py`, `utils/agent.py`               |
| 📈 历史记录     | `历史记录`      | 查看历史面试与趋势对比        | `app.py`, `utils/triple_tier_storage.py` |

### 程序入口与调用链

```
用户操作 (Streamlit UI)
    ↓
app.py (UI 控制 + 业务逻辑调度)
    ├─> utils/agent.py              # AI Agent 对话
    ├─> utils/knowledge_manager.py  # 知识库管理
    │   └─> utils/vector_index.py   # 向量索引检索
    ├─> utils/rag_engine.py         # JD/CV 处理
    ├─> utils/triple_tier_storage.py # 数据存储
    │   └─> utils/db_manager.py     # 数据库管理
    ├─> utils/metrics.py            # 指标计算
    ├─> utils/hybrid_search.py      # 混合搜索
    └─> utils/context_compressor.py # 上下文压缩
```

***

## 📚 功能模块详解

### 知识库系统（knowledge\_manager.py）

**功能**: 管理专业知识文档，支持智能检索与去重

**核心能力**:

- 文件上传（PDF/TXT/DOCX）
- MD5 内容去重
- 关键词搜索、向量搜索、混合搜索
- 段落级分块与匹配

**使用示例**:

```python
from utils.knowledge_manager import search_knowledge_smart

# 智能检索（自动选择最优策略）
results = search_knowledge_smart("Spring Boot 启动流程", max_results=5)
for r in results["results"]:
    print(f"{r['file_name']} - 相似度: {r.get('similarity', 0):.2%}")
```

### 向量索引系统（vector\_index.py）

**功能**: 将知识库文档转换为向量，实现语义相似度检索

**核心能力**:

- 使用 paraphrase-multilingual-MiniLM-L12-v2 模型编码
- 段落级智能分块
- 余弦相似度计算
- 支持增量索引

### AI Agent（agent.py）

**功能**: 具备"思考-行动-观察"循环能力的智能对话代理

**核心能力**:

- 工具注册与动态调用
- 对话记忆管理
- 自动选择合适工具响应查询
- 结构化输出

### 三级存储（triple\_tier\_storage.py）

**功能**: "云端 → 缓存 → 本地"三级架构

**核心能力**:

- Supabase 云端持久化（可选）
- Redis 缓存加速（可选）
- 本地 JSON 文件备份
- 自动降级与同步机制

### 上下文压缩（context\_compressor.py）

**功能**: 智能管理对话上下文长度

**触发条件**:

- 消息数量超过阈值（默认 20 条）
- Token 数量超过阈值（默认 3000）

**压缩策略**:

- 保留开头与结尾关键消息
- 摘要提取与核心话题保留
- 可手动触发压缩

***

## 🧪 测试脚本

项目包含多个测试脚本，用于验证各模块功能：

```bash
# 测试向量搜索功能
python test_vector_search.py

# 测试智能检索策略
python test_smart_search.py

# 测试上下文压缩
python test_context_compression.py

# 测试 Agent 功能
python test_agent.py

# 测试段落分割
python test_paragraph_segmentation.py

# 测试图片解析
python test_image_parser.py

# 测试知识库搜索
python test_knowledge_search.py
```

