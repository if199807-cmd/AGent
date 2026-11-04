# langchain_Agent — SCI Digital Pro 智能客服与故障诊断 Agent

> 基于 LangChain ReAct Agent 框架，为 **SCI Digital Pro 全自动数字 PCR 仪** 提供智能客服问答、错误码自动诊断以及多模态交互能力的 AI 助手。

---

## 目录

- [项目背景](#项目背景)
- [功能特性](#功能特性)
- [技术栈](#技术栈)
- [快速开始](#快速开始)
- [项目结构](#项目结构)
- [配置说明](#配置说明)
- [使用方式](#使用方式)
- [Streamlit Web 界面](#streamlit-web-界面)
- [工作流程详解](#工作流程详解)
- [RAG 知识库管理](#rag-知识库管理)
- [注意事项与代码审查发现](#注意事项与代码审查发现)
- [常见问题](#常见问题)
- [许可证](#许可证)

---

## 项目背景

**SCI Digital Pro** 是一款全自动芯片式数字 PCR（dPCR）一体机，包含进样、扩增、阅读三大子系统，配备 9 个电机模组、多路串口通信、复杂的气路与液路控制体系。

当设备在现场出现错误码（如 `CE0502`、`CE0020`）时，一线运维人员通常需要翻阅数百页的文档来定位原因和解决方案，效率低下。

本项目旨在利用大语言模型的 ReAct 推理能力，配合 RAG 知识库检索、动态提示词切换与跨轮次记忆技术，构建一个 **即问即答、自动诊断** 的智能客服 Agent，实现：

- 快速回答设备的日常操作与使用问题
- 自动分析错误码根因，并生成结构化诊断报告
- 支持图片多模态输入（设备照片、屏幕截图等）
- 跨轮次对话记忆（SqliteSaver Checkpointer）
- 提供 Streamlit 可视化 Web 界面

---

## 功能特性

### 🧠 双模式智能切换

| 模式 | 触发条件 | 行为 |
|------|---------|------|
| **普通问答模式** | 日常使用/操作咨询 | 调用 RAG 检索专业知识，以自然语言回答 |
| **错误诊断模式** | 用户询问错误码原因/解决方法 | Agent 自动调用 `fill_context_for_report` → 中间件捕获后将 `report` 标志置为 True → 动态切换 System Prompt → 执行完整诊断工作流（RAG 检索 + 获取错误信息 + 获取日志）→ 生成 Markdown 报告 |

切换通过 **中间件 + Runtime Context 标记** 动态完成，无需用户手动指定。

### 🔍 混合检索 RAG

- **BM25 关键词检索**：对精确错误码（如 `CE0502`）匹配更强
- **向量语义检索**：理解用户意图和上下文
- 结果去重合并，兼顾 **精度与召回**

### ⚙️ 可插拔中间件架构

利用 LangChain Agent 中间件机制（`@wrap_tool_call`、`@before_model`、`@dynamic_prompt`）实现：

- 工具调用监控与日志
- 模型调用前状态记录
- **运行时动态切换提示词**（普通对话 ↔ 报告生成）

### 💬 多模态交互

对话模型使用 `qwen3.5-omni-plus-2026-03-15`，支持文本与图片混合输入。用户可上传设备照片或屏幕截图，Agent 可结合视觉信息分析问题。

### 💾 跨轮次记忆

集成 `SqliteSaver` Checkpointer，通过 `thread_id` 区分不同会话，实现跨轮次对话历史恢复。

### 📄 多格式知识库加载

支持 PDF、TXT、CSV 三种格式的知识文档，自动分块、去重（MD5 校验）、向量化存入 ChromaDB。

### 🌐 联网搜索能力

集成 Tavily Search API，在本地知识不足时可实时联网获取信息。

### 🖥️ Streamlit 可视化界面

内置 `app.py` 基于 Streamlit 构建，提供浏览器端的对话界面，无需手写交互代码即可使用。

---

## 技术栈

| 类别 | 技术 | 用途 |
|------|------|------|
| **语言** | Python 3.10+ | 运行时 |
| **AI 框架** | LangChain / LangGraph / LangChain Community | Agent 创建、工具定义、提示词管道、运行时管理 |
| **对话模型** | Qwen 3.5-Omni-Plus（阿里云百炼平台，OpenAI 兼容接口） | 对话与推理（支持多模态） |
| **总结模型** | Qwen 3.7-Plus | RAG 总结与报告生成 |
| **嵌入模型** | Qwen 3.7-Text-Embedding | 文本向量化 |
| **向量数据库** | ChromaDB | 知识库向量存储 |
| **混合检索** | BM25Retriever + Chroma 语义检索 | 精确 + 语义双重检索 |
| **知识库文件** | PyPDFLoader / TextLoader / CSVLoader | 多格式文档加载 |
| **联网搜索** | Tavily Search API | 实时信息获取 |
| **对话记忆** | SqliteSaver（SQLite） | 跨轮次 Checkpointer |
| **Web 界面** | Streamlit | 浏览器可视化交互 |
| **配置管理** | PyYAML | YAML 配置文件解析 |
| **日志系统** | Python logging | 控制台 + 按日滚动文件日志 |

---

## 快速开始

### 前置要求

- Python 3.10+
- 安装依赖

```bash
pip install langchain langchain-core langchain-community langchain-chroma \
            langchain-text-splitters langgraph langchain-tavily \
            dashscope pyyaml pypdf chromadb streamlit
```

> **注意**：部分中间件 API（如 `langchain.agents.middleware`、`langgraph.runtime`）属于平台自定义扩展，不在 LangChain 官方稳定版中。建议锁定依赖版本或咨询框架维护者。

### 1. 进入项目目录

```bash
cd langchain_Agent
```

### 2. 配置 API 密钥

编辑 `config/rag.yml`，填入你的模型访问凭证：

```yaml
chat_model_name: qwen3.5-omni-plus-2026-03-15   # 对话模型（支持多模态）
summarize_model_name: qwen3.7-plus               # 总结模型
base_url: https://<你的服务地址>/compatible-mode/v1
model_key: sk-xxxxxxxxxxxxxxxxxxxxxxxxxxxxx
embedding_model_name: qwen3.7-text-embedding      # 嵌入模型
dashscope_api_key: sk-xxxxxxxxxxxxxxxxxxxxxxxxxxxxx
```

编辑 `config/agent.yml`，填入 Tavily 搜索 API Key 与 Checkpointer 路径：

```yaml
tavily_api_key: tvly-xxxxxxxxxxxxxxxxxxxxxxxxxxxxx
checkpointer_path: checkPoint/CheckPoint.db
```

### 3. 准备知识库数据

将你的知识文档放入 `data/` 目录（支持 `.txt`、`.pdf`、`.csv`），然后运行向量库构建脚本：

```bash
python rag/vector_store.py
```

导入成功后 `md5.text` 会记录已导入文件的指纹，后续重复执行会跳过已导入文件。

> `data/`、`chroma_db/`、`checkPoint/`、`logs/` 均在 `.gitignore` 中，不会提交到版本控制。

### 4. 启动 Agent

#### 命令行模式

```bash
python agent/react_agent.py
```

#### Streamlit Web 界面

```bash
streamlit run app.py
```

---

## 项目结构

```
langchain_Agent/
│
├── app.py                              # Streamlit Web 界面入口
│
├── agent/                              # Agent 核心模块
│   ├── react_agent.py                  # ReAct Agent 主入口，组装模型、工具、中间件
│   └── tools/
│       ├── agent_tools.py              # 工具定义：rag_summarize、web_search、fill_context_for_report 等
│       └── middleware.py               # 中间件：工具监控、模型前日志、动态提示词切换
│
├── config/                             # YAML 配置文件
│   ├── agent.yml                       # Tavily API Key + Checkpointer 路径
│   ├── chroma.yml                      # 向量库参数（按文件类型分块配置、检索数量）
│   ├── prompts.yml                     # 提示词模板文件路径注册
│   └── rag.yml                         # 模型配置（模型名、地址、API Key、嵌入模型）
│
├── rag/                                # 检索增强生成引擎
│   ├── rag_service.py                  # RAG 总结服务：检索 + 组装上下文 + 调用 LLM 生成回答
│   └── vector_store.py                 # 向量存储服务：ChromaDB 管理 + BM25 混合检索 + 文档导入
│
├── model/                              # 模型工厂
│   └── factory.py                      # 抽象工厂模式生产 ChatModel、EmbeddingModel、SummarizeModel
│
├── utils/                              # 工具层
│   ├── config_handler.py               # YAML 配置加载（一次性加载为全局变量）
│   ├── file_handler.py                 # 文件加载器（PDF/TXT/CSV）+ MD5 校验 + 编码探测
│   ├── logger_handler.py               # 日志系统（控制台 + 按日文件滚动）
│   ├── path_tool.py                    # 路径工具（获取项目根目录、相对→绝对路径）
│   └── prompt_loader.py                # 提示词文本加载器（根据配置路径读取 .txt 文件）
│
├── prompts/                            # 提示词模板（纯文本）
│   ├── main_prompt.txt                 # 主系统提示词：定义 ReAct 思考规则 + 全部工具说明
│   ├── rag_summarize_prompt.txt        # RAG 总结提示词：约束"仅依据参考资料回答"
│   └── report_prompt.txt               # 错误诊断报告提示词：定义报告生成规范
│
├── data/                               # ⚠️ 知识库数据（gitignored，需自行添加）
│   ├── error_codes.csv                 # 完整错误码表（CE/LE/PE/RE/ME 系列）
│   ├── deviceInfo.txt                  # 设备专业知识文档
│   ├── checklog.txt                    # 设备日志格式与解析规范说明
│   └── manual.pdf                      # 设备完整用户手册（PDF）
│
├── checkPoint/                         # ⚠️ Checkpointer 持久化（gitignored，运行生成）
│   └── CheckPoint.db                   # SQLite 对话状态数据库
│
├── chroma_db/                          # ⚠️ ChromaDB 向量数据库（gitignored，运行生成）
├── logs/                               # ⚠️ 运行时日志（gitignored，自动生成）
├── md5.text                            # ⚠️ MD5 指纹缓存（gitignored，运行生成）
│
├── 资料/                               # 项目辅助资料
│   └── 项目结构.png                    # 项目架构图
│
├── .gitignore                          # Git 忽略规则
└── README.md                           # 本文件
```

> ⚠️ 标记的目录/文件由 `.gitignore` 排除，不会进入版本控制。

---

## 配置说明

### `config/rag.yml` — 模型配置

```yaml
chat_model_name: qwen3.5-omni-plus-2026-03-15   # 对话模型（支持多模态）
summarize_model_name: qwen3.7-plus               # RAG 总结 & 报告生成模型
base_url: https://.../v1                         # OpenAI 兼容 API 地址
model_key: sk-...                                 # API Key
embedding_model_name: qwen3.7-text-embedding      # 嵌入模型名称
dashscope_api_key: sk-...                         # DashScope API Key
```

### `config/agent.yml` — Agent 配置

```yaml
tavily_api_key: tvly-...                  # Tavily 联网搜索 API Key
checkpointer_path: checkPoint/CheckPoint.db  # Checkpointer SQLite 数据库路径
```

### `config/chroma.yml` — 向量库配置（按文件类型分段）

```yaml
collection_name: "agent"                # ChromaDB 集合名称
persist_directory: "chroma_db"          # 持久化目录
k: 10                                   # 每次检索返回的文档数
data_path: data                         # 知识库原始文件目录
md5_hex_store: md5.text                 # MD5 指纹缓存文件
allow_knowledge_file_type: ["txt", "pdf", "csv"]

# ---- 切片配置（按文件类型分别设置） ----
chunk_config:
  txt:
    chunk_size: 400
    chunk_overlap: 50
    separators:
      - "\n\n## "
      - "\n\n### "
      - "\n\n"
      - "\n"
      - "。"
      - "！"
      - "？"
  pdf:
    chunk_size: 1000
    chunk_overlap: 100
    separators:
      - "\n\n"
      - "\n"
      - "。"
      - "！"
      - "？"
  csv:
    strategy: "no_split"                # CSV 每行已是独立信息，不拆分

# 混合检索权重 [BM25权重, 向量权重]（当前仅供配置参考，代码中未使用）
retriever_weights: [0.6, 0.4]
```

### `config/prompts.yml` — 提示词路径

```yaml
main_prompt_path: prompts/main_prompt.txt
rag_summarize_prompt_path: prompts/rag_summarize_prompt.txt
report_prompt_path: prompts/report_prompt.txt
```

---

## 使用方式

### 方式一：纯文本问答

```python
from agent.react_agent import ReactAgent

agent = ReactAgent()

# 普通问答（传入字符串，自动包装为消息）
for chunk in agent.execute_str_stream("SCI Digital Pro 怎么更换枪头？", thread_id="001"):
    print(chunk, end="", flush=True)

# 错误码诊断——Agent 会自动切换到诊断模式
for chunk in agent.execute_str_stream("设备报错 CE0502，怎么办？", thread_id="002"):
    print(chunk, end="", flush=True)
```

### 方式二：传入消息列表（支持多模态）

```python
from agent.react_agent import ReactAgent
from langchain_core.messages import HumanMessage

agent = ReactAgent()

# 纯文本消息列表
messages = [HumanMessage("我的设备报错 CE0502，帮忙分析一下")]
for chunk in agent.execute_messages_stream(messages, thread_id="003"):
    print(chunk, end="", flush=True)

# 多模态：上传设备照片让 Agent 分析
messages = [
    HumanMessage([
        {"type": "text", "text": "我的设备报错，帮我看看屏幕显示什么"},
        {"type": "image_url", "image_url": {"url": "https://example.com/device-screen.jpg"}},
    ])
]
for chunk in agent.execute_messages_stream(messages, thread_id="004"):
    print(chunk, end="", flush=True)

# 多模态：本地图片（base64）
import base64

def encode_image(path: str) -> str:
    with open(path, "rb") as f:
        return base64.b64encode(f.read()).decode("utf-8")

image_base64 = encode_image("your_photo.jpg")
messages = [
    HumanMessage([
        {"type": "text", "text": "描述这张图片的内容"},
        {"type": "image_url", "image_url": {"url": f"data:image/png;base64,{image_base64}"}},
    ])
]
for chunk in agent.execute_messages_stream(messages, thread_id="005"):
    print(chunk, end="", flush=True)
```

### 方式三：Streamlit Web 界面

```bash
streamlit run app.py
```

在浏览器中打开显示的地址（默认 `http://localhost:8501`），直接输入问题即可交互。

### 知识库维护

当 `data/` 目录下的知识文档更新后，重新运行向量库更新脚本即可：

```bash
python rag/vector_store.py
```

系统会自动计算文件 MD5，仅导入新增或修改过的文件，已有的不会重复导入。

---

## Streamlit Web 界面

`app.py` 提供了一个轻量级的 Web 交互界面：

```python
# 核心逻辑
st.title("小海龟智能客服")

agent = ReactAgent()  # 会话级别共享
# 用户输入 → execute_str_stream → 流式逐字输出
```

- 支持多轮对话，自动附加历史到消息列表
- 流式输出模拟逐字打印效果
- `thread_id` 固定为 `"001"`，同一浏览器会话内保持上下文连续

---

## 工作流程详解

### 普通问答流程

```
用户提问
    │
    ▼
ReactAgent 接收 query
    │
    ├── 主提示词 → 模型思考：需要专业知识吗？
    │      │
    │      ├── 是 → 调用 rag_summarize(query)
    │      │         → 混合检索 ChromaDB（BM25 + 向量）
    │      │         → LLM 总结 → 返回
    │      │
    │      └── 否 → 直接模型自身知识回答
    │
    ▼
输出自然语言回答
```

### 错误诊断流程

```
用户：设备报错 CE0502，怎么办？
    │
    ├── ① 主提示词引导 Agent 识别为"错误诊断"场景
    ├── ② 调用 fill_context_for_report()
    │       └── 中间件 monitor_tool 捕获该工具调用
    │           → runtime.context["report"] = True
    │
    ├── ③ 下一次模型调用前 →
    │       report_prompt_switch 检测到 report=True
    │       → System Prompt 动态切换为「报告生成模式」
    │       → 角色从"通用客服"变为"设备错误诊断专家"
    │
    ├── ④ 调用 rag_summarize("CE0502") → 检索知识库中的错误码说明
    ├── ⑤ 调用 fetch_external_data()   → 获取设备的实时错误信息（当前为模拟数据）
    ├── ⑥ 调用 get_logs_info()         → 获取设备运行日志（当前为模拟数据）
    │
    └── ⑦ 综合所有信息 → 生成 Markdown 格式的错误诊断报告
```

### 中间件架构

```
用户输入
    │
    ▼
┌─────────────────────────────────────┐
│  @before_model (log_before_model)   │ ← 模型调用前，记录消息数与内容摘要到日志
│  每次调模型前执行                      │
└─────────────────────────────────────┘
    │
    ▼
┌─────────────────────────────────────┐
│  @dynamic_prompt                    │ ← 检查 runtime.context["report"] 标记
│  (report_prompt_switch)             │    True  → 切换到报告提示词
│  动态选择 System Prompt              │    False → 使用主提示词
└─────────────────────────────────────┘
    │
    ▼
┌─────────────────────────────────────┐
│  模型推理 → 决定是否调用工具           │
└─────────────────────────────────────┘
    │
    ▼
┌─────────────────────────────────────┐
│  @wrap_tool_call (monitor_tool)     │ ← 记录工具调用日志
│  工具执行前后拦截                      │    若调用 fill_context_for_report
│                                      │    则将 report 标记置为 True
└─────────────────────────────────────┘
    │
    ▼
┌─────────────────────────────────────┐
│  SummarizationMiddleware            │ ← 消息数量超过阈值（6条）时，
│  (LangGraph 内置)                    │    自动摘要历史，保留最近 3 条
└─────────────────────────────────────┘
```

---

## RAG 知识库管理

### 数据文件说明

| 文件 | 类型 | 内容 |
|------|------|------|
| `error_codes.csv` | CSV | 设备全量错误码表，包含错误码、错误名称、可能原因、解决方案。自动提取 `error_code` 和 `error_prefix` 作为 metadata |
| `deviceInfo.txt` | TXT | 设备专业知识：通讯 COM 口、电机模组、运行流程、错误码分类、日志格式 |
| `checklog.txt` | TXT | 日志格式规范的详细说明，包含指令生命周期、错误码传播链、正则解析指导 |
| `manual.pdf` | PDF | 设备完整用户手册（约 8.9 MB） |

### 分块策略

| 文件类型 | 分块大小 | 重叠 | 分隔符策略 | 说明 |
|---------|---------|------|-----------|------|
| `.txt` | 400 字符 | 50 | 优先按标题（`##`/`###`）分割 | 结构化 Markdown 文档保持章节完整 |
| `.pdf` | 1000 字符 | 100 | 按段落/标点分割 | PDF 段落较长，分块更大 |
| `.csv` | 不分割 | — | — | 每行已是独立完整信息，附带 `error_code` metadata |

### 去重机制

每次导入文件时计算完整文件内容的 MD5（**注意：是读完整文件的 MD5，不是仅读前 4KB**），与 `md5.text` 中已记录的指纹比对。仅当 MD5 未命中时才执行向量化导入，避免重复构建知识库。

### Embedding 分批提交

DashScope Embedding API 限制每批不超过 20 条。`vector_store.py` 中的 `load_document()` 按 20 条一批分批提交，确保大文件导入不报错。

### 检索方式

| 方式 | 方法 | 说明 |
|------|------|------|
| **纯语义检索** | `get_retriever()` | 基于 ChromaDB 向量相似度 |
| **带过滤检索** | `get_retriever_with_filter(filter_dict)` | 按 metadata 过滤，如 `{"source_file": "error_codes.csv"}`、`{"error_prefix": "CE"}` |
| **混合检索** | `get_hybrid_retriever()` | BM25 关键词 + 向量语义，去重合并（BM25 结果优先） |

---

## 注意事项与代码审查发现

### 已修复的问题（较旧版本已解决）

以下问题在代码中已修复，保留供参考：

~~1. MD5 计算只读了文件首 4KB~~ ✅ **已修复**：`get_file_md5_hex()` 的 `return md5_hex` 已在 `while` 循环**外部**，读取完整文件后返回。

~~2. `execute_str_stream` 会输出非 AI 消息~~ ✅ **已修复**：已添加 `isinstance(latest_message, AIMessage)` 过滤。

### 当前仍存在的注意事项

#### 1. 非标准 LangChain API 依赖（`middleware.py`）

部分 API（如 `langchain.agents.middleware`、`langchain.agents.create_agent`、`langgraph.runtime` 中的 `@wrap_tool_call`、`@before_model`、`@dynamic_prompt`、`Runtime`、`StateSnapshot` 等）不在 LangChain 官方稳定版中，属于平台自定义扩展分支。升级 LangChain 版本可能导致导入失败。

**建议**：锁定依赖版本，在 `requirements.txt` 中注明分支来源或固定版本号。

#### 2. API Key 明文存储在配置文件中

`config/rag.yml` 和 `config/agent.yml` 中包含明文 API Key（模型密钥、Tavily 密钥、DashScope 密钥）。

**建议**：改用环境变量方式加载（如 `os.getenv("MODEL_API_KEY")`），YAML 中仅保留占位符。

#### 3. `print_prompt` 调试代码在生产中会输出完整提示词（`rag/rag_service.py`）

`rag_service.py` 第 16 行的 `print_prompt` 函数作为 LCEL 链的一环，每次 RAG 查询都会把完整提示词（含检索到的上下文）打印到 stdout。生产环境建议移除或加环境变量条件开关。

#### 4. `fetch_external_data` 和 `get_logs_info` 当前为模拟数据

这两个工具目前返回的是硬编码的 Mock 数据，未对接真实设备接口。对接真实数据源前，诊断报告中的"实时错误信息"和"日志分析"部分仅供参考。

#### 5. 混合检索权重未实际使用

`config/chroma.yml` 中的 `retriever_weights: [0.6, 0.4]` 已在配置中定义，但 `vector_store.py` 的 `hybrid_invoke` 函数中并未应用该权重，仅进行了去重合并。

#### 6. 缺少 `requirements.txt`

当前项目没有 `requirements.txt` 文件，依赖清单如下（供创建时参考）：

```
langchain>=0.3
langchain-core>=0.3
langchain-community>=0.3
langchain-chroma>=0.2
langchain-text-splitters>=0.3
langgraph>=0.2
langchain-tavily>=0.2
dashscope>=1.20
pyyaml>=6.0
pypdf>=5.0
chromadb>=0.5
streamlit>=1.38
```

> 注意：中间件相关模块（`langchain.agents.middleware` 等）不在上述标准包中，需额外安装或从平台获取。

---

## 常见问题

### Q: 模型 API 必须用阿里云通义千问吗？
不一定。`config/rag.yml` 中的 `base_url` 和 `model_key` 支持任何 OpenAI 兼容接口的服务商，只需修改模型名、地址和密钥即可。

### Q: 知识库更新后需要重启服务吗？
需要。知识库在 `VectorStoreService` 初始化时加载，更新后需重新运行 `vector_store.py` 并重启 Agent。

### Q: 新增错误码文件到 data 目录后如何处理？
将新文件放入 `data/` 目录，重新运行 `python rag/vector_store.py`，系统会自动计算 MD5 并导入新文件。

### Q: 如何扩展新的工具？
在 `agent/tools/agent_tools.py` 中用 `@tool` 装饰器定义新函数，然后在 `agent/react_agent.py` 的 `tools` 列表中加入该函数即可。如有特殊上下文注入需求，同时在 `middleware.py` 中增加处理逻辑。

### Q: 多模态传入为什么报错 400？
请检查 HumanMessage content 列表的格式：每个元素必须是 `{"type": "text", "text": "..."}` 或 `{"type": "image_url", "image_url": {"url": "..."}}` 格式。不要在外面再包一层 `{"role": "user", "content": [...]}` 格式。

### Q: `thread_id` 有什么用？
`thread_id` 用于 Checkpointer 区分不同会话。相同 `thread_id` 的对话历史会被恢复，实现跨轮次记忆；不同 `thread_id` 的对话相互隔离。不传默认为 `"default"`。

### Q: 两个 `execute_*` 方法有什么区别？
- `execute_str_stream(query, thread_id, report)`：传入字符串，内部自动包装为消息列表，适合纯文本快速问答。
- `execute_messages_stream(messages, thread_id, report)`：传入消息列表（`list[BaseMessage]` 或 `list[dict]`），支持多模态输入，更灵活。

### Q: Streamlit 界面如何修改标题？
修改 `app.py` 中的 `st.title("小海龟智能客服")` 为你想要的标题即可。

### Q: 为什么我的 data/ 目录是空的？
`data/` 目录在 `.gitignore` 中，不会被提交到 Git。你需要手动将知识文档放入该目录。


############################如何使用LangSmith：
添加.env文件，配置API和Project名称和网址地址
在Agent调用处导入环境变量
from dotenv import load_dotenv
load_dotenv()

---

## 许可证

本项目仅供学习和内部使用。SCI Digital Pro 相关设备文档和技术资料归原设备厂商所有。

---

> 项目基于 LangChain 框架构建，感谢 LangChain 社区的开源贡献。
