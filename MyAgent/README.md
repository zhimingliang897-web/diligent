# MyAgent (V3.2)

基于 LangChain + LangGraph 打造的个人专属 AI 助理系统。支持**多智能体协作**、**RAG 混合检索知识库**、**跨会话长期记忆**、**文生图/图生文视觉能力**与**实时工具过程展示**。

---

## 核心特性

### V3.2（当前版本）
- **文生图全面修复**：API 端点改为 DashScope 原生接口，模型默认 `qwen-image-2.0`，图片直接在聊天中展示并提供本地下载路径
- **图生文超时修复**：上传前自动压缩图片（缩放+JPEG），超时参数大幅提升
- **EnsembleRetriever 导入修复**：适配 langchain 0.2.x + langchain-community 0.4.x
- **依赖完整性**：`requirements.txt` 补充 `Pillow`
- **代码解释工具**：`explain_code` 从空壳变为实际调用 LLM
- **`--classic` 弃用**：给出明确提示，避免 ImportError

### V3.0
- **会话管理**：新建 / 切换 / 重命名 / 删除历史对话（WebUI 侧栏 + CLI 命令）
- **RAG 开关**：WebUI 和 CLI 均可开启/关闭 RAG，并选择检索模式
- **VL 视觉能力选配**：图生文 / 文生图在 UI 中可勾选启用，未配 API 时有提示
- **中间过程实时展示**：单智能体模式使用 `astream_events` 流式推送工具调用过程；多智能体 EventBus 增强
- **WebUI 全面重构**：现代仪表盘风格、响应式布局、深色模式 CSS 变量支持
- **安全加固**：AST 沙箱、`fetch_webpage` SSRF 防护、Supervisor JSON 健壮解析
- **性能优化**：Profile 读取 30s 缓存、Embedding 单例、工具闭包缓存

### V2.0 多智能体协作系统
- **Supervisor + Workers 架构**：智能任务分解与调度
- **Code Agent**：代码生成、调试、语法检查
- **Data Agent**：数据分析、统计计算、知识库检索
- **Writer Agent**：文章撰写、内容润色、格式化
- **Re-plan**：任务失败时自动重规划

### V1.0 基础能力
- **Web UI 界面**：基于 Gradio 构建的现代聊天交互窗口
- **长短期双重记忆**：SQLite 会话上下文 + 跨会话用户画像
- **混合检索 RAG**：FAISS 向量检索 + BM25 词频检索，RRF 融合打分
- **工具链网络**：网页搜索、天气查询、数学计算、单位换算等 13+ 个内置工具

---

## 内置工具

| 工具 | 描述 | 费用 |
|------|------|------|
| `get_current_datetime` | 获取当前日期、时间和星期 | 免费 |
| `calculate` | 安全数学表达式计算（AST 沙箱） | 免费 |
| `web_search` | DuckDuckGo 网络搜索 | 免费 |
| `get_weather` | 实时天气查询（wttr.in） | 免费 |
| `unit_convert` | 单位换算（长度 / 重量 / 温度） | 免费 |
| `fetch_webpage` | 抓取网页正文（含 SSRF 防护） | 免费 |
| `format_json` | JSON 格式化与校验 | 免费 |
| `summarize_text` | 长文本摘要 | 免费 |
| `translate_text` | 文本翻译（调用主 LLM） | 少量 Token |
| `remember_user_fact` | 跨会话保存用户偏好到长期记忆 | 免费 |
| `knowledge_search` | 本地知识库混合检索 | 免费 |
| `text_to_image` | 文生图（通义万相，UI 选配） | 付费可选 |
| `describe_image` | 图生文（VL 视觉模型，UI 选配） | 付费可选 |

---

## 快速开始

### 1. 环境准备

```bash
conda activate myagent    # Python 3.10+
pip install -r requirements.txt
```

### 2. 配置 API Key

```bash
cp .env.example .env
# 编辑 .env，填入 DASHSCOPE_API_KEY
```

完整配置项说明见 [`.env.example`](.env.example)。

### 3. 构建本地知识库（可选）

```bash
python scripts/index_docs.py data/
```

### 4. 启动应用

**Web 图形界面（推荐）**
```bash
python webui.py
# 浏览器打开 http://127.0.0.1:7860
```

**命令行界面**
```bash
python main.py                      # 单智能体模式（默认）
python main.py --multi              # 多智能体协作模式
python main.py --no-rag             # 关闭 RAG
python main.py --rag classic        # 使用纯 FAISS 检索
python main.py --no-stream          # 关闭流式输出

# CLI 会话命令:
#   /list          查看所有历史会话
#   /switch <id>   切换到指定会话
#   /rename <name> 重命名当前会话
#   /delete <id>   删除指定会话
```

**Windows 一键启动**
双击 `启动GUI.bat`，自动清理端口、激活环境、打开浏览器。

---

## 多智能体系统

### 架构图

```
用户输入
    │
    ▼
┌─────────────────┐
│   Supervisor    │  意图识别 + 任务分解
└────────┬────────┘
         │
    ┌────┼────┐
    ▼    ▼    ▼
 [Data] [Writer] [Code]    专业 Worker Agents
    │    │    │
    └────┼────┘
         │
    ┌────▼────┐
    │Aggregator│  结果汇总
    └─────────┘
         │
         ▼
     最终输出
```

### 工作流程

1. **Supervisor** 分析用户请求，创建有序任务计划
2. 按计划依次调度 Worker Agents 执行子任务
3. 每个 Worker 完成后，结果作为上下文传递给下一个 Worker
4. 若 Worker 执行失败，自动触发 Re-plan 重新规划
5. **Aggregator** 汇总所有结果，生成最终回答

---

## 配置参数

| 参数 | 默认值 | 说明 |
|------|--------|------|
| `MODEL_NAME` | qwen-plus | 主对话模型 |
| `TOOL_MODEL_NAME` | qwen-plus | 工具调用模型 |
| `EMBEDDING_MODEL` | text-embedding-v3 | RAG 向量嵌入模型 |
| `VISION_MODEL` | qwen3-vl-plus-2025-12-19 | 图生文 VL 模型 |
| `TEXT_TO_IMAGE_MODEL` | qwen-image-2.0 | 文生图模型 |
| `MAX_MESSAGES` | 20 | 消息窗口大小（保留最近 N 条） |
| `MAX_TOOL_ITERATIONS` | 5 | 工具调用最大轮数 |
| `ENABLE_RAG` | 1 | RAG 知识库总开关 |
| `RAG_MODE` | advanced | 检索模式：classic / advanced |
| `ENABLE_TEXT_TO_IMAGE` | 1 | 文生图开关 |
| `ENABLE_IMAGE_TO_TEXT` | 1 | 图生文开关 |
| `ENABLE_VL` | 1 | VL 视觉理解总开关 |
| `VL_API_KEY` | （复用主 Key） | VL 模型专用 API Key，留空则复用 `DASHSCOPE_API_KEY` |
| `TOOL_EVIDENCE_REQUIRED` | 1 | 回答末尾附加工具证据摘要 |
| `TOOL_EVIDENCE_MAX_CHARS` | 280 | 单条证据最大字符数 |
| `LOG_LEVEL` | INFO | 日志级别 |

---

## 视觉能力（文生图 / 图生文）使用说明

### 文生图

- **接口**：DashScope 原生端点 `POST /api/v1/services/aigc/multimodal-generation/generation`
- **模型**：`.env` 中 `TEXT_TO_IMAGE_MODEL`（默认 `qwen-image-2.0`）
- **size 格式**：`1328*1328`（DashScope 原生格式用星号）
- **图片展示**：生成后自动下载到本地临时目录，在聊天中直接显示并提供下载链接
- **开关**：WebUI 左侧栏勾选"文生图"后点击"应用设置"

### 图生文

- **接口**：DashScope OpenAI 兼容端点 `POST /compatible-mode/v1/chat/completions`（multimodal）
- **模型**：`.env` 中 `VISION_MODEL`（默认 `qwen3-vl-plus-2025-12-19`）
- **图片预处理**：上传前自动缩放至 1280px 内 + JPEG 压缩，避免超时
- **API Key**：使用 `VL_API_KEY`，留空则自动复用 `DASHSCOPE_API_KEY`
- **开关**：WebUI 左侧栏勾选"图生文"后点击"应用设置"

### 多智能体模式限制

- **图生文**：可用（WebUI 先调 `describe_image` 生成文字描述，再拼入用户消息）
- **文生图**：当前多智能体 Workers 未绑定 `text_to_image` 工具，多智能体模式下不会自动调用

---

## 图片上传与发送机制（WebUI）

- **图片临时路径**：Gradio 落盘到 `C:\Users\<用户名>\AppData\Local\Temp\gradio\...`，WebUI 回调拿到的是该临时路径
- **发送流程**：
  1. 收集临时文件路径到 `pending_images` State
  2. 若启用图生文：逐张读取 → base64 编码 → `httpx.post` 到 DashScope VL 接口
  3. 把"图片文字描述"拼到用户消息里，交给单 / 多智能体处理

---

## 常见问题与排障

**文生图报 404 / 403 错误**
- V3.2 已改为 DashScope 原生端点，确认 `TEXT_TO_IMAGE_MODEL` 拼写正确
- 403 `AllocationQuota.FreeTierOnly`：在阿里云控制台关闭"仅使用免费额度"，或切换到有额度的模型
- 确认已在阿里云控制台开通该模型权限

**图生文超时 `The write operation timed out`**
- V3.2 已加入自动图片压缩（缩放至 1280px 内 + JPEG quality=85），大幅降低上传体积
- 若仍超时，检查网络连接或代理设置

**图生文 401/403 权限错误**
- 检查阿里云控制台是否开通视觉模型权限
- 检查 `.env` 中 `VL_API_KEY` 或 `DASHSCOPE_API_KEY` 是否正确

**文生图图片未在聊天中显示**
- V3.2 使用 `gr.FileData` 直接在 Chatbot 中渲染图片，并附带下载链接
- 若仍未显示，检查 `webui.py` 中 `gr.Chatbot` 是否设置了 `type="messages"`

**`EnsembleRetriever` 导入失败**
- V3.2 已修正导入路径为 `langchain.retrievers.ensemble`，并添加多级 fallback
- 若仍报错，执行 `pip install -r requirements.txt` 确保 langchain 版本兼容

**`ModuleNotFoundError: No module named 'langchain_core'`**
- 执行 `pip install -r requirements.txt` 安装所有依赖

---

## 目录结构

```text
MyAgent/
├── .env.example               # 环境变量配置模板
├── .env                       # 本地环境变量（不提交 git）
├── webui.py                   # Gradio Web 界面入口
├── main.py                    # CLI 命令行入口
├── requirements.txt           # Python 依赖
│
├── agent/                     # 核心智能体模块
│   ├── config.py              # 统一配置中心
│   ├── graph.py               # 单智能体 StateGraph
│   ├── tools.py               # 工具库（含文生图 / 图生文）
│   ├── llm.py                 # LLM 封装（ChatTongyi）
│   ├── callbacks.py           # Token 统计回调
│   │
│   ├── multi/                 # 多智能体模块
│   │   ├── state.py           # MultiAgentState
│   │   ├── supervisor.py      # Supervisor 路由
│   │   ├── graph.py           # 多智能体 StateGraph
│   │   ├── event_bus.py       # 执行过程事件总线
│   │   └── workers/
│   │       ├── base.py        # Worker 基类
│   │       ├── code_agent.py  # 代码专家（AST 沙箱）
│   │       ├── data_agent.py  # 数据分析专家
│   │       └── writer_agent.py# 写作专家
│   │
│   ├── memory/
│   │   ├── checkpointer.py    # 会话记忆 + 会话管理 API
│   │   └── profile.py         # 长期用户画像（带缓存）
│   │
│   └── rag/
│       ├── loader.py          # 文档加载器
│       ├── vectorstore.py     # 向量存储（Embedding 单例）
│       └── retriever.py       # 混合检索器
│
├── scripts/
│   └── index_docs.py          # 建库脚本
├── data/                      # 知识库源文件
└── vectorstore/               # 向量索引（自动生成）
```

---

## 版本历史

| 版本 | 日期 | 主要更新 |
|------|------|---------|
| V3.1 | 2026-03-17 | 移除登录体系（本地直接打开）；文生图改用 OpenAI 兼容端点修复 url error；API Key 校验延迟到运行时 |
| V3.0 | 2026-03-17 | 会话管理、RAG 开关、VL 选配、中间过程实时展示、WebUI 重构、安全加固、性能优化 |
| V2.1 | 2026-03-17 | 修复 `web_search`（改用 `ddgs` 包）；工具链扩充至 13 个 |
| V2.0 | 2026-03-09 | 多智能体协作系统（Supervisor + Workers）；Tabler 风格 WebUI |
| V1.0 | 2026-02-07 | 完整单智能体系统（RAG + Memory + WebUI） |

---

## 许可证

本项目为个人 AI 学习渐进式练手工程，自由参考与取用！
