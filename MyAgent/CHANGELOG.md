# Changelog

## V3.2 — 2026-03-17 — Bug 修复 & 项目健壮性提升

### 修复

- **文生图 API 404 / 403**
  - 端点从不支持的 OpenAI 兼容模式 (`compatible-mode/v1/images/generations`) 改为 DashScope 原生端点 (`/api/v1/services/aigc/multimodal-generation/generation`)
  - 请求体改为 DashScope 原生 messages 格式
  - 模型切换为 `qwen-image-2.0`，默认分辨率 `1328*1328`
  - 生成的图片现在会下载到本地并在聊天中直接展示，附带本地文件路径供下载

- **图生文超时 (write timeout)**
  - 上传前自动将图片缩放至 1280px 内 + JPEG 压缩（quality=85），大幅减小请求体
  - httpx 超时参数从 `write=None` 改为显式 `write=300s, read=180s, connect=30s, pool=30s`
  - `agent/rag/loader.py` 的 `_describe_image()` 同步应用相同的压缩和超时修复

- **EnsembleRetriever 导入崩溃**
  - 修正导入路径为 `langchain.retrievers.ensemble.EnsembleRetriever`（适配 langchain 0.2.x + langchain-community 0.4.x）
  - 添加多级 fallback + `None` 守卫，导入失败时降级为纯 FAISS 检索

- **`--classic` 模式 ImportError**
  - `create_agent` / `before_model` 在当前 LangChain 中不存在，改为抛出明确的弃用提示

- **`code_agent.py` explain_code 空壳工具**
  - 原来只返回 prompt 字符串，现在实际调用 LLM 进行代码解释
  - 已添加到 `CodeAgent.get_tools()` 工具列表

### 改进

- **Chatbot 图片展示** — 使用 `gr.FileData` + `type="messages"` 确保图片在 Gradio 聊天中正确渲染
- **依赖完整性** — `requirements.txt` 新增 `Pillow>=10.0.0`（图片压缩依赖）

### 受影响文件

| 文件 | 变更 |
|------|------|
| `agent/tools.py` | 文生图端点/格式重写 + 图片下载；图生文压缩+超时 |
| `agent/config.py` | 默认文生图模型改为 `qwen-image-2.0` |
| `agent/rag/retriever.py` | EnsembleRetriever 导入修复 |
| `agent/rag/loader.py` | 图片压缩 + 超时修复 |
| `agent/multi/workers/code_agent.py` | explain_code 改为实际调用 LLM |
| `main.py` | --classic 弃用提示 |
| `webui.py` | 图片展示 + 下载链接 |
| `requirements.txt` | +Pillow |
| `.env` | TEXT_TO_IMAGE_MODEL=qwen-image-2.0 |

---

## V3.0 — 2026-03-17 — 全面升级

### 新功能
- **会话管理**: 新建/切换/重命名/删除历史对话（WebUI 下拉列表 + CLI `/list` `/switch` `/rename` `/delete` 命令）
- **RAG 开关**: WebUI 新增 Checkbox 启用/禁用 RAG；CLI 新增 `--no-rag` 参数
- **VL 视觉能力选配**: 图生文 / 文生图在 WebUI 侧栏可勾选启用，未配 API Key 时灰显提示
- **中间过程实时展示**: 单智能体使用 `astream_events` 流式推送工具调用过程到"执行过程"面板；多智能体 EventBus 增强
- **WebUI 全面重构**: 现代仪表盘风格、侧栏分组卡片、CSS 变量支持深色模式、移动端响应式
- **统一配置中心**: 所有参数集中 `agent/config.py`，配套 `.env.example` 模板

### 安全
- `execute_python` 沙箱改为 AST 解析检查，拦截 `__import__`/`importlib`/`eval`/`exec` 等危险调用
- `fetch_webpage` 新增 SSRF 防护：拒绝访问内网、localhost、link-local 地址
- `supervisor.py` JSON 解析抽取为 `_extract_plan_json()`，增加正则兜底 + worker 名校验

### 性能
- `profile.py` 增加 30 秒内存缓存 + 写入时主动失效
- `vectorstore.py` Embedding 改为模块级单例
- `tools.py` 文生图/图生文工具闭包缓存，避免每次 `get_all_tools()` 重复创建

### 配置变更
- `MAX_MESSAGES`: 10 → 20（上下文窗口扩大）
- `MAX_TOOL_ITERATIONS`: 8 → 5（工具调用轮数收紧）
- 新增配置项: `ENABLE_RAG`, `RAG_MODE`, `ENABLE_VL`, `VL_API_KEY`, `LOG_LEVEL`, `TEXT_TO_IMAGE_MODEL`
- 视觉能力默认策略调整:
  - 文生图模型由 `TEXT_TO_IMAGE_MODEL` 控制（默认 `qwen-image-plus-2026-01-09`）
  - WebUI 启动时图生文默认不启用，需在侧栏勾选并点击“应用设置”后才会加载 `describe_image`

### 代码质量
- 全项目 `print()` → `logging` 模块
- `graph.py` profile 注入逻辑抽取为 `_inject_system()`
- `supervisor.py` JSON 解析逻辑去重
- `profile.py` 删除未使用的 `import os`，异常不再静默吞掉

---

## Day 9 — 2026-03-17 — 搜索工具修复

### 问题

- **`web_search` 工具失效** — 用户查询实时数据（如股票、指数）时返回"网络错误无法找到"
- 根因：`ddgs` 包未安装，导致 `from ddgs import DDGS` 导入失败

### 修复

- 安装 `ddgs` 包（版本 9.11.4），这是 `duckduckgo-search` 的新包名
- `requirements.txt` 已正确声明 `ddgs>=9.0.0`
- `agent/tools.py` 导入语句 `from ddgs import DDGS` 保持不变

### 技术说明

- PyPI 上 `duckduckgo-search` 已重命名为 `ddgs`，两者等效
- 正确用法：
  ```bash
  pip install ddgs           # 安装
  ```
  ```python
  from ddgs import DDGS      # 导入
  with DDGS() as ddgs:
      results = list(ddgs.text('query', max_results=5))
  ```

---

## Day 8 — 2026-03-15 — 服务器部署优化 & 工具扩充 (V2.1)

### 新增

- **工具链扩充**（`agent/tools.py`）

| 工具 | 功能 | 费用 |
|------|------|------|
| `get_weather` | 实时天气查询（wttr.in 免费 API） | 免费 |
| `unit_convert` | 单位换算（长度/重量/温度） | 免费 |
| `fetch_webpage` | 抓取网页正文并返回纯文本 | 免费 |
| `format_json` | JSON 格式化 / 校验 | 免费 |
| `summarize_text` | 对长文本做简短摘要 | 免费 |
| `translate_text` | 文本翻译（调用主 LLM） | 少量 Token |
| `text_to_image` | 文生图（通义万相，可选开启） | 付费可选 |
| `describe_image` | 图生文（视觉模型，可选开启） | 付费可选 |

- **服务器部署备份** — `server_deploy_backup_20260315/`，涵盖所有服务代码、Nginx 配置及一键启停脚本
- **长期记忆模块** — `agent/memory/profile.py`，用户画像持久化至 `data/user_profile.json`

### 变更

- **Nginx HTTP 访问** — 移除 HTTPS 强制跳转，改为同时支持 IP 直连和域名访问，解决浏览器 HSTS 缓存导致的无法访问问题
- **WebUI 重构**（`webui.py`）— Tabler 风格现代化 UI，新增会话命名、Token 统计面板、执行过程 Terminal 展示
- **配置变更**（`agent/config.py`）— 主模型由 `deepseek-v3.1` 切换为 `qwen-plus`；新增付费能力开关 `ENABLE_TEXT_TO_IMAGE` / `ENABLE_IMAGE_TO_TEXT`

### 服务器更新

- Nginx 配置改为 HTTP 模式，新增 `/files/` 路由指向 File Agent
- 访问地址改为 `http://8.138.164.133/agent/`（移除 HTTPS）

---

## Day 7 — 2026-03-09 — Phase 7: 多智能体协作系统 (V2.0)

### 新增

- **多智能体模块** (`agent/multi/`)
  - `state.py` — `MultiAgentState` 多智能体共享状态
  - `supervisor.py` — Supervisor 路由决策 + Aggregator 结果汇总
  - `graph.py` — 多智能体 StateGraph，支持动态 Worker 调度
  - `workers/base.py` — Worker 基类，统一工具绑定和执行接口
  - `workers/code_agent.py` — 代码专家（execute_python, check_syntax）
  - `workers/data_agent.py` — 数据分析专家（analyze_numbers, describe_trend + RAG）
  - `workers/writer_agent.py` — 写作专家（generate_outline, word_count, format_as_markdown）

- **CLI 多智能体模式**
  - `main.py` 新增 `--multi` 参数启动多智能体模式
  - 与 `--classic` 互斥

- **WebUI 模式切换**
  - 左侧控制面板新增「智能体模式」选项
  - 支持在线切换单智能体/多智能体模式

- **测试脚本** (`test_multi_agent.py`)
  - 支持单 Agent 和多 Agent 协作场景测试

### 变更

- `README.md` 更新为 V2.0，添加多智能体系统文档
- `webui.py` 重构，支持双模式运行

---

## Day 6 — 2026-02-24 — Phase 6: 高级特性 (V1.0 完结)

### 新增

- **流式输出** — `astream_events` 实现 Token-by-Token 打字机效果
- **长期记忆** — 跨会话用户画像存储（`remember_user_fact` 工具）
- **Web UI** — Gradio 聊天界面，支持 RAG 模式切换、记忆管理

---

## Day 5 — 2026-02-21 — Phase 5: 深入 RAG 管线

### 新增

- **Markdown 语义分块** — `MarkdownHeaderTextSplitter` 带层级记忆
- **BM25 词频检索** — `rank_bm25` 精确匹配专有名词
- **混合检索** — `EnsembleRetriever` + RRF 融合算法
- **查询改写增强** — 为 Hybrid Search 优化关键词扩充

---

## Day 4 — 2026-02-11 — Phase 4: 手动构建 StateGraph

### 新增

- **`agent/graph.py`** — 手动构建 LangGraph StateGraph，替代 `create_agent` 黑盒
  - 自定义 `AgentState`（messages + iteration_count）
  - 6 个节点: trim、rewrite、agent、tools、increment、force_reply
  - 查询改写（rewrite_node）— 短问题/含代词时 LLM 先改写再检索
  - 工具调用上限（max_iterations=5）— 防止无限循环，超限强制回答
- `main.py` 双模式支持: `--classic` 使用旧版，默认使用新版 StateGraph

### 变更

- `main.py` 重构为 `_build_classic_agent()` 和 `_build_graph_agent()` 双分支
- 新增 `argparse` 命令行参数解析
- 启动时显示当前 Agent 模式名称

---

## Day 3 — 2026-02-09 — Phase 3: Memory + Token 优化

### 新增

- **Memory 子包** (`agent/memory/`)
  - `checkpointer.py` — SqliteSaver 持久化对话记忆
  - 数据库路径: `data/db/agent_memory.db`
  - 支持多 thread_id 管理，可切换会话线程
- **数据库检查脚本** (`scripts/inspect_db.py`)
  - 列出数据库中的表和行数
  - 可选查看最近 checkpoint 详情
  - 自动处理相对路径，避免转义问题
- **Token 优化**
  - `trim_message_history()` 状态修改器
  - 消息窗口限制（默认保留最近 10 条）
  - 长对话 token 消耗减少 50-80%
- `main.py` 集成 Memory 和 Token 优化
  - `/thread <id>` 命令切换会话
  - `clear` 生成新 thread_id
  - 启动时显示消息窗口限制配置

### 变更

- `main.py`:
  - 导入 `trim_messages`
  - 添加 `MAX_MESSAGES` 配置常量
  - `create_react_agent` 新增 `checkpointer` 和 `state_modifier` 参数
  - `invoke` 调用传入 `config` 字典指定 `thread_id`
- `requirements.txt` 新增 `langgraph-checkpoint-sqlite`
- `data/db/` 目录创建，存放所有数据库文件

### 修复

- `scripts/inspect_db.py` 路径转义问题（`\a` 被解释为响铃符）

---

## Day 2 — 2026-02-08 — Phase 2: RAG + 多格式支持

### 新增

- **RAG 子包** (`agent/rag/`)
  - `loader.py` — 文档加载器，支持 txt/md/pdf/docx/xlsx/pptx + 图片/音频/视频
  - `vectorstore.py` — FAISS 向量存储，支持分 batch 向量化和增量追加
  - `retriever.py` — 将检索器包装为 Agent 工具 `knowledge_search`
- **索引脚本** (`scripts/index_docs.py`)
  - 支持 `--append` 增量模式
  - 支持 `--batch-size`、`--chunk-size`、`--chunk-overlap` 参数
- **多格式文档加载**
  - Office: Word (.docx)、Excel (.xlsx)、PowerPoint (.pptx)
  - 多媒体: 图片→视觉模型描述、音频→语音转文字、视频→视觉模型理解
- **模型配置集中化** — 所有模型名称收拢到 `config.py`
- `main.py` 集成 RAG 工具，启动时自动检测向量索引
- `requirements.txt` 新增 `faiss-cpu`、`python-docx`、`openpyxl`、`python-pptx`

### 变更

- `config.py` 新增 `EMBEDDING_MODEL`、`VISION_MODEL`、`ASR_MODEL` 配置项
- `vectorstore.py` 的 `build_vectorstore()` 新增 `batch_size` 和 `append` 参数
- 系统提示词增加 `knowledge_search` 工具说明和来源引用规则

---

## Day 1 — 2026-02-07 — Phase 1: 基础 Agent

### 新增

- **项目基础设施**: `.gitignore`、`.env`、`requirements.txt`
- **Agent 核心包** (`agent/`)
  - `config.py` — 环境变量加载 (python-dotenv)
  - `llm.py` — ChatTongyi LLM 封装 (DeepSeek V3.1 via DashScope)
  - `tools.py` — 3 个工具: 日期时间、安全计算器 (AST)、DuckDuckGo 搜索
  - `callbacks.py` — Token 用量统计回调 (用户自行添加)
- **CLI 入口** (`main.py`) — `create_react_agent` + 交互式聊天循环
- 基础 LCEL chain 测试 (`test.py`)

### 变更

- `test.py` 移除硬编码 API key，改用 `agent.config`
