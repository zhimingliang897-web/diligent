# MyAgent 更新日志（整理版）

按时间倒序，便于查阅近期变更。原有 `CHANGELOG.md` 保留不变。

---

## 2026-03 — Phase 8: 工具扩展与 WebUI 优化

### 新增

- **工具扩展**（`agent/tools.py`）
  - 常驻工具：`unit_convert`（单位换算）、`fetch_webpage`（网页抓取）、`format_json`（JSON 格式化）、`summarize_text`（长文本摘要）、`translate_text`（翻译）
  - 可选付费工具（需在 `.env` 显式开启）：`text_to_image`（文生图，通义万相）、`describe_image`（图生文，视觉模型）
  - `get_all_tools()`：按配置返回基础 + 扩展 + 可选工具，Web 与 CLI 统一使用

- **配置项**（`agent/config.py`）
  - `ENABLE_TEXT_TO_IMAGE`、`ENABLE_IMAGE_TO_TEXT`（默认关闭），避免误扣费

- **对话管理可编辑**（`webui.py`）
  - 会话名称可编辑并保存为 thread_id，同一名称延续历史
  - 「保存名称」按钮 + 规范化（长度/特殊字符），留空则使用随机 ID
  - 「开启新对话」清空名称输入框与对话
  - 保存后 `gr.Info` 提示；发送中禁用输入框与发送按钮，回复结束后恢复

- **启动脚本**
  - `启动GUI.bat`：释放 7860 端口、启动 `python webui.py`、延迟打开浏览器

- **文档**
  - README 增加「可选能力（可能产生费用）」说明与 `.env` 示例
  - 目录结构补充 `启动GUI.bat`

### 变更

- `main.py`、`webui.py` 工具列表改为使用 `get_all_tools()`，SYSTEM_PROMPT 补充新工具说明
- 左侧「工具箱」展示当前实际加载的工具名列表

### 修复

- `agent/rag/vectorstore.py`：KeyboardInterrupt 分支中补充 `import sys`，避免 Ctrl+C 时 `NameError`

---

## 2026-03-09 — Phase 7: 多智能体协作系统 (V2.0)

### 新增

- **多智能体模块** (`agent/multi/`)
  - `state.py` — MultiAgentState、TaskItem、create_initial_state
  - `supervisor.py` — Supervisor 路由 + Aggregator 汇总，支持 Re-plan
  - `graph.py` — 多智能体 StateGraph，动态 Worker 调度
  - `workers/base.py` — Worker 基类
  - `workers/code_agent.py` — 代码专家（execute_python, check_syntax）
  - `workers/data_agent.py` — 数据分析专家（analyze_numbers, describe_trend + RAG）
  - `workers/writer_agent.py` — 写作专家（generate_outline, word_count, format_as_markdown）

- **CLI**：`main.py --multi` 启动多智能体模式（与 `--classic` 互斥）
- **WebUI**：左侧「智能体模式」可切换单/多智能体
- **测试**：`test_multi_agent.py` 单 Agent 与多 Agent 协作场景

### 变更

- README 更新为 V2.0；`webui.py` 支持双模式运行

---

## 2026-02-24 — Phase 6: 高级特性 (V1.0 完结)

### 新增

- 流式输出（`astream_events` Token 级）
- 长期记忆（`remember_user_fact` 工具）
- Web UI（Gradio），RAG 模式切换、记忆管理

---

## 2026-02-21 — Phase 5: 深入 RAG 管线

### 新增

- Markdown 语义分块（`MarkdownHeaderTextSplitter`）
- BM25 词频检索 + 混合检索（EnsembleRetriever、RRF）
- 查询改写增强（短问/代词时先改写再检索）

---

## 2026-02-11 — Phase 4: 手动构建 StateGraph

### 新增

- `agent/graph.py`：手动 StateGraph（trim → rewrite → agent → tools → increment / force_reply）
- 查询改写节点、工具调用上限（max_iterations=5）
- `main.py` 双模式：`--classic` 与默认 StateGraph

### 变更

- `main.py` 拆分为 `_build_classic_agent()` 与 `_build_graph_agent()`

---

## 2026-02-09 — Phase 3: Memory + Token 优化

### 新增

- `agent/memory/`：checkpointer（SQLite）、profile（长期画像）
- `/thread <id>`、`clear` 会话管理；消息窗口限制（MAX_MESSAGES）
- `scripts/inspect_db.py` 数据库检查

### 修复

- `scripts/inspect_db.py` 路径转义问题

---

## 2026-02-08 — Phase 2: RAG + 多格式支持

### 新增

- `agent/rag/`：loader（txt/md/pdf/docx/xlsx/pptx、图/音/视频）、vectorstore（FAISS）、retriever（knowledge_search）
- `scripts/index_docs.py`（含 `--append`、`--batch-size` 等）
- 模型配置集中到 `config.py`

---

## 2026-02-07 — Phase 1: 基础 Agent

### 新增

- 项目基础：`.gitignore`、`.env`、`requirements.txt`
- `agent/`：config、llm（ChatTongyi）、tools（日期时间、计算器、DuckDuckGo）、callbacks
- CLI（`main.py`）+ 基础测试（`test.py`）

---

*本文档为整理版，与根目录 `CHANGELOG.md` 并存，不替代原文件。*
