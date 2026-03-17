"""MyAgent WebUI — Gradio 前端

功能:
- 单智能体 / 多智能体模式切换
- RAG 知识库 开/关 + 模式选择
- VL 视觉能力选配（图生文 / 文生图）
- 会话管理（新建 / 切换 / 重命名 / 删除）
- 中间过程实时展示（工具调用、流式文本）
- Token 统计 & 长期记忆面板
"""

import re
import uuid
import asyncio
import logging
import threading
import os
import socket

import gradio as gr
from langchain_core.messages import HumanMessage, ToolMessage

from agent.config import (
    MAX_MESSAGES, MAX_TOOL_ITERATIONS,
    ENABLE_RAG, RAG_MODE,
    ENABLE_TEXT_TO_IMAGE, ENABLE_IMAGE_TO_TEXT, ENABLE_VL,
    TOOL_EVIDENCE_REQUIRED, TOOL_EVIDENCE_MAX_CHARS,
    TEXT_TO_IMAGE_MODEL, VISION_MODEL,
)
from agent.llm import get_llm
from agent.tools import get_all_tools
from agent.rag.retriever import create_rag_tool
from agent.memory.checkpointer import (
    get_checkpointer_context, list_sessions, rename_session, delete_session,
)
from agent.graph import build_agent
from agent.multi import build_multi_agent_graph
from agent.multi import event_bus
from agent.callbacks import get_token_counter, reset_token_counter
from main import SYSTEM_PROMPT
from agent.memory.profile import get_profile_summary, clear_profile

logger = logging.getLogger(__name__)

# ━━━━━━━━━━━━━━ 会话名称规范化 ━━━━━━━━━━━━━━

def _sanitize_thread_name(name: str, max_len: int = 32) -> str:
    if not name or not isinstance(name, str):
        return ""
    s = name.strip()
    s = re.sub(r"[^\w\u4e00-\u9fff\-]", "_", s)
    s = re.sub(r"_+", "_", s).strip("_")
    return s[:max_len] if s else ""


# ━━━━━━━━━━━━━━ 全局 Agent 状态 ━━━━━━━━━━━━━━

logger.info("初始化 LLM...")
llm = get_llm(for_tools=True)
logger.info("LLM 初始化完成")

global_agent = None
global_multi_agent = None
global_checkpointer = None
_checkpointer_cm = None  # 保存上下文管理器，用于正确释放资源
_rag_enabled = ENABLE_RAG
_rag_mode = RAG_MODE
_agent_mode = "single"
_enable_t2i = ENABLE_TEXT_TO_IMAGE
_enable_i2t = ENABLE_IMAGE_TO_TEXT
_init_lock = threading.Lock()


def init_agent_sync(
    rag_enabled: bool = True,
    rag_mode: str = "advanced",
    agent_mode: str = "single",
    enable_t2i: bool = False,
    enable_i2t: bool = False,
):
    global global_agent, global_multi_agent, global_checkpointer
    global _rag_enabled, _rag_mode, _agent_mode, _enable_t2i, _enable_i2t

    with _init_lock:
        _rag_enabled = rag_enabled
        _rag_mode = rag_mode
        _agent_mode = agent_mode
        _enable_t2i = enable_t2i
        _enable_i2t = enable_i2t

        logger.info("初始化 Agent (RAG=%s/%s, mode=%s, T2I=%s, I2T=%s)",
                     rag_enabled, rag_mode, agent_mode, enable_t2i, enable_i2t)

        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)

        async def _init():
            global global_checkpointer, _checkpointer_cm
            if global_checkpointer is None:
                _checkpointer_cm = get_checkpointer_context()
                global_checkpointer = await _checkpointer_cm.__aenter__()

            tools = list(get_all_tools(enable_t2i=enable_t2i, enable_i2t=enable_i2t))
            if rag_enabled:
                rag_tool = create_rag_tool(mode=rag_mode)
                if rag_tool:
                    tools.append(rag_tool)

            return build_agent(llm, tools, global_checkpointer, SYSTEM_PROMPT,
                               MAX_MESSAGES, MAX_TOOL_ITERATIONS)

        global_agent = loop.run_until_complete(_init())
        global_multi_agent = build_multi_agent_graph()

        logger.info("Agent 初始化完成")


init_agent_sync(ENABLE_RAG, RAG_MODE, "single", ENABLE_TEXT_TO_IMAGE, ENABLE_IMAGE_TO_TEXT)


# ━━━━━━━━━━━━━━ 核心交互逻辑 ━━━━━━━━━━━━━━

_generated_images: list[str] = []

async def bot_response(message, thread_id: str):
    """处理用户消息，支持流式输出和中间过程推送。"""
    global global_agent, global_multi_agent, _agent_mode
    _generated_images.clear()

    if isinstance(message, list):
        text_parts = []
        for item in message:
            if isinstance(item, dict) and item.get("type") == "text":
                text_parts.append(item.get("text", ""))
            elif isinstance(item, str):
                text_parts.append(item)
        message = "".join(text_parts)
    else:
        message = str(message) if message is not None else ""

    if not message or not message.strip():
        yield "请输入有效内容。"
        return

    # ── 多智能体模式 ──
    if _agent_mode == "multi":
        if global_multi_agent is None:
            yield "多智能体系统未初始化，请刷新页面重试。"
            return

        try:
            logger.info("[Multi-Agent] 收到: %s", message[:30])

            initial_state = {
                "messages": [],
                "iteration_count": 0,
                "task_plan": [],
                "current_worker": "supervisor",
                "worker_results": {},
                "handoff_context": "",
                "original_query": message,
            }

            loop = asyncio.get_running_loop()
            import concurrent.futures
            with concurrent.futures.ThreadPoolExecutor() as pool:
                task = loop.run_in_executor(pool, global_multi_agent.invoke, initial_state)
                while not task.done():
                    yield None
                    await asyncio.sleep(0.15)
                result = task.result()

            messages = result.get("messages", [])
            if messages:
                final_msg = messages[-1]
                content = final_msg.content if hasattr(final_msg, "content") else str(final_msg)
                yield content
            else:
                yield "（多智能体处理完成，但没有生成回答）"
        except Exception as e:
            import traceback
            traceback.print_exc()
            yield f"多智能体出错: {str(e)}"
        return

    # ── 单智能体模式（流式 astream_events）──
    if global_agent is None:
        yield "Agent 未初始化，请刷新页面重试。"
        return

    config = {"configurable": {"thread_id": thread_id}}

    try:
        logger.info("[Agent] 收到: %s", message[:30])
        accumulated_text = ""

        async for ev in global_agent.astream_events(
            {"messages": [HumanMessage(content=message)]},
            config=config,
            version="v2",
        ):
            kind = ev.get("event", "")

            if kind == "on_tool_start":
                tool_name = ev.get("name", "unknown")
                event_bus.emit(f"🔧 调用工具: {tool_name}")
                yield None

            elif kind == "on_tool_end":
                tool_name = ev.get("name", "")
                event_bus.emit(f"✅ 工具完成: {tool_name}")
                tool_output = ev.get("data", {}).get("output", "")
                if hasattr(tool_output, "content"):
                    tool_output = tool_output.content
                tool_output = str(tool_output) if tool_output else ""
                if "[GENERATED_IMAGE]" in tool_output:
                    import re as _re
                    for m in _re.finditer(r"\[GENERATED_IMAGE\](.*?)\[/GENERATED_IMAGE\]", tool_output):
                        _generated_images.append(m.group(1))
                yield None

            elif kind == "on_chat_model_stream":
                chunk = ev["data"].get("chunk")
                if chunk and hasattr(chunk, "content") and chunk.content:
                    c = chunk.content
                    if isinstance(c, str):
                        accumulated_text += c
                    elif isinstance(c, list):
                        for item in c:
                            if isinstance(item, dict) and item.get("type") == "text":
                                accumulated_text += item.get("text", "")
                            elif isinstance(item, str):
                                accumulated_text += item
                    yield accumulated_text

        if not accumulated_text:
            result = await global_agent.ainvoke(
                {"messages": [HumanMessage(content=message)]}, config=config,
            )
            msgs = result.get("messages", [])
            evidence = _build_tool_evidence_suffix(msgs)
            for msg in reversed(msgs):
                if hasattr(msg, "type") and msg.type in ("ai", "assistant"):
                    if hasattr(msg, "content") and msg.content:
                        ct = msg.content
                        if isinstance(ct, str):
                            yield ct + evidence
                        return
            yield "（无响应）"
        else:
            yield accumulated_text

    except Exception as e:
        import traceback
        traceback.print_exc()
        err_text = str(e)
        if "function.arguments" in err_text and "JSON format" in err_text:
            fallback_thread = str(uuid.uuid4())[:8]
            try:
                retry_result = await global_agent.ainvoke(
                    {"messages": [HumanMessage(content=message)]},
                    config={"configurable": {"thread_id": fallback_thread}},
                )
                retry_msgs = retry_result.get("messages", [])
                for msg in reversed(retry_msgs):
                    if hasattr(msg, "type") and msg.type in ("ai", "assistant") and msg.content:
                        ct = msg.content if isinstance(msg.content, str) else str(msg.content)
                        yield ct + "\n\n（检测到旧会话异常，已自动切换新会话重试成功）"
                        return
            except Exception:
                pass
        yield f"出错了: {err_text}"


def _build_tool_evidence_suffix(messages) -> str:
    if not TOOL_EVIDENCE_REQUIRED:
        return ""
    if not isinstance(messages, list) or not messages:
        return ""

    picked, seen = [], set()
    target_names = ("web_search", "get_weather", "get_current_datetime")
    for m in reversed(messages):
        if not isinstance(m, ToolMessage):
            continue
        name = (getattr(m, "name", "") or "").strip()
        if not name:
            name = str(getattr(m, "additional_kwargs", {}).get("name", "")).strip()
        if name not in target_names or name in seen:
            continue
        raw = m.content if isinstance(m.content, str) else str(m.content)
        raw = re.sub(r"\s+", " ", raw).strip()
        if len(raw) > TOOL_EVIDENCE_MAX_CHARS:
            raw = raw[:TOOL_EVIDENCE_MAX_CHARS] + "..."
        picked.append((name, raw or "（空输出）"))
        seen.add(name)
        if len(picked) >= 2:
            break
    if not picked:
        return ""
    lines = ["\n\n【工具证据摘要】"]
    for name, text in picked:
        lines.append(f"- {name}: {text}")
    return "\n".join(lines)


# ━━━━━━━━━━━━━━ CSS 样式（现代仪表盘风格）━━━━━━━━━━━━━━

CUSTOM_CSS = """
/* ========== 全局 ========== */
:root {
    --sidebar-width: 280px;
    --primary: #2563eb;
    --primary-light: #dbeafe;
    --bg-page: #f8fafc;
    --bg-card: #ffffff;
    --bg-sidebar: #f1f5f9;
    --border: #e2e8f0;
    --text-primary: #1e293b;
    --text-secondary: #64748b;
    --text-muted: #94a3b8;
    --radius: 10px;
    --shadow-sm: 0 1px 3px rgba(0,0,0,0.06);
    --shadow-md: 0 4px 12px rgba(0,0,0,0.08);
}

.dark {
    --bg-page: #0f172a;
    --bg-card: #1e293b;
    --bg-sidebar: #1e293b;
    --border: #334155;
    --text-primary: #f1f5f9;
    --text-secondary: #94a3b8;
    --text-muted: #64748b;
}

.gradio-container {
    max-width: 1480px !important;
    margin: 0 auto !important;
    font-family: 'Inter', -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, sans-serif !important;
    background: var(--bg-page) !important;
}

/* ========== 顶栏 ========== */
.topbar {
    background: var(--bg-card) !important;
    border-bottom: 1px solid var(--border) !important;
    padding: 0.75rem 1.5rem !important;
    margin: 0 -1rem 0 -1rem !important;
    box-shadow: var(--shadow-sm) !important;
}
.topbar h1 { font-size: 1.35rem !important; font-weight: 700 !important; color: var(--text-primary) !important; margin: 0 !important; }
.topbar p { font-size: 0.8rem !important; color: var(--text-secondary) !important; margin: 0.15rem 0 0 0 !important; }

/* ========== 侧栏 ========== */
.sidebar {
    background: transparent !important;
    padding: 0 !important;
    display: flex !important;
    flex-direction: column !important;
    gap: 0.75rem !important;
}

.sidebar-card {
    background: var(--bg-card) !important;
    border: 1px solid var(--border) !important;
    border-radius: var(--radius) !important;
    padding: 0.875rem 1rem !important;
    box-shadow: var(--shadow-sm) !important;
}

.sidebar-card h3 {
    font-size: 0.75rem !important;
    font-weight: 700 !important;
    color: var(--text-secondary) !important;
    text-transform: uppercase !important;
    letter-spacing: 0.05em !important;
    margin: 0 0 0.6rem 0 !important;
}

/* ========== 聊天主区域 ========== */
.chat-main {
    background: var(--bg-card) !important;
    border: 1px solid var(--border) !important;
    border-radius: var(--radius) !important;
    padding: 1rem !important;
    box-shadow: var(--shadow-sm) !important;
    display: flex !important;
    flex-direction: column !important;
}

.chat-main .gr-chatbot {
    border-radius: var(--radius) !important;
    flex: 1 !important;
}

/* ========== 输入行 ========== */
.input-row {
    display: flex !important;
    gap: 0.6rem !important;
    margin-top: 0.75rem !important;
    align-items: flex-end !important;
}
.input-row .gr-textbox { border-radius: var(--radius) !important; }
.input-row .gr-button { border-radius: var(--radius) !important; font-weight: 600 !important; }

/* ========== 执行过程面板 ========== */
#process-log textarea {
    background: #0f172a !important;
    color: #a5f3fc !important;
    font-family: 'Cascadia Code', 'Fira Code', 'Consolas', monospace !important;
    font-size: 0.78rem !important;
    border: 1px solid #1e293b !important;
    border-radius: 8px !important;
    padding: 0.6rem 0.8rem !important;
    line-height: 1.5 !important;
}

/* ========== 模式徽章 ========== */
.badge-single { background: #eff6ff !important; border: 1px solid #bfdbfe !important; color: #1e40af !important; border-radius: 6px !important; padding: 0.5rem 0.75rem !important; font-size: 0.8rem !important; }
.badge-multi { background: #f5f3ff !important; border: 1px solid #ddd6fe !important; color: #5b21b6 !important; border-radius: 6px !important; padding: 0.5rem 0.75rem !important; font-size: 0.8rem !important; }

/* ========== 按钮行 ========== */
.btn-row { display: flex !important; gap: 0.4rem !important; }
.btn-row .gr-button { flex: 1 !important; border-radius: 6px !important; font-size: 0.8rem !important; }

/* ========== Accordion 统一 ========== */
.sidebar .gr-accordion { background: var(--bg-card) !important; border: 1px solid var(--border) !important; border-radius: var(--radius) !important; box-shadow: var(--shadow-sm) !important; overflow: hidden !important; }
.sidebar .gr-accordion .gr-form { border: none !important; }

/* ========== 提示文字 ========== */
.hint-text { font-size: 0.72rem !important; color: var(--text-muted) !important; margin-top: 0.2rem !important; }

/* ========== 响应式 ========== */
@media (max-width: 768px) {
    .gradio-container { padding: 0 0.5rem !important; }
    .sidebar { min-width: 100% !important; }
}
"""


# ━━━━━━━━━━━━━━ 辅助函数 ━━━━━━━━━━━━━━

def _get_session_choices():
    """获取会话下拉列表选项。"""
    sessions = list_sessions()
    if not sessions:
        return []
    return [s["thread_id"] for s in sessions]


# ━━━━━━━━━━━━━━ 构建界面 ━━━━━━━━━━━━━━

with gr.Blocks(title="MyAgent 智能体助手") as demo:

    # 状态
    session_id = gr.State(value=lambda: str(uuid.uuid4())[:8])
    pending_images = gr.State(value=[])  # 暂存本轮上传的图片路径

    # ══════════ 顶栏 ══════════
    with gr.Row(elem_classes="topbar"):
        with gr.Column(scale=4):
            gr.Markdown("# MyAgent")
            gr.Markdown("多智能体协作 · RAG 知识库 · 长期记忆 · 实时工具调用")

    # ══════════ 主内容 ══════════
    with gr.Row():

        # ──── 左侧栏 ────
        with gr.Column(scale=1, min_width=280, elem_classes="sidebar"):

            # 会话管理
            with gr.Group(elem_classes="sidebar-card"):
                gr.Markdown("### 💬 会话管理")
                session_dropdown = gr.Dropdown(
                    choices=_get_session_choices(),
                    label="历史会话",
                    interactive=True,
                    allow_custom_value=True,
                )
                with gr.Row(elem_classes="btn-row"):
                    new_thread_btn = gr.Button("新建", variant="primary", size="sm")
                    rename_btn = gr.Button("重命名", size="sm")
                    delete_btn = gr.Button("删除", variant="stop", size="sm")
                rename_input = gr.Textbox(
                    label="新名称", visible=False, max_lines=1, container=False,
                )
                rename_confirm_btn = gr.Button("确认重命名", visible=False, size="sm")

            # 智能体模式
            with gr.Group(elem_classes="sidebar-card"):
                gr.Markdown("### 🤖 智能体模式")
                agent_mode_radio = gr.Radio(
                    choices=[("单智能体", "single"), ("多智能体协作", "multi")],
                    value="single", label="", interactive=True,
                )
                mode_status = gr.Markdown(
                    "**单智能体模式** — 通用问答 + 工具调用 + RAG",
                    elem_classes="badge-single",
                )

            # RAG 知识库
            with gr.Group(elem_classes="sidebar-card"):
                gr.Markdown("### 📚 知识库 RAG")
                rag_enabled_cb = gr.Checkbox(label="启用 RAG 知识库", value=ENABLE_RAG, interactive=True)
                rag_mode_radio = gr.Radio(
                    choices=[("FAISS 向量检索", "classic"), ("混合检索 (推荐)", "advanced")],
                    value=RAG_MODE, label="", interactive=ENABLE_RAG,
                )

            # 视觉能力
            with gr.Group(elem_classes="sidebar-card"):
                gr.Markdown("### 👁 视觉能力 (选配)")
                vl_t2i_cb = gr.Checkbox(label=f"文生图 (T2I: {TEXT_TO_IMAGE_MODEL})", value=ENABLE_TEXT_TO_IMAGE, interactive=True)
                vl_i2t_cb = gr.Checkbox(label=f"图生文 (VL: {VISION_MODEL})", value=ENABLE_IMAGE_TO_TEXT, interactive=True)
                vl_hint = gr.Markdown(
                    "启用前请确保已在阿里云开通对应模型权限，并在 `.env` 中配置 API Key",
                    elem_classes="hint-text",
                )

            # 应用设置按钮
            apply_all_btn = gr.Button("应用设置", variant="primary", size="sm")

            # 记忆管理
            with gr.Accordion("🧠 长期记忆", open=False):
                current_memory_box = gr.Textbox(
                    label="", value=get_profile_summary(),
                    interactive=False, lines=4, max_lines=6,
                )
                with gr.Row(elem_classes="btn-row"):
                    refresh_mem_btn = gr.Button("刷新", size="sm")
                    clear_mem_btn = gr.Button("清除", variant="stop", size="sm")

            # Token 统计
            with gr.Accordion("📊 Token 统计", open=True):
                token_stats_box = gr.Textbox(
                    label="", value="暂无统计数据",
                    interactive=False, lines=4, max_lines=5,
                )

            # 工具箱
            with gr.Accordion("🧰 工具箱", open=False):
                tool_names = [t.name for t in get_all_tools()]
                tool_list_display = gr.CheckboxGroup(
                    choices=tool_names, value=tool_names,
                    label="已加载工具", interactive=False,
                )

        # ──── 右侧聊天区 ────
        with gr.Column(scale=3, elem_classes="chat-main"):

            # 兼容不同版本的 Gradio：部分版本的 Chatbot 不支持 type 参数
            _chatbot_kwargs = dict(
                height=500,
                show_label=False,
                avatar_images=(None, "https://api.dicebear.com/7.x/bottts/svg?seed=agent"),
            )
            try:
                chatbot = gr.Chatbot(type="messages", **_chatbot_kwargs)
            except TypeError:
                chatbot = gr.Chatbot(**_chatbot_kwargs)

            with gr.Row(elem_classes="input-row"):
                image_input = gr.Files(
                    label="拖拽图片到这里",
                    file_types=["image"],
                    file_count="multiple",
                    height=90,
                    scale=2,
                    show_label=True,
                )
                msg_input = gr.MultimodalTextbox(
                    placeholder="输入问题或上传图片，按 Enter 发送... (Shift+Enter 换行)",
                    sources=["upload"],
                    file_types=["image"],
                    file_count="multiple",
                    show_label=False,
                    scale=4,
                    lines=1,
                    max_lines=5,
                    autofocus=True,
                )
                send_btn = gr.Button("发送", variant="primary", scale=1, min_width=80)

            with gr.Accordion("🔍 执行过程", open=True):
                process_log_box = gr.Textbox(
                    label="", value="等待执行...",
                    interactive=False, lines=6, max_lines=15,
                    autoscroll=True, elem_id="process-log",
                )

    # ━━━━━━━━━━━━━━ 事件处理 ━━━━━━━━━━━━━━

    # ── 页面加载 ──
    def on_load(sid):
        choices = _get_session_choices()
        return sid or "", gr.update(choices=choices, value=sid if sid in choices else None)

    demo.load(on_load, [session_id], [session_id, session_dropdown])

    # ── 会话切换 ──
    def switch_session(selected):
        if not selected:
            return gr.update(), gr.update()
        return selected, []

    session_dropdown.change(switch_session, [session_dropdown], [session_id, chatbot])

    # ── 新建会话 ──
    def new_session():
        new_id = str(uuid.uuid4())[:8]
        reset_token_counter()
        event_bus.clear()
        choices = _get_session_choices()
        if new_id not in choices:
            choices.append(new_id)
        return (
            new_id,
            gr.update(choices=choices, value=new_id),
            [],
            "暂无统计数据",
            "等待执行...",
        )

    new_thread_btn.click(
        new_session, None,
        [session_id, session_dropdown, chatbot, token_stats_box, process_log_box],
    )

    # ── 重命名会话 ──
    def show_rename():
        return gr.update(visible=True), gr.update(visible=True)

    def do_rename(old_id, new_name):
        sanitized = _sanitize_thread_name(new_name)
        if not sanitized:
            gr.Warning("名称无效")
            return old_id, gr.update(), gr.update(visible=False), gr.update(visible=False)
        rename_session(old_id, sanitized)
        choices = _get_session_choices()
        return (
            sanitized,
            gr.update(choices=choices, value=sanitized),
            gr.update(visible=False),
            gr.update(visible=False),
        )

    rename_btn.click(show_rename, None, [rename_input, rename_confirm_btn])
    rename_confirm_btn.click(
        do_rename, [session_id, rename_input],
        [session_id, session_dropdown, rename_input, rename_confirm_btn],
    )

    # ── 删除会话 ──
    def do_delete(current_id, selected):
        target = selected or current_id
        if not target:
            return current_id, gr.update(), gr.update()
        delete_session(target)
        new_id = str(uuid.uuid4())[:8] if target == current_id else current_id
        choices = _get_session_choices()
        return new_id, gr.update(choices=choices, value=new_id if new_id in choices else None), []

    delete_btn.click(do_delete, [session_id, session_dropdown], [session_id, session_dropdown, chatbot])

    # ── 切换智能体模式 ──
    def on_mode_change(mode):
        if mode == "multi":
            return gr.Markdown(
                "**多智能体协作** — Supervisor + Code/Data/Writer",
                elem_classes="badge-multi",
            )
        return gr.Markdown(
            "**单智能体模式** — 通用问答 + 工具调用 + RAG",
            elem_classes="badge-single",
        )

    agent_mode_radio.change(on_mode_change, [agent_mode_radio], [mode_status])

    # ── RAG 启用/禁用联动 ──
    def on_rag_toggle(enabled):
        return gr.update(interactive=enabled)

    rag_enabled_cb.change(on_rag_toggle, [rag_enabled_cb], [rag_mode_radio])

    # ── 应用所有设置 ──
    def apply_all_settings(agent_mode, rag_en, rag_m, t2i, i2t):
        init_agent_sync(
            rag_enabled=rag_en, rag_mode=rag_m,
            agent_mode=agent_mode, enable_t2i=t2i, enable_i2t=i2t,
        )
        tool_names = [t.name for t in get_all_tools(enable_t2i=t2i, enable_i2t=i2t)]
        mode_text = "多智能体协作" if agent_mode == "multi" else "单智能体"
        gr.Info(f"已应用: {mode_text} | RAG={'开' if rag_en else '关'}")
        return gr.update(choices=tool_names, value=tool_names)

    apply_all_btn.click(
        apply_all_settings,
        [agent_mode_radio, rag_enabled_cb, rag_mode_radio, vl_t2i_cb, vl_i2t_cb],
        [tool_list_display],
    )

    # ── 记忆管理 ──
    refresh_mem_btn.click(lambda: get_profile_summary(), None, [current_memory_box])

    def clear_and_refresh():
        clear_profile()
        return get_profile_summary()

    clear_mem_btn.click(clear_and_refresh, None, [current_memory_box])

    # ── 发送消息 ──
    def _resolve_path(f):
        """从各种格式中提取文件路径字符串。"""
        if isinstance(f, str):
            return f
        if hasattr(f, "path"):
            return str(f.path)
        if isinstance(f, dict):
            return str(f.get("path") or f.get("name") or "")
        return str(f)

    def add_user_message(history, payload, images):
        # payload: str | dict(text+files) from MultimodalTextbox; images: list from gr.Files
        if payload is None and not images:
            return history, gr.update(), gr.update(), gr.update(), []
        text = ""
        files_from_payload = []
        if isinstance(payload, dict):
            text = str(payload.get("text") or "")
            files_from_payload = list(payload.get("files") or [])
        elif isinstance(payload, str):
            text = payload
        if not text.strip() and not files_from_payload and not images:
            return history, gr.update(), gr.update(), gr.update(), []

        history = history or []

        file_paths = [_resolve_path(f) for f in files_from_payload if f]
        if images:
            file_paths.extend([_resolve_path(f) for f in images if f])
        file_paths = [p for p in file_paths if p]

        content = []
        if text and text.strip():
            content.append({"type": "text", "text": text.strip()})
        for p in file_paths:
            content.append({"path": p})

        if len(content) == 1 and isinstance(content[0], dict) and content[0].get("type") == "text":
            history.append({"role": "user", "content": content[0]["text"]})
        elif content:
            history.append({"role": "user", "content": content})
        else:
            history.append({"role": "user", "content": text or ""})
        history.append({"role": "assistant", "content": ""})
        return (
            history,
            gr.update(value="", interactive=False),
            gr.update(interactive=False),
            gr.update(value=None, interactive=False),
            file_paths,  # 存入 pending_images State
        )

    def _extract_text_and_images(raw_content):
        """从 Chatbot 的 MessageDict.content 中提取文本与图片路径列表。"""
        if raw_content is None:
            return "", []
        # MultimodalTextbox 常见输出: {"text": "...", "files": [path|{...}]}
        if isinstance(raw_content, dict):
            text = str(raw_content.get("text") or "")
            files = raw_content.get("files") or []
            image_paths = []
            if isinstance(files, (list, tuple)):
                for f in files:
                    if isinstance(f, str):
                        image_paths.append(f)
                    elif isinstance(f, dict):
                        p = f.get("path") or f.get("name")
                        if p:
                            image_paths.append(str(p))
            return text, image_paths
        # content 是 list（如 [{"type":"text",...},{"type":"file",...}]）
        if isinstance(raw_content, list):
            text_parts = []
            image_paths = []
            for item in raw_content:
                if isinstance(item, dict) and item.get("type") == "text":
                    text_parts.append(item.get("text", ""))
                elif isinstance(item, dict) and "path" in item and "type" not in item:
                    image_paths.append(str(item["path"]))
                elif isinstance(item, dict) and item.get("type") == "file":
                    f = item.get("file") or {}
                    p = f.get("path") or item.get("path") or item.get("name")
                    if p:
                        image_paths.append(str(p))
                elif isinstance(item, str):
                    text_parts.append(item)
            return "".join(text_parts), image_paths
        return str(raw_content), []

    async def _describe_images_if_enabled(image_paths: list[str]) -> str:
        """若启用图生文，则对上传图片生成描述；否则返回提示。"""
        if not image_paths:
            return ""
        if not _enable_i2t:
            return "（检测到你上传了图片，但当前未启用“图生文”。请在左侧栏勾选“图生文”并点击“应用设置”后再发送。）"

        try:
            tools = get_all_tools(enable_t2i=_enable_t2i, enable_i2t=True)
            describe_tool = next((t for t in tools if getattr(t, "name", "") == "describe_image"), None)
            if describe_tool is None:
                return "（图生文工具未加载，请在左侧栏启用后点击“应用设置”。）"

            parts = []
            for p in image_paths:
                def _run_one():
                    if hasattr(describe_tool, "invoke"):
                        return describe_tool.invoke({"image_path": p})
                    return describe_tool(p)

                desc = await asyncio.to_thread(_run_one)
                parts.append(f"[图片 {p} 的描述]\n{desc}")
            return "\n\n".join(parts)
        except Exception as e:
            return f"（图生文失败: {e}）"

    async def generate_response(history, thread_id, img_paths):
        if not history or len(history) < 2:
            yield history, gr.update(), gr.update()
            return

        last = history[-1]
        prev = history[-2]

        if not (isinstance(last, dict) and last.get("role") == "assistant"):
            yield history, gr.update(), gr.update()
            return

        raw_content = prev.get("content", "")
        user_text, _ = _extract_text_and_images(raw_content)
        user_text = (user_text or "").strip()

        image_paths = img_paths or []
        image_context = await _describe_images_if_enabled(image_paths)
        if image_context:
            user_message = (user_text + "\n\n" if user_text else "") + image_context
        else:
            user_message = user_text

        reset_token_counter()
        event_bus.clear()
        accumulated_log = ""

        async for chunk in bot_response(user_message, thread_id):
            if chunk is not None:
                last["content"] = chunk
                history[-1] = last

            new_events = event_bus.drain()
            if new_events:
                accumulated_log = accumulated_log + ("\n" if accumulated_log else "") + new_events

            yield history, accumulated_log or "执行中...", gr.update()

        final_events = event_bus.drain()
        if final_events:
            accumulated_log = accumulated_log + ("\n" if accumulated_log else "") + final_events

        if _generated_images:
            for img_path in _generated_images:
                history.append({
                    "role": "assistant",
                    "content": gr.FileData(path=img_path, mime_type="image/png"),
                })
                history.append({
                    "role": "assistant",
                    "content": f"图片已保存至本地: `{img_path}`\n\n"
                               f"[点击下载图片](file={img_path})",
                })
            _generated_images.clear()

        token_summary = get_token_counter().summary()
        yield history, accumulated_log or "执行完毕", token_summary

    def reenable():
        return gr.update(interactive=True), gr.update(interactive=True), gr.update(interactive=True)

    msg_input.submit(
        add_user_message, [chatbot, msg_input, image_input],
        [chatbot, msg_input, send_btn, image_input, pending_images], queue=False,
    ).then(
        generate_response, [chatbot, session_id, pending_images],
        [chatbot, process_log_box, token_stats_box],
    ).then(
        reenable, None, [msg_input, send_btn, image_input], queue=False,
    )

    send_btn.click(
        add_user_message, [chatbot, msg_input, image_input],
        [chatbot, msg_input, send_btn, image_input, pending_images], queue=False,
    ).then(
        generate_response, [chatbot, session_id, pending_images],
        [chatbot, process_log_box, token_stats_box],
    ).then(
        reenable, None, [msg_input, send_btn, image_input], queue=False,
    )


# ━━━━━━━━━━━━━━ 启动 ━━━━━━━━━━━━━━

async def _cleanup_checkpointer():
    """清理 checkpointer 异步资源。"""
    global _checkpointer_cm
    if _checkpointer_cm is not None:
        try:
            await _checkpointer_cm.__aexit__(None, None, None)
            logger.info("Checkpointer 资源已释放")
        except Exception as e:
            logger.warning("释放 checkpointer 资源时出错: %s", e)
        _checkpointer_cm = None


def _is_port_free(port: int) -> bool:
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        s.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        try:
            s.bind(("127.0.0.1", port))
        except OSError:
            return False
    return True


def _find_free_port(base_port: int = 7860, max_offset: int = 50) -> int:
    env_port = os.getenv("MYAGENT_PORT") or os.getenv("PORT")
    if env_port and env_port.isdigit():
        candidate = int(env_port)
        if _is_port_free(candidate):
            return candidate
    for offset in range(max_offset + 1):
        port = base_port + offset
        if _is_port_free(port):
            return port
    return base_port


if __name__ == "__main__":
    import signal
    import sys

    base_env = os.getenv("MYAGENT_BASE_PORT") or os.getenv("PORT_BASE")
    try:
        base_port = int(base_env) if base_env and base_env.isdigit() else 7860
    except ValueError:
        base_port = 7860

    server_port = _find_free_port(base_port)

    def _shutdown(sig, frame):
        logger.info("正在停止服务，释放端口 %d...", server_port)
        # 同步执行异步清理
        try:
            loop = asyncio.new_event_loop()
            asyncio.set_event_loop(loop)
            loop.run_until_complete(_cleanup_checkpointer())
        except Exception as e:
            logger.warning("清理资源时出错: %s", e)
        try:
            demo.close()
        except Exception:
            pass
        logger.info("服务已停止")
        sys.exit(0)

    signal.signal(signal.SIGINT, _shutdown)
    signal.signal(signal.SIGTERM, _shutdown)

    try:
        logger.info("使用端口 %d", server_port)
        demo.launch(
            server_name="0.0.0.0",
            server_port=server_port,
            css=CUSTOM_CSS,
            theme=gr.themes.Soft(
                primary_hue="blue",
                secondary_hue="slate",
                neutral_hue="slate",
                font=gr.themes.GoogleFont("Inter"),
                radius_size="sm",
            ),
        )
    except KeyboardInterrupt:
        logger.info("检测到 Ctrl+C，停止服务...")
        demo.close()
