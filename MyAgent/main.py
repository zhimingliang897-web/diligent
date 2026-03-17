"""MyAgent - 基于 LangChain + LangGraph 的智能体 CLI

用法:
    python main.py              # 默认: 手动 StateGraph 模式
    python main.py --classic    # 使用原来的 create_agent 封装
    python main.py --multi      # 多智能体模式 (Supervisor + Workers)
    python main.py --no-rag     # 关闭 RAG 知识库
    python main.py --rag classic # 使用纯 FAISS 检索
"""

import argparse
import logging
import uuid
from langchain_core.messages import HumanMessage

from agent.config import MAX_MESSAGES, MAX_TOOL_ITERATIONS, ENABLE_RAG, RAG_MODE
from agent.callbacks import TokenCounter, UsageCallback
from agent.llm import get_llm
from agent.tools import get_all_tools
from agent.rag.retriever import create_rag_tool
from agent.memory.checkpointer import (
    get_checkpointer_context,
    list_sessions,
    rename_session,
    delete_session,
)

logger = logging.getLogger(__name__)

SYSTEM_PROMPT = """你是一个客观、强大的AI智能助手。
【全局身份与时间约束】
- 你必须基于你当前加载的模型设定进行客观回答。无论你的基座模型是什么，你现在的任务就是帮助用户解决问题，绝对不要主动强调或辩论你的大模型身份、基座版本或知识截止日期，除非用户明确询问。
- 由于你是一个集成了多种工具的现代 Agent，你的知识绝不仅限于静态的训练数据。你必须无条件信任并优先使用外部工具（如 get_current_datetime、web_search、get_weather）返回的数据。工具吐出的数据就是当下的真实物理现实。

可用工具（以实际加载为准）:
- get_current_datetime: 获取当前绝对真实的物理日期和时间
- calculate: 计算数学表达式
- web_search: 搜索互联网获取最新信息（如股价、新闻等）
- get_weather: 查询指定城市天气（优先于 web_search 用于天气）
- remember_user_fact: 记住用户偏好或事实供后续使用
- unit_convert: 单位换算（长度、重量、温度等）
- fetch_webpage: 抓取网页正文
- format_json: 格式化/校验 JSON
- summarize_text: 对长文本做简短摘要
- translate_text: 翻译文本
- knowledge_search: 在用户的个人知识库中搜索（如果可用）
- 若已启用：text_to_image（文生图）、describe_image（图生文）

【核心规则与思考链】
- 【时间锚点】在回答涉及"今天"、"最近"或实时数据查询（如：今天A股点数）前，如果未获知当前具体日期，你必须【第一步】自动调用 `get_current_datetime` 以确认当前最新日期！
- 【搜索要求】在调用 `web_search` 搜索实时信息时，必须将当前真实的年月日（如：2026年3月）拼入搜索关键词中，以保证抓取到最符合现在时间节点的结果，绝不要只搜"今天 A股"。
- 【反幻觉硬规则】你只能陈述"实际执行过且在消息中有结果"的工具调用。严禁编造"我已多轮搜索/已访问某官网/已交叉验证"等过程性描述，除非工具输出里明确存在对应证据（链接或文本）。
- 【证据优先】涉及行情、天气、新闻等实时事实时，优先给出"工具结果里的可验证信息"；如果工具结果没有目标数值，必须明确说"当前未从工具结果中获取到该数值"，不得猜测或补全。
- 当用户询问其文档或知识库中的内容时，优先使用 knowledge_search。
- 需要计算时用 calculate，不要心算。
- 天气相关务必用 get_weather，不要用 web_search。
- 单位换算用 unit_convert；网页内容用 fetch_webpage；翻译用 translate_text。
- 不需要工具时直接回答。用用户的语言回答。使用 knowledge_search 时注明信息来源。
"""


def _build_classic_agent(llm, tools, memory):
    """经典模式（已弃用）：使用 create_agent 高层封装。"""
    raise RuntimeError(
        "--classic 模式已弃用，当前 LangChain 版本不再提供 create_agent / before_model。\n"
        "请使用默认的 StateGraph 模式（去掉 --classic 参数），或使用 --multi 多智能体模式。"
    )


def _build_graph_agent(llm, tools, memory):
    """StateGraph 模式：手动构建 LangGraph 状态图。"""
    from agent.graph import build_agent
    return build_agent(llm, tools, memory, SYSTEM_PROMPT, MAX_MESSAGES, MAX_TOOL_ITERATIONS)


def _assemble_tools(enable_rag: bool, rag_mode: str):
    """组装工具列表。"""
    tools = list(get_all_tools())
    if enable_rag:
        rag_tool = create_rag_tool(mode=rag_mode)
        if rag_tool:
            tools.append(rag_tool)
            logger.info("知识库已加载 (RAG: %s)", rag_mode)
        else:
            logger.info("知识库未建立，跳过 RAG 工具")
    else:
        logger.info("RAG 已关闭")
    return tools


def _handle_cli_command(user_input: str, thread_id: str) -> tuple[str | None, str]:
    """处理 CLI 命令，返回 (response_text_or_None, updated_thread_id)。"""
    cmd = user_input.strip()

    if cmd.startswith("/thread ") or cmd.startswith("/switch "):
        new_id = cmd.split(" ", 1)[1].strip()
        if new_id:
            print(f"[已切换到会话: {new_id}]")
            return ("", new_id)
        return ("", thread_id)

    if cmd == "/list":
        sessions = list_sessions()
        if not sessions:
            print("[暂无历史会话]")
        else:
            print(f"[共 {len(sessions)} 个会话]")
            for s in sessions:
                marker = " <-- 当前" if s["thread_id"] == thread_id else ""
                print(f"  - {s['thread_id']}{marker}")
        return ("", thread_id)

    if cmd.startswith("/rename "):
        new_name = cmd.split(" ", 1)[1].strip()
        if new_name:
            if rename_session(thread_id, new_name):
                print(f"[已重命名: {thread_id} -> {new_name}]")
                return ("", new_name)
            else:
                print("[重命名失败]")
        return ("", thread_id)

    if cmd.startswith("/delete "):
        target = cmd.split(" ", 1)[1].strip()
        if target:
            if delete_session(target):
                print(f"[已删除会话: {target}]")
                if target == thread_id:
                    new_id = str(uuid.uuid4())[:8]
                    print(f"[已切换到新会话: {new_id}]")
                    return ("", new_id)
            else:
                print("[删除失败]")
        return ("", thread_id)

    return (None, thread_id)


def main():
    parser = argparse.ArgumentParser(description="MyAgent 智能体 CLI")
    parser.add_argument(
        "--classic", action="store_true",
        help="使用原来的 create_agent 封装（默认使用手动 StateGraph）",
    )
    parser.add_argument(
        "--multi", action="store_true",
        help="多智能体模式：Supervisor + Workers (Code/Data/Writer)",
    )
    parser.add_argument(
        "--rag", type=str, choices=["classic", "advanced"], default=None,
        help="RAG检索模式：classic (仅FAISS) 或 advanced (混合检索)",
    )
    parser.add_argument(
        "--no-rag", action="store_true",
        help="关闭 RAG 知识库检索",
    )
    parser.add_argument(
        "--no-stream", action="store_true",
        help="关闭流式输出，等待回答完全生成后再一次性打印",
    )
    args = parser.parse_args()

    if args.multi and args.classic:
        print("[错误] --multi 和 --classic 不能同时使用")
        return

    use_stream = not args.no_stream
    enable_rag = (not args.no_rag) and ENABLE_RAG
    rag_mode = args.rag or RAG_MODE

    if args.multi:
        mode_name = "Multi-Agent (Supervisor + Workers)"
    elif args.classic:
        mode_name = "Classic (create_agent)"
    else:
        mode_name = "StateGraph (手动构建)"

    print("=" * 50)
    print("  MyAgent - 智能体")
    print(f"  模式: {mode_name}")
    if args.multi:
        print("  Workers: Code / Data / Writer")
    print(f"  输出: {'流式 (Streaming)' if use_stream else '整块 (Blocking)'}")
    print(f"  上下文窗口: {MAX_MESSAGES} 条 | 工具上限: {MAX_TOOL_ITERATIONS} 轮")
    print(f"  RAG: {'启用 (' + rag_mode + ')' if enable_rag else '关闭'}")
    print("  输入 'quit' 退出 | 'clear' 清空对话")
    if not args.multi:
        print("  /list  /switch <id>  /rename <name>  /delete <id>")
    print("=" * 50)

    tools = _assemble_tools(enable_rag, rag_mode)
    print("  [记忆模块已启用 (SQLite)]")

    counter = TokenCounter()
    cb = UsageCallback(counter)
    llm = get_llm(callbacks=[cb], for_tools=True)

    thread_id = "default"
    if not args.multi:
        print(f"  [当前会话 ID: {thread_id}]")

    # ================== 多智能体模式 ==================
    if args.multi:
        from agent.multi import build_multi_agent_graph

        print("\n[Multi-Agent 系统就绪]")
        graph = build_multi_agent_graph()

        while True:
            try:
                user_input = input("You: ").strip()
            except (EOFError, KeyboardInterrupt):
                print("\n再见!")
                break

            if not user_input:
                continue
            if user_input.lower() in ("quit", "exit"):
                print("再见!")
                break
            if user_input.lower() == "clear":
                print("[对话已清空]")
                continue

            from agent.multi.state import create_initial_state
            initial_state = create_initial_state(user_input)

            try:
                result = graph.invoke(initial_state)
                messages = result.get("messages", [])
                if messages:
                    final_message = messages[-1]
                    content = final_message.content if hasattr(final_message, "content") else str(final_message)
                    print(f"\nAgent:\n{content}")
                else:
                    print("\n[处理完成，但没有生成回答]")
            except Exception as e:
                print(f"\n[错误]: {e}")
                import traceback
                traceback.print_exc()

            print()

        return

    # ================== 单智能体模式 ==================
    async def process_chat():
        nonlocal thread_id

        async with get_checkpointer_context() as memory:
            if args.classic:
                agent = _build_classic_agent(llm, tools, memory)
            else:
                agent = _build_graph_agent(llm, tools, memory)

            while True:
                try:
                    user_input = input(f"\nYou ({thread_id}): ").strip()
                except (EOFError, KeyboardInterrupt):
                    print("\n再见!")
                    break

                if not user_input:
                    continue
                if user_input.lower() in ("quit", "exit"):
                    print("再见!")
                    break

                if user_input.startswith("/"):
                    resp, thread_id = _handle_cli_command(user_input, thread_id)
                    if resp is not None:
                        continue

                if user_input.lower() == "clear":
                    thread_id = str(uuid.uuid4())[:8]
                    print(f"[对话已清空 - 新会话 ID: {thread_id}]")
                    continue

                config = {"configurable": {"thread_id": thread_id}}
                print(f"\nAgent: ", end="", flush=True)

                try:
                    if use_stream:
                        async for event in agent.astream_events(
                            {"messages": [HumanMessage(content=user_input)]},
                            config=config,
                            version="v2"
                        ):
                            kind = event.get("event", "")

                            if kind == "on_chat_model_stream":
                                chunk = event["data"].get("chunk")
                                if chunk and hasattr(chunk, "content") and chunk.content:
                                    content = chunk.content
                                    if isinstance(content, str):
                                        print(content, end="", flush=True)
                                    elif isinstance(content, list):
                                        for item in content:
                                            if isinstance(item, dict) and item.get("type") == "text":
                                                print(item.get("text", ""), end="", flush=True)
                                            elif isinstance(item, str):
                                                print(item, end="", flush=True)

                            elif kind == "on_tool_start":
                                tool_name = event.get("name", "unknown")
                                tool_input = event.get("data", {}).get("input", {})
                                print(f"\n  [调用工具: {tool_name}, 参数: {tool_input}]", flush=True)
                                print("Agent: ", end="", flush=True)

                        print()
                    else:
                        result = await agent.ainvoke(
                            {"messages": [HumanMessage(content=user_input)]},
                            config=config,
                        )
                        ai_message = result["messages"][-1]
                        print(ai_message.content)

                        print(
                            f"\n[usage] calls={counter.calls} "
                            f"prompt={counter.prompt_tokens} "
                            f"completion={counter.completion_tokens} "
                            f"total={counter.total_tokens}"
                        )
                except Exception as e:
                    print(f"\n[错误]: {e}")

    import asyncio
    asyncio.run(process_chat())


if __name__ == "__main__":
    main()
