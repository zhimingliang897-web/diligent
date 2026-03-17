"""手动构建 LangGraph StateGraph — 替代 create_agent 的自定义 ReAct 循环

图结构:
    START → trim → rewrite → agent → should_continue
                                        ├─ "tools" → tools → check_iterations
                                        │                       ├─ "continue" → agent (循环)
                                        │                       └─ "limit" → force_reply (强制结束)
                                        └─ END (直接回复)

增强功能:
  1. 查询改写 (rewrite_node) — 首轮对话时改写用户问题，提高 RAG 命中率
  2. 工具调用上限 (max_iterations) — 防止 LLM 陷入无限调工具循环
"""

import logging
from typing import Annotated, TypedDict

from langchain_core.messages import SystemMessage, HumanMessage, AIMessage, ToolMessage
from langgraph.graph import StateGraph, END
from langgraph.graph.message import add_messages
from langgraph.prebuilt import ToolNode

from agent.config import MAX_MESSAGES as _DEFAULT_MAX_MESSAGES
from agent.config import MAX_TOOL_ITERATIONS as _DEFAULT_MAX_ITERATIONS

logger = logging.getLogger(__name__)


class AgentState(TypedDict):
    """Agent 状态，在 MessagesState 基础上增加迭代计数。"""
    messages: Annotated[list, add_messages]
    iteration_count: int


def build_agent(
    llm,
    tools,
    checkpointer,
    system_prompt: str,
    max_messages: int = _DEFAULT_MAX_MESSAGES,
    max_iterations: int = _DEFAULT_MAX_ITERATIONS,
):
    """构建手动 StateGraph Agent。

    Args:
        llm: ChatTongyi 实例（已带 callbacks）
        tools: 工具列表（@tool 装饰的函数）
        checkpointer: SqliteSaver 检查点器
        system_prompt: 系统提示词
        max_messages: 消息窗口大小（保留最近 N 条非系统消息）
        max_iterations: 工具调用最大轮数（防止无限循环）

    Returns:
        编译后的 CompiledGraph
    """

    llm_with_tools = llm.bind_tools(tools)

    REWRITE_PROMPT = (
        "你是一个专门为搜索引擎（包含关键词匹配和向量匹配）准备查询语句的助手。\n"
        "请将用户的原始问题改写成更精确、更详细、且包含核心关键词的形式。\n"
        "规则：\n"
        "- 只输出改写后的句子，不要解释\n"
        "- 补全所有的代词（他/这/那个），结合上下文还原所指对象\n"
        "- 适当扩充同义词，或把口语化的词语转换为书面的专业术语关键词（这对命中搜索引擎极度重要）\n"
        "- 保持用户的语言（如中文）\n"
    )

    # ──────────────── 公共辅助 ────────────────

    def _inject_system(messages):
        """注入系统提示词 + 动态长期记忆。"""
        from agent.memory.profile import get_profile_summary
        full = system_prompt + "\n\n" + get_profile_summary()
        if not messages or not isinstance(messages[0], SystemMessage):
            return [SystemMessage(content=full)] + messages
        return messages

    # ──────────────── 节点定义 ────────────────

    def trim_node(state: AgentState) -> dict:
        """裁剪消息历史 + 重置迭代计数。"""
        messages = state["messages"]

        system_msgs = [m for m in messages if isinstance(m, SystemMessage)]
        other_msgs = [m for m in messages if not isinstance(m, SystemMessage)]

        sanitized_msgs = []
        for m in other_msgs:
            if isinstance(m, ToolMessage):
                continue
            if isinstance(m, AIMessage) and getattr(m, "tool_calls", None):
                content = m.content if isinstance(m.content, str) else ""
                if not content.strip():
                    content = "（上一轮已执行工具并得到结果）"
                sanitized_msgs.append(AIMessage(content=content))
            else:
                sanitized_msgs.append(m)
        other_msgs = sanitized_msgs

        if len(other_msgs) > max_messages:
            other_msgs = other_msgs[-max_messages:]

        return {"messages": system_msgs + other_msgs, "iteration_count": 0}

    def rewrite_node(state: AgentState) -> dict:
        """查询改写：对用户最新问题进行改写，提高检索命中率。"""
        messages = state["messages"]
        if not messages:
            return {}

        last_human = None
        for m in reversed(messages):
            if isinstance(m, HumanMessage):
                last_human = m
                break

        if not last_human:
            return {}

        original = last_human.content.strip()

        needs_rewrite = len(original) < 6 or any(
            w in original for w in ["那个", "这个", "它", "他", "她", "这", "那", "上面", "之前"]
        )

        if not needs_rewrite:
            return {}

        rewrite_messages = [
            SystemMessage(content=REWRITE_PROMPT),
            HumanMessage(content=f"原始问题：{original}"),
        ]
        rewritten = llm.invoke(rewrite_messages)
        new_query = rewritten.content.strip()

        if new_query and new_query != original:
            logger.info("[查询改写] %s -> %s", original, new_query)
            new_messages = []
            replaced = False
            for m in reversed(messages):
                if isinstance(m, HumanMessage) and not replaced:
                    new_messages.append(HumanMessage(content=new_query))
                    replaced = True
                else:
                    new_messages.append(m)
            new_messages.reverse()
            return {"messages": new_messages}

        return {}

    def agent_node(state: AgentState) -> dict:
        """调用 LLM：注入系统提示词和动态长期记忆。"""
        messages = _inject_system(state["messages"])
        response = llm_with_tools.invoke(messages)
        return {"messages": [response]}

    tool_node = ToolNode(tools)

    def force_reply(state: AgentState) -> dict:
        """工具调用超过上限时，强制 LLM 给出最终回答。"""
        messages = _inject_system(state["messages"])
        force_msg = SystemMessage(
            content=(
                f"你已经调用了 {max_iterations} 轮工具，禁止继续调用工具。\n"
                "请只基于当前消息里已经存在的工具输出作答，并严格遵守：\n"
                "1) 不得虚构任何检索过程、数据来源、官网访问或多轮验证；\n"
                "2) 不得猜测具体数值（尤其是实时行情/收盘价）；\n"
                "3) 若现有工具输出不足以回答，直接明确说明目前无法从已获取工具结果确认，并给出下一步可执行建议。"
            )
        )
        response = llm.invoke(messages + [force_msg])
        return {"messages": [response]}

    # ──────────────── 条件路由 ────────────────

    def should_continue(state: AgentState) -> str:
        last_message = state["messages"][-1]
        if hasattr(last_message, "tool_calls") and last_message.tool_calls:
            return "tools"
        return END

    def check_iterations(state: AgentState) -> str:
        count = state.get("iteration_count", 0) + 1
        if count >= max_iterations:
            logger.warning("工具调用已达上限 (%d 轮)，强制结束", max_iterations)
            return "limit"
        return "continue"

    def increment_counter(state: AgentState) -> dict:
        return {"iteration_count": state.get("iteration_count", 0) + 1}

    # ──────────────── 构建图 ────────────────

    graph = StateGraph(AgentState)

    graph.add_node("trim", trim_node)
    graph.add_node("rewrite", rewrite_node)
    graph.add_node("agent", agent_node)
    graph.add_node("tools", tool_node)
    graph.add_node("increment", increment_counter)
    graph.add_node("force_reply", force_reply)

    graph.set_entry_point("trim")
    graph.add_edge("trim", "rewrite")
    graph.add_edge("rewrite", "agent")
    graph.add_conditional_edges(
        "agent",
        should_continue,
        {"tools": "tools", END: END},
    )
    graph.add_edge("tools", "increment")
    graph.add_conditional_edges(
        "increment",
        check_iterations,
        {"continue": "agent", "limit": "force_reply"},
    )
    graph.add_edge("force_reply", END)

    return graph.compile(checkpointer=checkpointer)
