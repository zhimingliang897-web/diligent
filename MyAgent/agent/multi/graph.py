"""多智能体 StateGraph — Supervisor + Workers 架构 (支持 Re-plan)

图结构:
    START → supervisor → route_to_worker
                          ├─ "code" → code_agent → increment → supervisor (循环)
                          ├─ "data" → data_agent → increment → supervisor (循环)
                          ├─ "writer" → writer_agent → increment → supervisor (循环)
                          └─ "FINISH" → aggregator → END

工作流程:
1. Supervisor 分析用户请求，创建任务计划
2. 根据任务计划，依次路由到对应的 Worker
3. Worker 完成任务后返回 Supervisor
4. Supervisor 检查是否还有待执行任务
5. 如果 Worker 失败，触发 Re-plan 重新规划
6. 所有任务完成后，Aggregator 汇总结果
"""

from langgraph.graph import StateGraph, END

from agent.multi.state import MultiAgentState
from agent.multi.supervisor import create_supervisor_node, create_aggregator_node
from agent.multi.workers import CodeAgent, DataAgent, WriterAgent


# 最大迭代次数（防止无限循环）
MAX_ITERATIONS = 10


def build_multi_agent_graph(checkpointer=None):
    """构建多智能体 StateGraph

    Args:
        checkpointer: 可选的检查点器（用于持久化状态）

    Returns:
        编译后的 CompiledGraph
    """

    # 创建 Workers
    code_agent = CodeAgent()
    data_agent = DataAgent()
    writer_agent = WriterAgent()

    # 创建节点函数
    supervisor_node = create_supervisor_node()
    aggregator_node = create_aggregator_node()

    code_node = code_agent.create_node()
    data_node = data_agent.create_node()
    writer_node = writer_agent.create_node()

    # ──────────────── 辅助节点 ────────────────

    def increment_counter(state: MultiAgentState) -> dict:
        """递增迭代计数器"""
        count = state.get("iteration_count", 0) + 1
        if count >= MAX_ITERATIONS:
            print(f"  [警告] 迭代次数已达上限 ({MAX_ITERATIONS})，强制结束", flush=True)
        return {"iteration_count": count}

    # ──────────────── 路由函数 ────────────────

    def route_to_worker(state: MultiAgentState) -> str:
        """根据 current_worker 路由到对应的 Worker"""
        # 检查迭代次数
        if state.get("iteration_count", 0) >= MAX_ITERATIONS:
            return "aggregator"

        current = state.get("current_worker", "FINISH")

        if current == "code":
            return "code_agent"
        elif current == "data":
            return "data_agent"
        elif current == "writer":
            return "writer_agent"
        elif current == "FINISH":
            return "aggregator"
        else:
            # 未知类型，结束
            print(f"  [警告] 未知的 worker 类型: {current}", flush=True)
            return "aggregator"

    # ──────────────── 构建图 ────────────────

    graph = StateGraph(MultiAgentState)

    # 添加节点
    graph.add_node("supervisor", supervisor_node)
    graph.add_node("code_agent", code_node)
    graph.add_node("data_agent", data_node)
    graph.add_node("writer_agent", writer_node)
    graph.add_node("aggregator", aggregator_node)
    graph.add_node("increment", increment_counter)

    # 设置入口点
    graph.set_entry_point("supervisor")

    # Supervisor → 路由到 Worker 或 Aggregator
    graph.add_conditional_edges(
        "supervisor",
        route_to_worker,
        {
            "code_agent": "code_agent",
            "data_agent": "data_agent",
            "writer_agent": "writer_agent",
            "aggregator": "aggregator",
        },
    )

    # Workers → increment → supervisor (循环)
    graph.add_edge("code_agent", "increment")
    graph.add_edge("data_agent", "increment")
    graph.add_edge("writer_agent", "increment")
    graph.add_edge("increment", "supervisor")

    # Aggregator → END
    graph.add_edge("aggregator", END)

    # 编译
    if checkpointer:
        return graph.compile(checkpointer=checkpointer)
    return graph.compile()


def run_multi_agent(query: str, checkpointer=None) -> str:
    """运行多智能体系统处理用户请求

    Args:
        query: 用户输入
        checkpointer: 可选的检查点器

    Returns:
        最终回答
    """
    from agent.multi.state import create_initial_state

    graph = build_multi_agent_graph(checkpointer)

    # 使用新的初始状态（包含 replan 字段）
    initial_state = create_initial_state(query)

    # 执行图
    print("\n" + "=" * 50)
    print(f"[Multi-Agent] 处理请求: {query}")
    print("=" * 50)

    result = graph.invoke(initial_state)

    # 提取最终回答
    messages = result.get("messages", [])
    if messages:
        final_message = messages[-1]
        return final_message.content if hasattr(final_message, "content") else str(final_message)

    return "处理完成，但没有生成回答。"
