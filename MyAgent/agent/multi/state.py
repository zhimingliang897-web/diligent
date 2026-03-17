"""多智能体状态定义

MultiAgentState 扩展了基础 AgentState，增加了:
- task_plan: 任务分解计划
- current_worker: 当前激活的 Worker
- worker_results: 各 Worker 的执行结果
- handoff_context: Worker 间传递的上下文信息
- needs_replan: 是否需要重新规划（Re-plan 支持）
"""

from typing import Annotated, TypedDict, Literal
from langgraph.graph.message import add_messages


# Worker 类型枚举
WorkerType = Literal["code", "data", "writer", "supervisor", "FINISH"]

# 任务状态枚举
TaskStatus = Literal["pending", "in_progress", "completed", "failed"]


class TaskItem(TypedDict):
    """单个子任务"""
    id: int                     # 任务编号
    description: str            # 任务描述
    assigned_to: WorkerType     # 分配给哪个 Worker
    status: TaskStatus          # pending / in_progress / completed / failed
    result: str                 # 执行结果
    error: str                  # 错误信息（失败时）


class MultiAgentState(TypedDict):
    """多智能体共享状态

    Attributes:
        messages: 对话消息历史（自动累加）
        iteration_count: 总迭代轮数（防止无限循环）
        task_plan: 任务分解计划列表
        current_worker: 当前激活的 Worker 名称
        worker_results: 各 Worker 的执行结果 {worker_name: result}
        handoff_context: Worker 间传递的上下文信息
        original_query: 用户原始问题（用于最终汇总）
        needs_replan: 是否需要重新规划
        replan_reason: 重新规划的原因
        replan_count: 重新规划次数（防止无限 replan）
    """
    messages: Annotated[list, add_messages]
    iteration_count: int
    task_plan: list[TaskItem]
    current_worker: WorkerType
    worker_results: dict[str, str]
    handoff_context: str
    original_query: str
    needs_replan: bool
    replan_reason: str
    replan_count: int


def create_initial_state(user_message: str) -> dict:
    """创建初始状态"""
    return {
        "messages": [],
        "iteration_count": 0,
        "task_plan": [],
        "current_worker": "supervisor",
        "worker_results": {},
        "handoff_context": "",
        "original_query": user_message,
        "needs_replan": False,
        "replan_reason": "",
        "replan_count": 0,
    }
