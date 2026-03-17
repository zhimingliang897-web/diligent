"""多智能体模块 — Supervisor + Workers 架构

提供基于 LangGraph 的多智能体协作系统:
- Supervisor: 意图识别 + 任务分解 + 执行编排
- Workers: Code Agent / Data Agent / Writer Agent
"""

from agent.multi.graph import build_multi_agent_graph
from agent.multi.state import MultiAgentState

__all__ = ["build_multi_agent_graph", "MultiAgentState"]
