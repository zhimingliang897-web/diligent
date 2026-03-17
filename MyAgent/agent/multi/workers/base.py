"""Worker 基类 — 所有专业 Agent 的基础

BaseWorker 提供:
- 统一的工具绑定和 LLM 调用接口
- 标准化的节点执行逻辑
- 工具调用的 ReAct 循环
- 失败检测和 Re-plan 触发
"""

from abc import ABC, abstractmethod
from typing import Any

from langchain_core.messages import SystemMessage, AIMessage, HumanMessage
from langgraph.graph import StateGraph, END
from langgraph.prebuilt import ToolNode

from agent.llm import get_llm
from agent.multi.state import MultiAgentState
from agent.multi import event_bus


# 失败关键词列表（用于检测任务是否失败）
FAILURE_KEYWORDS = [
    "无法完成", "无法处理", "不能完成", "做不到",
    "缺少信息", "信息不足", "需要更多",
    "抱歉", "sorry", "cannot", "unable",
    "错误", "失败", "error", "failed",
    "没有找到", "找不到", "not found",
]


class BaseWorker(ABC):
    """Worker Agent 基类

    子类需要实现:
    - name: Worker 名称
    - system_prompt: 系统提示词
    - tools: 工具列表
    """

    def __init__(self):
        self.llm = get_llm(for_tools=True)
        self._tools = self.get_tools()
        self.llm_with_tools = self.llm.bind_tools(self._tools) if self._tools else self.llm

    @property
    @abstractmethod
    def name(self) -> str:
        """Worker 名称，用于路由标识"""
        pass

    @property
    @abstractmethod
    def system_prompt(self) -> str:
        """系统提示词，定义 Worker 的角色和行为"""
        pass

    @abstractmethod
    def get_tools(self) -> list:
        """返回该 Worker 可用的工具列表"""
        pass

    def _check_task_success(self, result: str, error: str = None) -> tuple[bool, str]:
        """检查任务是否成功完成

        Returns:
            (is_success, failure_reason)
        """
        # 如果有明确的错误
        if error:
            return False, error

        # 如果结果为空
        if not result or not result.strip():
            return False, "任务返回空结果"

        # 检查是否包含失败关键词
        result_lower = result.lower()
        for keyword in FAILURE_KEYWORDS:
            if keyword.lower() in result_lower:
                # 避免误判：如果结果很长且只是提到了错误处理，可能不是失败
                if len(result) > 200:
                    continue
                return False, f"任务结果包含失败指示: '{keyword}'"

        return True, ""

    def create_node(self):
        """创建该 Worker 的执行节点函数"""

        def worker_node(state: MultiAgentState) -> dict:
            """Worker 执行节点"""
            messages = state["messages"]
            handoff_context = state.get("handoff_context", "")

            # 获取当前任务
            current_task = None
            task_plan = state.get("task_plan", [])
            for task in task_plan:
                if task.get("assigned_to") == self.name and task.get("status") == "in_progress":
                    current_task = task
                    break

            # 构建工作上下文
            task_context = ""
            if current_task:
                task_context = f"\n\n当前任务: {current_task.get('description', '')}"

            # 构建完整的系统提示词
            full_prompt = self.system_prompt
            if handoff_context:
                full_prompt += f"\n\n## 上游 Agent 传递的上下文:\n{handoff_context}"
            if task_context:
                full_prompt += task_context

            # 构建消息列表
            work_messages = [SystemMessage(content=full_prompt)]

            # 添加用户原始问题
            original_query = state.get("original_query", "")
            if original_query:
                work_messages.append(HumanMessage(content=original_query))

            # 添加历史消息中的关键信息（最近 5 条）
            recent_msgs = [m for m in messages if not isinstance(m, SystemMessage)][-5:]
            work_messages.extend(recent_msgs)

            # 调用 LLM
            print(f"  [{self.name.upper()} Agent] 开始处理...", flush=True)
            event_bus.emit(f"🔄 [{self.name.upper()} Agent] 开始处理任务: {current_task.get('description', '')[:40] if current_task else ''}")

            result = ""
            error = ""
            needs_replan = False
            replan_reason = ""

            try:
                response = self.llm_with_tools.invoke(work_messages)

                # 如果有工具调用，执行工具
                if hasattr(response, "tool_calls") and response.tool_calls:
                    tool_names = [tc['name'] for tc in response.tool_calls]
                    print(f"  [{self.name.upper()} Agent] 调用工具: {tool_names}", flush=True)
                    event_bus.emit(f"🔧 [{self.name.upper()} Agent] 调用工具: {', '.join(tool_names)}")
                    tool_node = ToolNode(self._tools)

                    try:
                        # 执行工具并获取结果
                        tool_result = tool_node.invoke({"messages": [response]})
                        tool_messages = tool_result.get("messages", [])
                        event_bus.emit(f"✅ [{self.name.upper()} Agent] 工具执行完成")

                        # 将工具结果传回 LLM 获取最终回答。
                        # 注意：不直接回放含 tool_calls 的 AIMessage，避免部分模型端
                        # 因 function.arguments 非 JSON 字符串而报 400。
                        interim_text = response.content if isinstance(response.content, str) else ""
                        if not interim_text.strip():
                            interim_text = "已调用工具，请基于工具输出给出最终回答。"
                        final_messages = work_messages + [AIMessage(content=interim_text)] + tool_messages
                        response = self.llm.invoke(final_messages)
                    except Exception as tool_error:
                        error = f"工具调用失败: {str(tool_error)}"
                        print(f"  [{self.name.upper()} Agent] 工具调用失败: {tool_error}", flush=True)
                        event_bus.emit(f"❌ [{self.name.upper()} Agent] 工具调用失败: {tool_error}")

                result = response.content if hasattr(response, "content") else str(response)

            except Exception as e:
                error = f"LLM 调用失败: {str(e)}"
                print(f"  [{self.name.upper()} Agent] 执行失败: {e}", flush=True)

            # 检查任务是否成功
            is_success, failure_reason = self._check_task_success(result, error)

            if is_success:
                print(f"  [{self.name.upper()} Agent] 完成 ✓", flush=True)
                event_bus.emit(f"✅ [{self.name.upper()} Agent] 任务完成")
                task_status = "completed"
            else:
                print(f"  [{self.name.upper()} Agent] 失败: {failure_reason}", flush=True)
                event_bus.emit(f"❌ [{self.name.upper()} Agent] 任务失败: {failure_reason}")
                task_status = "failed"
                needs_replan = True
                replan_reason = f"[{self.name}] {failure_reason}"

            # 更新任务状态
            updated_plan = []
            for task in task_plan:
                if task.get("assigned_to") == self.name and task.get("status") == "in_progress":
                    task = {
                        **task,
                        "status": task_status,
                        "result": result,
                        "error": error if error else ""
                    }
                updated_plan.append(task)

            # 更新 worker_results
            worker_results = state.get("worker_results", {}).copy()
            worker_results[self.name] = result

            return {
                "messages": [AIMessage(content=f"[{self.name}]: {result}")],
                "task_plan": updated_plan,
                "worker_results": worker_results,
                "handoff_context": result,  # 传递给下一个 Worker
                "needs_replan": needs_replan,
                "replan_reason": replan_reason,
            }

        return worker_node
