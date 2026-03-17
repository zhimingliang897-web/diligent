"""Data Agent — 数据分析专家

职责:
- 数据分析与统计
- 知识库检索（复用 RAG）
- 数据洞察与总结
- 图表分析与描述
"""

from langchain_core.tools import tool

from agent.multi.workers.base import BaseWorker
from agent.tools import calculate, web_search, get_weather


# ──────────────── Data Agent 专用工具 ────────────────

@tool
def analyze_numbers(data: str) -> str:
    """分析一组数字数据，计算基本统计指标。

    Args:
        data: 逗号分隔的数字字符串，如 "1, 2, 3, 4, 5"

    Returns:
        包含均值、中位数、最大值、最小值、总和等统计信息
    """
    try:
        numbers = [float(x.strip()) for x in data.split(",")]
        if not numbers:
            return "没有有效的数字数据"

        n = len(numbers)
        total = sum(numbers)
        mean = total / n
        sorted_nums = sorted(numbers)

        # 中位数
        if n % 2 == 0:
            median = (sorted_nums[n // 2 - 1] + sorted_nums[n // 2]) / 2
        else:
            median = sorted_nums[n // 2]

        # 方差和标准差
        variance = sum((x - mean) ** 2 for x in numbers) / n
        std_dev = variance ** 0.5

        return f"""数据分析结果:
- 数据个数: {n}
- 总和: {total:.2f}
- 均值: {mean:.2f}
- 中位数: {median:.2f}
- 最大值: {max(numbers):.2f}
- 最小值: {min(numbers):.2f}
- 标准差: {std_dev:.2f}
- 方差: {variance:.2f}"""
    except Exception as e:
        return f"数据分析错误: {e}"


@tool
def describe_trend(data: str) -> str:
    """分析数据的趋势特征。

    Args:
        data: 逗号分隔的数字字符串，按时间顺序排列

    Returns:
        趋势分析结果（上升/下降/平稳/波动）
    """
    try:
        numbers = [float(x.strip()) for x in data.split(",")]
        if len(numbers) < 2:
            return "数据点不足，无法分析趋势"

        # 计算变化
        changes = [numbers[i + 1] - numbers[i] for i in range(len(numbers) - 1)]
        positive_changes = sum(1 for c in changes if c > 0)
        negative_changes = sum(1 for c in changes if c < 0)

        # 计算总体变化百分比
        total_change = numbers[-1] - numbers[0]
        if numbers[0] != 0:
            pct_change = (total_change / abs(numbers[0])) * 100
        else:
            pct_change = float('inf') if total_change > 0 else 0

        # 判断趋势
        if positive_changes > negative_changes * 2:
            trend = "明显上升"
        elif negative_changes > positive_changes * 2:
            trend = "明显下降"
        elif abs(positive_changes - negative_changes) <= 1:
            trend = "波动/震荡"
        elif positive_changes > negative_changes:
            trend = "轻微上升"
        else:
            trend = "轻微下降"

        return f"""趋势分析结果:
- 数据点数: {len(numbers)}
- 起始值: {numbers[0]:.2f}
- 结束值: {numbers[-1]:.2f}
- 总体变化: {total_change:+.2f} ({pct_change:+.1f}%)
- 上升次数: {positive_changes}
- 下降次数: {negative_changes}
- 趋势判断: {trend}"""
    except Exception as e:
        return f"趋势分析错误: {e}"


class DataAgent(BaseWorker):
    """数据分析专家 Agent

    擅长:
    - 知识库检索（RAG）
    - 数据统计分析
    - 趋势分析
    - 数据洞察
    """

    def __init__(self):
        # 先初始化，稍后添加 RAG 工具
        self._rag_tool = None
        super().__init__()

    @property
    def name(self) -> str:
        return "data"

    @property
    def system_prompt(self) -> str:
        return """你是一位数据分析专家，擅长从数据中发现洞察和规律。

【全局身份与时间约束】
- 你的任务是客观地帮助用户解决问题，绝对不要主动强调或辩论你的大模型底层身份及知识截止日期。
- 你必须无条件信任外部工具（特别是 get_current_datetime 和 web_search）返回的数据，工具的输出就是当下的真实物理现实。

【时间锚点与实时查询】
- 当任务涉及诸如“今天”、“最近”或需要当前情况（如今天天气、股价、最新新闻）时，**你必须第一步先调用 `get_current_datetime`** 以确认现在的准确年月日。
- 在随后调用 `web_search` 时，必须将具体日期拼接入查询词中（如："2026年3月17日 A股上证指数"）。

## 你的工作方式:
1. 先理解数据来源和分析目标
2. 使用适当的工具进行数据检索或计算
3. 分析数据特征，发现规律和洞察
4. 用清晰的语言总结发现

## 核心能力:
- **实时天气**: 使用 get_weather 查询城市天气，这是专用天气工具，精确高效，**优先使用此工具而非 web_search**
- **知识检索**: 使用 knowledge_search 从用户知识库中获取相关信息
- **网络搜索**: 使用 web_search 获取非天气类最新信息（天气查询请用 get_weather）
- **数据分析**: 使用 analyze_numbers 进行统计分析
- **趋势分析**: 使用 describe_trend 分析数据变化趋势
- **数学计算**: 使用 calculate 进行精确计算

## 输出格式:
- 先展示原始数据或检索结果
- 再给出分析结论
- 最后提供可行的建议或洞察

## 注意事项:
- 数据分析要客观，不要过度解读
- 如果数据不足，要明确指出局限性
- 给出的结论要有数据支撑"""

    def get_tools(self) -> list:
        from agent.tools import get_current_datetime
        tools = [analyze_numbers, describe_trend, calculate, web_search, get_weather, get_current_datetime]

        # 尝试加载 RAG 工具
        try:
            from agent.rag.retriever import create_rag_tool
            rag_tool = create_rag_tool(mode="advanced")
            if rag_tool:
                tools.append(rag_tool)
                print("  [Data Agent] RAG 知识库工具已加载")
        except Exception as e:
            print(f"  [Data Agent] RAG 工具加载失败: {e}")

        return tools
