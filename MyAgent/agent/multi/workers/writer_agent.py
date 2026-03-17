"""Writer Agent — 写作专家

职责:
- 文章撰写与内容创作
- 内容润色与优化
- 摘要与总结
- 格式化与排版
"""

from langchain_core.tools import tool

from agent.multi.workers.base import BaseWorker


# ──────────────── Writer Agent 专用工具 ────────────────

@tool
def generate_outline(topic: str, style: str = "技术博客") -> str:
    """为给定主题生成文章大纲框架。

    Args:
        topic: 文章主题
        style: 文章风格，如 "技术博客"、"报告"、"教程"

    Returns:
        结构化的文章大纲
    """
    # 基础大纲模板
    templates = {
        "技术博客": """# {topic}

## 1. 引言
- 背景介绍
- 问题引出
- 文章目标

## 2. 核心内容
- 概念解释
- 原理分析
- 实现方法

## 3. 实践案例
- 示例代码/操作步骤
- 效果展示

## 4. 总结
- 关键要点回顾
- 延伸阅读/下一步""",
        "报告": """# {topic} 分析报告

## 摘要

## 1. 背景与目的

## 2. 数据/信息来源

## 3. 分析过程
### 3.1 方法说明
### 3.2 详细分析

## 4. 结论与发现

## 5. 建议与下一步

## 附录""",
        "教程": """# {topic} 完全指南

## 前置要求
- 知识储备
- 环境准备

## 第一步: 基础入门

## 第二步: 核心操作

## 第三步: 进阶技巧

## 常见问题 FAQ

## 总结与资源"""
    }

    template = templates.get(style, templates["技术博客"])
    return template.format(topic=topic)


@tool
def word_count(text: str) -> str:
    """统计文本的字数和段落数。

    Args:
        text: 要统计的文本

    Returns:
        字数统计结果
    """
    # 中文字符数
    chinese_chars = sum(1 for c in text if '\u4e00' <= c <= '\u9fff')
    # 英文单词数（简单分词）
    english_words = len([w for w in text.split() if w.isascii() and w.isalpha()])
    # 段落数（按空行分割）
    paragraphs = len([p for p in text.split('\n\n') if p.strip()])
    # 总字符数
    total_chars = len(text)

    return f"""文本统计:
- 中文字符: {chinese_chars}
- 英文单词: {english_words}
- 段落数: {paragraphs}
- 总字符数: {total_chars}"""


@tool
def format_as_markdown(content: str, title: str = "") -> str:
    """将内容格式化为规范的 Markdown 格式。

    Args:
        content: 原始内容
        title: 文章标题（可选）

    Returns:
        格式化后的 Markdown 文本
    """
    result = []
    if title:
        result.append(f"# {title}\n")

    # 简单的格式化处理
    lines = content.split('\n')
    for line in lines:
        line = line.strip()
        if not line:
            result.append("")
            continue

        # 识别列表项
        if line.startswith(('-', '*', '•')):
            result.append(f"- {line[1:].strip()}")
        # 识别数字列表
        elif line[0].isdigit() and (line[1] == '.' or (len(line) > 2 and line[1:2].isdigit() and line[2] == '.')):
            result.append(line)
        else:
            result.append(line)

    return '\n'.join(result)


class WriterAgent(BaseWorker):
    """写作专家 Agent

    擅长:
    - 文章撰写
    - 内容润色
    - 摘要生成
    - 格式排版
    """

    @property
    def name(self) -> str:
        return "writer"

    @property
    def system_prompt(self) -> str:
        return """你是一位专业的技术写作者，擅长把复杂内容写得清晰易懂。

## 你的工作方式:
1. 先理解写作目标和受众
2. 使用 generate_outline 工具构建文章框架
3. 按照大纲逐节撰写内容
4. 最后使用 format_as_markdown 确保格式规范

## 写作风格:
- **清晰**: 用简洁的语言表达复杂概念
- **结构化**: 使用标题、列表、代码块等元素增强可读性
- **有价值**: 每段内容都应该给读者带来收获
- **适度技术性**: 根据受众调整专业术语的使用

## 输出要求:
- 使用规范的 Markdown 格式
- 代码示例用 ```language 包裹
- 适当使用粗体、斜体强调关键点
- 保持段落简短，一个段落一个核心观点

## 特殊能力:
- 可以将分析报告转化为易读的文章
- 可以将技术文档改写为教程
- 可以生成内容摘要

## 注意事项:
- 如果收到上游 Agent 的数据分析结果，要将其融入文章中
- 保持客观，数据说话
- 适当使用图表描述（用文字描述图表内容）"""

    def get_tools(self) -> list:
        return [generate_outline, word_count, format_as_markdown]
