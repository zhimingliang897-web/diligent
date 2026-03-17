"""Code Agent — 代码专家

职责:
- 代码生成与编写
- 代码解释与分析
- 代码调试与重构
- 生成可视化代码（matplotlib, seaborn 等）
"""

import ast
import sys
from io import StringIO
from langchain_core.tools import tool

from agent.multi.workers.base import BaseWorker


# ──────────────── Code Agent 专用工具 ────────────────

_BLOCKED_MODULES = frozenset({
    "os", "sys", "subprocess", "shutil", "pathlib", "socket", "http",
    "ftplib", "smtplib", "ctypes", "signal", "multiprocessing", "threading",
    "importlib", "builtins", "code", "codeop", "compileall", "runpy",
})
_BLOCKED_NAMES = frozenset({
    "__import__", "eval", "exec", "compile", "globals", "locals",
    "getattr", "setattr", "delattr", "breakpoint", "exit", "quit",
    "open",
})


def _ast_security_check(code: str) -> str | None:
    """用 AST 解析检查代码安全性，返回 None 表示安全，否则返回拒绝原因。"""
    try:
        tree = ast.parse(code)
    except SyntaxError:
        return None

    for node in ast.walk(tree):
        if isinstance(node, (ast.Import, ast.ImportFrom)):
            names = []
            if isinstance(node, ast.Import):
                names = [alias.name.split(".")[0] for alias in node.names]
            elif node.module:
                names = [node.module.split(".")[0]]
            for n in names:
                if n in _BLOCKED_MODULES:
                    return f"安全限制: 禁止导入模块 '{n}'"

        if isinstance(node, ast.Call):
            func = node.func
            name = None
            if isinstance(func, ast.Name):
                name = func.id
            elif isinstance(func, ast.Attribute):
                name = func.attr
            if name and name in _BLOCKED_NAMES:
                return f"安全限制: 禁止调用 '{name}'"

    return None


@tool
def execute_python(code: str) -> str:
    """在安全沙箱中执行 Python 代码并返回结果。

    Args:
        code: 要执行的 Python 代码

    Returns:
        代码执行的输出结果，包括 print 输出和最后一个表达式的值

    注意: 只支持安全的操作，不能访问文件系统或网络
    """
    violation = _ast_security_check(code)
    if violation:
        return violation

    old_stdout = sys.stdout
    sys.stdout = mystdout = StringIO()

    safe_builtins = {
        k: v for k, v in __builtins__.__dict__.items()
        if k not in _BLOCKED_NAMES
    } if hasattr(__builtins__, '__dict__') else {
        k: v for k, v in __builtins__.items()
        if k not in _BLOCKED_NAMES
    }

    try:
        try:
            tree = ast.parse(code, mode='eval')
            result = eval(compile(tree, '<string>', 'eval'), {"__builtins__": safe_builtins}, {})
            output = mystdout.getvalue()
            if output:
                return f"输出:\n{output}\n返回值: {result}"
            return f"返回值: {result}"
        except SyntaxError:
            pass

        exec(code, {"__builtins__": safe_builtins}, {})
        output = mystdout.getvalue()
        return output if output else "代码执行成功（无输出）"

    except Exception as e:
        return f"执行错误: {type(e).__name__}: {e}"
    finally:
        sys.stdout = old_stdout


@tool
def explain_code(code: str) -> str:
    """分析并解释一段代码的功能和逻辑。

    Args:
        code: 要解释的代码片段

    Returns:
        代码的功能说明，包括主要逻辑、使用的算法等
    """
    try:
        ast.parse(code)
    except SyntaxError as e:
        return f"代码语法错误: {e}"

    try:
        from agent.llm import get_llm
        llm = get_llm(streaming=False)
        prompt = f"请用中文简明扼要地解释以下代码的功能、主要逻辑和使用的算法:\n```python\n{code}\n```"
        msg = llm.invoke(prompt)
        return (msg.content or "").strip() or "未能生成解释。"
    except Exception as e:
        return f"代码解释失败: {e}"


@tool
def check_syntax(code: str) -> str:
    """检查 Python 代码的语法是否正确。

    Args:
        code: 要检查的 Python 代码

    Returns:
        语法检查结果
    """
    try:
        ast.parse(code)
        return "语法检查通过，代码没有语法错误"
    except SyntaxError as e:
        return f"语法错误:\n  行 {e.lineno}: {e.msg}\n  {e.text}"


class CodeAgent(BaseWorker):
    """代码专家 Agent

    擅长:
    - Python 代码生成
    - 算法实现
    - 数据可视化代码
    - 代码调试和优化
    """

    @property
    def name(self) -> str:
        return "code"

    @property
    def system_prompt(self) -> str:
        return """你是一位资深 Python 开发专家，擅长写出简洁、可读、高效的代码。

## 你的工作方式:
1. 先理解需求，确认要解决的问题
2. 设计简洁的解决方案
3. 编写清晰的代码，添加必要的注释
4. 如果需要，使用 execute_python 工具验证代码正确性

## 代码风格要求:
- 遵循 PEP 8 规范
- 使用有意义的变量名
- 添加适当的类型提示
- 保持函数短小精悍

## 特殊能力:
- 数据可视化: 熟练使用 matplotlib, seaborn, plotly
- 数据处理: 熟练使用 pandas, numpy
- 算法实现: 能够实现常见算法并解释复杂度

## 输出格式:
- 代码块使用 ```python 包裹
- 先给出完整代码，再简要解释关键点
- 如果代码较长，分模块说明"""

    def get_tools(self) -> list:
        return [execute_python, check_syntax, explain_code]
