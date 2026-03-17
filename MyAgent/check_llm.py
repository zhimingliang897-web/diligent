import dashscope
from langchain_community.chat_models import ChatTongyi
from agent.config import DASHSCOPE_API_KEY, MODEL_NAME
from langchain_core.messages import HumanMessage
from langchain_core.tools import tool

print(f"DashScope version: {dashscope.__version__}")

@tool
def add(a: int, b: int) -> int:
    """Add two numbers."""
    return a + b

llm = ChatTongyi(model=MODEL_NAME, api_key=DASHSCOPE_API_KEY, temperature=0).bind_tools([add])
try:
    res = llm.invoke([HumanMessage(content="1加1等于几")])
    print("Result:", res)
except Exception as e:
    print("Error:", e)
