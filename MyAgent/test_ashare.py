import asyncio
from main import _build_graph_agent, SYSTEM_PROMPT
from agent.llm import get_llm
from agent.tools import get_all_tools
from langgraph.checkpoint.sqlite.aio import AsyncSqliteSaver
from langchain_core.messages import HumanMessage

async def t():
    async with AsyncSqliteSaver.from_conn_string(":memory:") as m:
        a = _build_graph_agent(get_llm(), list(get_all_tools()), m, SYSTEM_PROMPT)
        print("======== 发送请求 ========")
        r = await a.ainvoke({'messages': [HumanMessage(content='3.17a股开盘点数是大概多少')]}, config={'configurable': {'thread_id': 'test'}})
        
        for i, msg in enumerate(r['messages']):
            print(f"[{msg.type.upper()}] {msg.content}")
            if hasattr(msg, "tool_calls") and msg.tool_calls:
                print(f"   Tool Calls: {msg.tool_calls}")

asyncio.run(t())
