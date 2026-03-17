import asyncio
from agent.llm import get_llm
from agent.tools import calculate, web_search, get_current_datetime
from langchain_core.messages import HumanMessage, SystemMessage
from main import SYSTEM_PROMPT

async def test_repro():
    llm = get_llm(streaming=False) # Use non-streaming to see the full error if any
    tools = [calculate, web_search, get_current_datetime]
    llm_with_tools = llm.bind_tools(tools)
    
    messages = [
        SystemMessage(content=SYSTEM_PROMPT),
        HumanMessage(content="看看今天新加坡天气")
    ]
    
    try:
        print("--- Testing Tool Call ---")
        response = await llm_with_tools.ainvoke(messages)
        print("Response:", response)
        if response.tool_calls:
            print("Tool Calls found:", response.tool_calls)
        else:
            print("No tool calls found.")
    except Exception as e:
        print("Caught Error:", e)

if __name__ == "__main__":
    asyncio.run(test_repro())
