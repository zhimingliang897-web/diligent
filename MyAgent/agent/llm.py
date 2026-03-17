from langchain_community.chat_models import ChatTongyi
from agent.config import MODEL_NAME, TOOL_MODEL_NAME, require_dashscope_api_key
from agent.callbacks import get_usage_callback


def get_llm(callbacks=None, streaming=True, for_tools=False):
    """Create a ChatTongyi LLM instance.

    Args:
        callbacks: optional callback list
        streaming: whether stream tokens
        for_tools: use tool-calling friendly model for bind_tools scenarios
    """
    # 默认挂载全局 Token 统计回调
    if callbacks is None:
        callbacks = [get_usage_callback()]
    elif isinstance(callbacks, list) and get_usage_callback() not in callbacks:
        callbacks.append(get_usage_callback())

    model_name = TOOL_MODEL_NAME if for_tools else MODEL_NAME

    return ChatTongyi(
        model=model_name,
        api_key=require_dashscope_api_key(),
        temperature=0.7,
        streaming=streaming,
        callbacks=callbacks
    )
