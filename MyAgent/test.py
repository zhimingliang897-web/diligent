from langchain_community.chat_models import ChatTongyi
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser

from agent.config import DASHSCOPE_API_KEY, MODEL_NAME

# 1. 配置模型 (API key 从 .env 文件加载)
llm = ChatTongyi(
    model=MODEL_NAME,
    api_key=DASHSCOPE_API_KEY,
)

# 2. 定义标准的 ChatPromptTemplate (修复了之前的语法错误)
prompt = ChatPromptTemplate.from_messages([
    ("system", "You are a helpful assistant."),
    ("user", "Hello, how are you?")
])

# 3. 构建 LCEL 链
chain = prompt | llm | StrOutputParser()

# 4. 执行并运行
# 这里会实际调用通义千问平台的 API
try:
    response = chain.invoke({})
    print(response)
except Exception as e:
    print(f"调用出错: {e}")