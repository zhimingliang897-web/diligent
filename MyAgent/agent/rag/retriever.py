"""RAG 检索工具 — 包装为 LangChain Tool 供 Agent 调用"""

from pathlib import Path
from langchain_core.tools import tool

try:
    from langchain.retrievers.ensemble import EnsembleRetriever
except ImportError:
    try:
        from langchain_community.retrievers import EnsembleRetriever
    except ImportError:
        EnsembleRetriever = None

import pickle
from agent.rag.vectorstore import load_vectorstore, DEFAULT_STORE_PATH


def create_rag_tool(store_path=None, mode="advanced"):
    """创建一个 RAG 检索工具。返回 None 如果向量存储不存在。"""
    try:
        vs = load_vectorstore(store_path)
    except FileNotFoundError:
        return None

    # 1. 基础配置：FAISS 向量检索
    faiss_retriever = vs.as_retriever(search_type="mmr", search_kwargs={"k": 4})
    retriever_to_use = faiss_retriever

    # 2. 如果是 advanced 模式，尝试加载 BM25 组成混合检索
    if mode == "advanced":
        load_dir = Path(store_path) if store_path else DEFAULT_STORE_PATH
        bm25_path = load_dir / "index.bm25"
        
        if EnsembleRetriever is None:
            print("  [提示] 未找到 EnsembleRetriever，使用纯 FAISS 向量检索")
        elif bm25_path.exists():
            try:
                with open(bm25_path, "rb") as f:
                    bm25_retriever = pickle.load(f)
                bm25_retriever.k = 4
                
                retriever_to_use = EnsembleRetriever(
                    retrievers=[bm25_retriever, faiss_retriever], 
                    weights=[0.5, 0.5]
                )
                print("  [加载完毕] 启用 Hybrid Search (BM25 + FAISS)")
            except Exception as e:
                print(f"  [警告] 加载 BM25 失败，回退到纯 FAISS 检索: {e}")
        else:
            print("  [提示] 未找到 BM25 索引，使用纯 FAISS 向量检索")

    @tool
    def knowledge_search(query: str) -> str:
        """在用户的个人知识库中搜索相关信息。
        当用户询问关于其文档、笔记或知识库中的内容时使用此工具。
        返回最相关的文档片段。"""
        # 使用对应的检索器查库
        docs = retriever_to_use.invoke(query)
        if not docs:
            return "知识库中未找到相关内容。"

        results = []
        for i, doc in enumerate(docs, 1):
            source = str(doc.metadata.get("source", "未知来源"))
            # 只取文件名
            source_file = source.split("\\")[-1].split("/")[-1]
            
            # 如果是 advanced 模式，尝试提取刚才我们塞进去的被 Markdown 分块记忆的 Header
            if mode == "advanced":
                headers = []
                if "Header 1" in doc.metadata:
                    headers.append(doc.metadata["Header 1"])
                if "Header 2" in doc.metadata:
                    headers.append(doc.metadata["Header 2"])
                if "Header 3" in doc.metadata:
                    headers.append(doc.metadata["Header 3"])
                    
                if headers:
                    header_str = " > ".join(headers)
                    source_str = f"{source_file} - {header_str}"
                else:
                    source_str = source_file
            else:
                source_str = source_file
                
            results.append(f"[{i}] 来源: {source_str}\n{doc.page_content}")

        return "\n\n---\n\n".join(results)

    return knowledge_search
