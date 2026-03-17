"""向量存储 — 文档分块 + FAISS 向量化 + 持久化"""

import sys
import time
from pathlib import Path
from typing import List, Optional

from langchain_community.embeddings import DashScopeEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_core.documents import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter, MarkdownHeaderTextSplitter
from langchain_community.retrievers import BM25Retriever
import pickle

from agent.config import DASHSCOPE_API_KEY, EMBEDDING_MODEL

# 默认持久化路径
DEFAULT_STORE_PATH = Path(__file__).resolve().parent.parent.parent / "vectorstore"

# DashScope text-embedding-v3 限制: 每次最多 25 个文本
# 参考: https://help.aliyun.com/zh/model-studio/text-embedding-api
DEFAULT_BATCH_SIZE = 25


_embeddings_instance: DashScopeEmbeddings | None = None


def get_embeddings() -> DashScopeEmbeddings:
    global _embeddings_instance
    if _embeddings_instance is None:
        _embeddings_instance = DashScopeEmbeddings(
            model=EMBEDDING_MODEL,
            dashscope_api_key=DASHSCOPE_API_KEY,
        )
    return _embeddings_instance


def split_documents(
    docs: List[Document],
    chunk_size: int = 500,
    chunk_overlap: int = 100,
) -> List[Document]:
    """将文档分块，针对 Markdown 尝试按标题进行语义分块。"""
    
    # 定义 Markdown 标题层级，用于把标题信息放入 metadata
    headers_to_split_on = [
        ("#", "Header 1"),
        ("##", "Header 2"),
        ("###", "Header 3"),
    ]
    md_splitter = MarkdownHeaderTextSplitter(headers_to_split_on=headers_to_split_on, strip_headers=False)
    
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
        separators=["\n\n", "\n", "。", "！", "？", ".", "!", "?", " "],
    )

    all_chunks = []
    
    for doc in docs:
        source = str(doc.metadata.get("source", ""))
        # 如果是 markdown 文件，先尝试按语义标题切分
        if source.lower().endswith(".md"):
            try:
                # 语义切分
                md_chunks = md_splitter.split_text(doc.page_content)
                # 即使按标题切了，每个段落依然可能超过 500 字，所以还要过一遍按字数切分的 splitter
                # 并且要把原有的 metadata (如 source) 给继承过来
                for chunk in md_chunks:
                    chunk.metadata.update(doc.metadata) 
                
                final_chunks = text_splitter.split_documents(md_chunks)
                all_chunks.extend(final_chunks)
            except Exception as e:
                print(f"  [警告] Markdown 语义分块失败 ({source})，退回基础分块: {e}")
                all_chunks.extend(text_splitter.split_documents([doc]))
        else:
            # 非 Markdown 文件直接按字数切分
            all_chunks.extend(text_splitter.split_documents([doc]))

    print(f"  分块完成: {len(docs)} 个文档 → {len(all_chunks)} 个片段")
    return all_chunks

def build_vectorstore(
    chunks: List[Document],
    store_path: Optional[str] = None,
    batch_size: int = DEFAULT_BATCH_SIZE,
    append: bool = False,
    vectorstore: Optional[FAISS] = None,       # ✅ 允许复用已加载的 vs，避免反复 load_local
    save_every_batch: bool = True,             # ✅ 每批落盘，防止中断白跑
    sleep_s: float = 0.0,                      # ✅ 批间等待（限流时可调大）
) -> FAISS:
    """从文档片段构建/追加 FAISS 向量存储并持久化。"""
    save_dir = Path(store_path) if store_path else DEFAULT_STORE_PATH
    embeddings = get_embeddings()
    save_dir.mkdir(parents=True, exist_ok=True)

    # 需要追加且还没传入 vectorstore 时，尝试从磁盘加载
    if vectorstore is None and append and (save_dir / "index.faiss").exists():
        vectorstore = FAISS.load_local(
            str(save_dir), embeddings, allow_dangerous_deserialization=True
        )
        print(f"  已加载现有索引，将追加 {len(chunks)} 个片段", flush=True)

    total = len(chunks)

    try:
        for i in range(0, total, batch_size):
            batch = chunks[i : i + batch_size]
            end = min(i + batch_size, total)
            print(f"  向量化 [{i + 1}-{end}] / {total} ...", flush=True)

            if vectorstore is None:
                vectorstore = FAISS.from_documents(batch, embeddings)
            else:
                vectorstore.add_documents(batch)

            if save_every_batch:
                vectorstore.save_local(str(save_dir))
                # 可选：打印更少一点就把下面这行注释掉
                print(f"  已保存进度到 {save_dir}", flush=True)

            if sleep_s and end < total:
                time.sleep(sleep_s)

    except KeyboardInterrupt:
        print("\n  收到 Ctrl+C，中断构建；正在保存当前索引...", flush=True)
        if vectorstore is not None:
            vectorstore.save_local(str(save_dir))
            print("  已保存。退出。", flush=True)
        sys.exit(130)

    # 最后再保存一次 (FAISS)
    vectorstore.save_local(str(save_dir))
    
    # ──────────────────────── BM25 构建与落盘 ────────────────────────
    print("  [BM25] 正在基于全库重建 BM25 词频索引...", flush=True)
    try:
        # 直接从 FAISS 内置的 docstore 中提取所有已入库的文档片段
        all_docs_in_store = list(vectorstore.docstore._dict.values())
        if all_docs_in_store:
            bm25_retriever = BM25Retriever.from_documents(all_docs_in_store)
            
            # 使用 pickle 序列化整个 BM25 检索器并保存为文件
            bm25_path = save_dir / "index.bm25"
            with open(bm25_path, "wb") as f:
                pickle.dump(bm25_retriever, f)
            print(f"  [BM25] 索引已保存到 {bm25_path} (共 {len(all_docs_in_store)} 个库内片段)", flush=True)
        else:
            print("  [BM25] 警告：向量库为空，跳过 BM25 构建。", flush=True)
    except Exception as e:
        print(f"  [BM25] 构建/保存失败: {e}", flush=True)
        
    print(f"  向量存储构建完毕 (本次处理 {total} 个新片段)", flush=True)
    return vectorstore

def load_vectorstore(store_path: Optional[str] = None) -> FAISS:
    """从磁盘加载已有的 FAISS 向量存储。"""
    load_dir = Path(store_path) if store_path else DEFAULT_STORE_PATH

    if not (load_dir / "index.faiss").exists():
        raise FileNotFoundError(
            f"向量存储不存在: {load_dir}\n"
            "请先运行 python scripts/index_docs.py <文档路径> 建立索引"
        )

    embeddings = get_embeddings()
    return FAISS.load_local(
        str(load_dir), embeddings, allow_dangerous_deserialization=True
    )
