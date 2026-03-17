"""
一键索引文档脚本（稳健版）

用法:
    python scripts/index_docs.py <文档路径>                    # 全量重建（会备份旧索引）
    python scripts/index_docs.py <文档路径> --append            # 增量追加到已有索引
    python scripts/index_docs.py <文档路径> --batch-size 10      # 自定义 batch 大小
    python scripts/index_docs.py <文档路径> --chunk-size 800     # 调整分块
    python scripts/index_docs.py <文档路径> --store-path path/to/vectorstore
    python scripts/index_docs.py <文档路径> --fail-log index_failures.txt

支持: .txt, .md, .pdf
"""

import argparse
import sys
import shutil
import traceback
from datetime import datetime
from pathlib import Path

# 让 import agent 正常工作
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from agent.rag.loader import load_documents
from agent.rag.vectorstore import split_documents, build_vectorstore, DEFAULT_STORE_PATH, DEFAULT_BATCH_SIZE


SUPPORTED_EXTS = {".txt", ".md", ".pdf"}


def iter_supported_files(path: str):
    p = Path(path)
    if p.is_file():
        if p.suffix.lower() in SUPPORTED_EXTS:
            yield p
        return
    if not p.is_dir():
        raise FileNotFoundError(f"路径不存在: {path}")

    for f in p.rglob("*"):
        if f.is_file() and f.suffix.lower() in SUPPORTED_EXTS:
            yield f


def log_failure(fail_log: Path, file_path: Path, stage: str, err: Exception):
    fail_log.parent.mkdir(parents=True, exist_ok=True)
    with open(fail_log, "a", encoding="utf-8") as fp:
        fp.write("=" * 88 + "\n")
        fp.write(f"time : {datetime.now().isoformat(timespec='seconds')}\n")
        fp.write(f"file : {file_path}\n")
        fp.write(f"stage: {stage}\n")
        fp.write(f"error: {repr(err)}\n")
        fp.write(traceback.format_exc())
        fp.write("\n")


def backup_existing_store(store_dir: Path):
    """全量重建时：备份旧索引目录（比直接删更稳妥）"""
    if not store_dir.exists():
        return
    if not (store_dir / "index.faiss").exists():
        return

    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    backup_dir = store_dir.parent / f"{store_dir.name}_backup_{ts}"
    shutil.move(str(store_dir), str(backup_dir))
    print(f"  已备份旧索引到: {backup_dir}", flush=True)


def main():
    parser = argparse.ArgumentParser(description="索引文档到向量存储（稳健版）")
    parser.add_argument("path", help="文档文件或目录路径")

    parser.add_argument(
        "--append", action="store_true",
        help="追加到已有索引（默认全量重建：会备份旧索引）",
    )
    parser.add_argument(
        "--store-path", type=str, default=None,
        help=f"向量存储目录（默认 {DEFAULT_STORE_PATH}）",
    )
    parser.add_argument(
        "--fail-log", type=str, default=None,
        help="失败文件记录日志路径（默认放在 store 的同级目录）",
    )

    parser.add_argument(
        "--batch-size", type=int, default=DEFAULT_BATCH_SIZE,
        help=f"每批发送给 embedding API 的文档数 (默认 {DEFAULT_BATCH_SIZE})",
    )
    parser.add_argument(
        "--chunk-size", type=int, default=500,
        help="分块大小，单位字符 (默认 500)",
    )
    parser.add_argument(
        "--chunk-overlap", type=int, default=100,
        help="分块重叠字符数 (默认 100)",
    )
    parser.add_argument(
        "--sleep", type=float, default=0.0,
        help="每个 batch 之间 sleep 秒数（限流时可设 0.5~2.0）",
    )
    parser.add_argument(
        "--no-save-every-batch", action="store_true",
        help="关闭每批落盘（不推荐；中断会损失更多进度）",
    )

    args = parser.parse_args()

    store_dir = Path(args.store_path) if args.store_path else DEFAULT_STORE_PATH
    fail_log = Path(args.fail_log) if args.fail_log else (store_dir.parent / "index_failures.txt")

    # 全量模式：先备份旧索引
    if not args.append:
        backup_existing_store(store_dir)

    files = list(iter_supported_files(args.path))
    if not files:
        print("未找到任何支持的文档（.txt/.md/.pdf），退出。")
        sys.exit(1)

    print(f"将处理 {len(files)} 个文件。失败日志: {fail_log}", flush=True)
    print(f"索引目录: {store_dir} | 模式: {'追加' if args.append else '全量重建'}", flush=True)
    print(
        f"chunk_size={args.chunk_size}, overlap={args.chunk_overlap}, "
        f"batch_size={args.batch_size}, save_every_batch={not args.no_save_every_batch}, sleep={args.sleep}",
        flush=True,
    )

    total_chunks = 0
    ok_files = 0
    failed_files = 0

    vs = None  # ✅ 复用向量库对象，避免每个文件都 load_local 一次

    try:
        for idx, file_path in enumerate(files, 1):
            print(f"\n[{idx}/{len(files)}] {file_path}", flush=True)

            # 1) 加载
            try:
                docs = load_documents(str(file_path))
            except Exception as e:
                print(f"  ❌ 加载失败: {e}", flush=True)
                log_failure(fail_log, file_path, "load", e)
                failed_files += 1
                continue

            if not docs:
                print("  ⚠️ 加载为空，跳过。", flush=True)
                continue

            # 2) 分块
            try:
                chunks = split_documents(
                    docs,
                    chunk_size=args.chunk_size,
                    chunk_overlap=args.chunk_overlap,
                )
            except Exception as e:
                print(f"  ❌ 分块失败: {e}", flush=True)
                log_failure(fail_log, file_path, "split", e)
                failed_files += 1
                continue

            if not chunks:
                print("  ⚠️ 分块后为空，跳过。", flush=True)
                continue

            # 3) 向量化 + 追加入库（全量模式也用“从空库开始追加”的方式）
            try:
                vs = build_vectorstore(
                    chunks,
                    store_path=str(store_dir),
                    batch_size=args.batch_size,
                    append=True,  # ✅ 统一走追加：若不存在索引，会自动从第一批创建
                    vectorstore=vs,
                    save_every_batch=(not args.no_save_every_batch),
                    sleep_s=args.sleep,
                )
                total_chunks += len(chunks)
                ok_files += 1
            except Exception as e:
                print(f"  ❌ 向量化失败: {e}", flush=True)
                log_failure(fail_log, file_path, "embed", e)
                failed_files += 1
                continue

    except KeyboardInterrupt:
        print("\n收到 Ctrl+C：已尽力保存当前进度（若开启每批落盘，通常不会白跑太多）。", flush=True)
        sys.exit(130)

    print("\n完成!", flush=True)
    print(f"成功文件: {ok_files} | 失败文件: {failed_files}", flush=True)
    print(f"累计写入片段: {total_chunks}", flush=True)
    print(f"失败日志: {fail_log}", flush=True)


if __name__ == "__main__":
    main()
