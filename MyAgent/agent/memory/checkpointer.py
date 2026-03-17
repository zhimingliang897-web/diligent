import logging
import os
import sqlite3
from datetime import datetime
from typing import List, Dict, Optional

from langgraph.checkpoint.sqlite.aio import AsyncSqliteSaver

logger = logging.getLogger(__name__)

DEFAULT_DB_PATH = "data/db/agent_memory.db"


def _ensure_dir(db_path: str):
    directory = os.path.dirname(db_path)
    if directory and not os.path.exists(directory):
        os.makedirs(directory, exist_ok=True)


def get_checkpointer_context(db_path: str = DEFAULT_DB_PATH):
    """Returns an async context manager for AsyncSqliteSaver."""
    _ensure_dir(db_path)
    return AsyncSqliteSaver.from_conn_string(db_path)


# ──────────────── 会话管理 ────────────────

def list_sessions(db_path: str = DEFAULT_DB_PATH) -> List[Dict]:
    """查询所有会话 thread_id 及其最近写入时间。"""
    _ensure_dir(db_path)
    if not os.path.exists(db_path):
        return []
    try:
        conn = sqlite3.connect(db_path)
        cursor = conn.execute(
            "SELECT DISTINCT thread_id FROM checkpoints ORDER BY thread_id"
        )
        sessions = []
        for (tid,) in cursor.fetchall():
            sessions.append({"thread_id": tid})
        conn.close()
        return sessions
    except Exception as e:
        logger.warning("查询会话列表失败: %s", e)
        return []


def delete_session(thread_id: str, db_path: str = DEFAULT_DB_PATH) -> bool:
    """删除指定会话的所有 checkpoint 数据。"""
    if not os.path.exists(db_path):
        return False
    try:
        conn = sqlite3.connect(db_path)
        conn.execute("DELETE FROM checkpoints WHERE thread_id = ?", (thread_id,))
        try:
            conn.execute("DELETE FROM writes WHERE thread_id = ?", (thread_id,))
        except sqlite3.OperationalError:
            pass
        conn.commit()
        conn.close()
        logger.info("已删除会话: %s", thread_id)
        return True
    except Exception as e:
        logger.warning("删除会话失败: %s", e)
        return False


def rename_session(old_id: str, new_id: str, db_path: str = DEFAULT_DB_PATH) -> bool:
    """重命名会话（更新 thread_id）。"""
    if not os.path.exists(db_path):
        return False
    try:
        conn = sqlite3.connect(db_path)
        conn.execute(
            "UPDATE checkpoints SET thread_id = ? WHERE thread_id = ?",
            (new_id, old_id),
        )
        try:
            conn.execute(
                "UPDATE writes SET thread_id = ? WHERE thread_id = ?",
                (new_id, old_id),
            )
        except sqlite3.OperationalError:
            pass
        conn.commit()
        conn.close()
        logger.info("会话已重命名: %s -> %s", old_id, new_id)
        return True
    except Exception as e:
        logger.warning("重命名会话失败: %s", e)
        return False
