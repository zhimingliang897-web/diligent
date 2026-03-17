import json
import logging
import time
from pathlib import Path
from typing import Dict, Any

logger = logging.getLogger(__name__)

PROFILE_PATH = Path("data/user_profile.json")

_cache: Dict[str, Any] | None = None
_cache_ts: float = 0.0
_CACHE_TTL = 30.0


def load_profile() -> Dict[str, Any]:
    """读取用户的长期画像和偏好信息（带 30s 内存缓存）。"""
    global _cache, _cache_ts
    now = time.time()
    if _cache is not None and (now - _cache_ts) < _CACHE_TTL:
        return _cache

    if not PROFILE_PATH.exists():
        _cache, _cache_ts = {}, now
        return _cache
    try:
        with open(PROFILE_PATH, "r", encoding="utf-8") as f:
            _cache = json.load(f)
            _cache_ts = now
            return _cache
    except Exception as e:
        logger.warning("加载用户画像失败: %s", e)
        return {}

def save_profile(profile: Dict[str, Any]):
    """保存用户的长期画像和偏好信息。"""
    PROFILE_PATH.parent.mkdir(parents=True, exist_ok=True)
    with open(PROFILE_PATH, "w", encoding="utf-8") as f:
        json.dump(profile, f, ensure_ascii=False, indent=2)

def _invalidate_cache():
    global _cache, _cache_ts
    _cache, _cache_ts = None, 0.0


def update_user_fact(key: str, value: str) -> str:
    """更新用户的单个信息（例如：姓名、职业、偏好）。"""
    profile = load_profile()
    profile[key] = value
    save_profile(profile)
    _invalidate_cache()
    return f"已成功记住关于你的信息: {key} = {value}"

def get_profile_summary() -> str:
    """获取格式化的用户画像摘要，用于注入 Prompt。"""
    profile = load_profile()
    if not profile:
        return "暂无关于该用户的特殊背景信息。"
    
    summary = "=== 用户个人画像与偏好 ===\n"
    for k, v in profile.items():
        summary += f"- {k}: {v}\n"
    return summary

def clear_profile() -> str:
    """清空用户长期记忆。"""
    save_profile({})
    _invalidate_cache()
    return "已清空所有长期记忆！"
