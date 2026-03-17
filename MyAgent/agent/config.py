import logging
import os
from dotenv import load_dotenv

load_dotenv()

# ──────────────────────── 日志配置 ────────────────────────
LOG_LEVEL = os.getenv("LOG_LEVEL", "INFO").upper()
logging.basicConfig(
    level=getattr(logging, LOG_LEVEL, logging.INFO),
    format="%(asctime)s [%(name)s] %(levelname)s: %(message)s",
    datefmt="%H:%M:%S",
)

DASHSCOPE_API_KEY = os.getenv("DASHSCOPE_API_KEY")


def require_dashscope_api_key() -> str:
    """Return DashScope API key or raise a helpful error at runtime.

    Avoid raising during import so non-LLM modules can still load.
    """
    if not DASHSCOPE_API_KEY:
        raise ValueError(
            "DASHSCOPE_API_KEY not found. Please create a .env file (or set env var) with your API key."
        )
    return DASHSCOPE_API_KEY

# ──────────────────────── 模型配置 ────────────────────────
MODEL_NAME = os.getenv("MODEL_NAME", "qwen-plus")
TOOL_MODEL_NAME = os.getenv("TOOL_MODEL_NAME", "qwen-plus")
EMBEDDING_MODEL = os.getenv("EMBEDDING_MODEL", "text-embedding-v3")
VISION_MODEL = os.getenv("VISION_MODEL", "qwen3-vl-plus-2025-12-19")
ASR_MODEL = os.getenv("ASR_MODEL", "qwen3-tts-vd-realtime-2026-01-15")
TEXT_TO_IMAGE_MODEL = os.getenv("TEXT_TO_IMAGE_MODEL", "qwen-image-2.0")

# VL 视觉模型可使用独立 API Key（留空则复用主 Key）
VL_API_KEY = os.getenv("VL_API_KEY", "").strip() or DASHSCOPE_API_KEY

# ──────────────────────── 对话与工具限制 ────────────────────────
def _int_env(key: str, default: int) -> int:
    try:
        return int(os.getenv(key, str(default)))
    except ValueError:
        return default

MAX_MESSAGES = _int_env("MAX_MESSAGES", 20)
MAX_TOOL_ITERATIONS = _int_env("MAX_TOOL_ITERATIONS", 5)

# ──────────────────────── RAG 配置 ────────────────────────
def _bool_env(key: str, default: str = "0") -> bool:
    return os.getenv(key, default).strip().lower() in ("1", "true", "yes")

ENABLE_RAG = _bool_env("ENABLE_RAG", "1")
RAG_MODE = os.getenv("RAG_MODE", "advanced")

# ──────────────────────── 可选付费能力（默认关闭，避免误扣费）────────────────────────
ENABLE_TEXT_TO_IMAGE = _bool_env("ENABLE_TEXT_TO_IMAGE", "1")
ENABLE_IMAGE_TO_TEXT = _bool_env("ENABLE_IMAGE_TO_TEXT")
ENABLE_VL = _bool_env("ENABLE_VL")

# ──────────────────────── 回答附加配置 ────────────────────────
TOOL_EVIDENCE_REQUIRED = _bool_env("TOOL_EVIDENCE_REQUIRED", "1")
TOOL_EVIDENCE_MAX_CHARS = _int_env("TOOL_EVIDENCE_MAX_CHARS", 280)
