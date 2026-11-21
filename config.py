"""Environment-backed settings for the local DocuMindAI app."""

from __future__ import annotations

import os
from pathlib import Path

from dotenv import load_dotenv


PROJECT_DIR = Path(__file__).resolve().parent
load_dotenv(PROJECT_DIR / ".env")


def _env_bool(name: str, default: bool = False) -> bool:
    value = os.getenv(name)
    if value is None:
        return default
    return value.strip().lower() in {"1", "true", "yes", "on"}


def _env_int(name: str, default: int) -> int:
    value = os.getenv(name)
    if value is None:
        return default
    try:
        return int(value)
    except ValueError as error:
        raise ValueError(f"{name} must be an integer.") from error


OLLAMA_BASE_URL = os.getenv("OLLAMA_BASE_URL", "http://127.0.0.1:11434").rstrip("/")
OLLAMA_CHAT_MODEL = os.getenv("OLLAMA_CHAT_MODEL", "llama3.2")
OLLAMA_EMBED_MODEL = os.getenv("OLLAMA_EMBED_MODEL", "nomic-embed-text")
CHROMA_DIR = Path(os.getenv("CHROMA_DIR", str(PROJECT_DIR / "data" / "chroma"))).expanduser()
if not CHROMA_DIR.is_absolute():
    CHROMA_DIR = PROJECT_DIR / CHROMA_DIR
CHROMA_COLLECTION = os.getenv("CHROMA_COLLECTION", "documind")
CHUNK_SIZE = _env_int("CHUNK_SIZE", 1000)
CHUNK_OVERLAP = _env_int("CHUNK_OVERLAP", 150)
RETRIEVAL_K = _env_int("RETRIEVAL_K", 4)
MAX_PAGE_BYTES = _env_int("MAX_PAGE_MB", 5) * 1024 * 1024
REQUEST_TIMEOUT = (5, 20)
ALLOW_PRIVATE_URLS = _env_bool("ALLOW_PRIVATE_URLS", False)

if CHUNK_SIZE <= 0 or CHUNK_OVERLAP < 0 or CHUNK_OVERLAP >= CHUNK_SIZE:
    raise ValueError("Chunk settings must satisfy 0 <= CHUNK_OVERLAP < CHUNK_SIZE.")
if MAX_PAGE_BYTES <= 0:
    raise ValueError("MAX_PAGE_MB must be positive.")
