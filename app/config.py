from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path
from urllib.parse import urlsplit
from dotenv import load_dotenv

load_dotenv(Path(__file__).resolve().parent.parent / ".env")


def validate_base_url(value: str) -> str:
    value = value.strip().rstrip("/")
    if not value:
        return ""
    parsed = urlsplit(value)
    if (parsed.scheme not in {"http", "https"} or not parsed.hostname
            or parsed.username or parsed.password or parsed.query or parsed.fragment
            or re_invalid_host(parsed.hostname) or any(c.isspace() for c in value)
            or value.endswith((",", "，", ";", "；"))):
        raise ValueError("LLM_BASE_URL 格式错误：应为 http(s) API 基础地址，不可包含末尾逗号、空格、凭据或查询参数；请检查 .env。")
    if parsed.path.endswith(("/chat/completions", "/responses", "/models")):
        raise ValueError("LLM_BASE_URL 应填写 API 基础路径（如 /v1），不能填写 /chat/completions、/responses 或 /models。")
    return value


def re_invalid_host(host: str) -> bool:
    import re
    return bool(re.search(r"[^a-zA-Z0-9.\-:]", host))


def _get_int(name: str, default: int) -> int:
    try:
        return int(os.getenv(name, str(default)))
    except (TypeError, ValueError):
        return default


def _get_float(name: str, default: float) -> float:
    try:
        return float(os.getenv(name, str(default)))
    except (TypeError, ValueError):
        return default


@dataclass(frozen=True)
class Settings:
    llm_model: str = os.getenv("LLM_MODEL", "gpt-4o-mini")
    llm_api_key: str = os.getenv("LLM_API_KEY", os.getenv("OPENAI_API_KEY", ""))
    llm_base_url: str = os.getenv("LLM_BASE_URL", os.getenv("OPENAI_BASE_URL", ""))
    llm_temperature: float = _get_float("LLM_TEMPERATURE", 0.0)
    llm_timeout_seconds: int = _get_int("LLM_TIMEOUT_SECONDS", 120)
    llm_retries: int = _get_int("LLM_RETRIES", 1)
    reference_normalization_batch_size: int = _get_int("REFERENCE_NORMALIZATION_BATCH_SIZE", 40)
    reference_normalization_concurrency: int = _get_int("REFERENCE_NORMALIZATION_CONCURRENCY", 20)
    failed_node_retries: int = _get_int("FAILED_NODE_RETRIES", 2)
    failed_node_retry_delay_seconds: float = _get_float("FAILED_NODE_RETRY_DELAY_SECONDS", 1.0)
    confidence_threshold: float = _get_float("CONFIDENCE_THRESHOLD", 0.75)
    max_concurrent_llm_calls: int = _get_int("MAX_CONCURRENT_LLM_CALLS", 4)
    max_lookup_concurrency: int = _get_int("MAX_LOOKUP_CONCURRENCY", 3)
    max_pdf_bytes: int = _get_int("MAX_PDF_BYTES", 25_000_000)
    max_fulltext_chars: int = _get_int("MAX_FULLTEXT_CHARS", 40_000)
    min_fulltext_chars: int = _get_int("MIN_FULLTEXT_CHARS", 1_200)
    mcp_url: str = os.getenv("MCP_URL", "http://127.0.0.1:8001/mcp")
    mcp_transport: str = os.getenv("MCP_TRANSPORT", "stdio")
    mcp_retries: int = _get_int("MCP_RETRIES", 2)
    mcp_retry_delay_seconds: float = _get_float("MCP_RETRY_DELAY_SECONDS", 0.5)
    mcp_timeout_seconds: int = _get_int("MCP_TIMEOUT_SECONDS", 90)
    unpaywall_email: str = os.getenv("UNPAYWALL_EMAIL", "")
    crossref_mailto: str = os.getenv("CROSSREF_MAILTO", "")
    openalex_mailto: str = os.getenv("OPENALEX_MAILTO", "")
    app_host: str = os.getenv("APP_HOST", "0.0.0.0")
    app_port: int = _get_int("APP_PORT", 8000)
    max_upload_mb: int = _get_int("MAX_UPLOAD_MB", 40)


settings = Settings()
