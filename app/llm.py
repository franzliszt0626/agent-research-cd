from __future__ import annotations

import json
import re
import asyncio
from functools import lru_cache
from typing import Any

from langchain_core.messages import HumanMessage, SystemMessage
from langchain_openai import ChatOpenAI

from .config import settings, validate_base_url


class LLMServiceError(RuntimeError):
    def __init__(self, code: str, detail: str):
        self.code = code
        super().__init__(f"{code}: {detail}")


def describe_llm_error(exc: Exception) -> LLMServiceError:
    if isinstance(exc, LLMServiceError):
        return exc
    status = getattr(exc, "status_code", None)
    chain = []
    current = exc
    while current is not None and len(chain) < 5:
        chain.append(type(current).__name__)
        current = current.__cause__
    if status in (401, 403):
        return LLMServiceError("authentication_error", "API Token 无效或无模型访问权限；检查 LLM_API_KEY。")
    if status == 404:
        return LLMServiceError("endpoint_or_model_not_found", "API 路径或模型不存在；检查 LLM_BASE_URL 的 /v1 路径和 LLM_MODEL。")
    if status == 429:
        return LLMServiceError("rate_limited", "模型服务限流或额度不足；降低并发或检查额度。")
    if any("Timeout" in name for name in chain):
        return LLMServiceError("network_timeout", "模型服务连接/响应超时；检查地址、网络和代理。" + " → ".join(chain))
    if any("Connect" in name or "Connection" in name for name in chain):
        return LLMServiceError("connection_error", "无法连接模型服务；检查 LLM_BASE_URL（尤其末尾逗号）、DNS、代理及服务状态。" + " → ".join(chain))
    if isinstance(exc, (ValueError, json.JSONDecodeError)):
        return LLMServiceError("invalid_json", "模型未返回合法 JSON。")
    return LLMServiceError("provider_error", f"模型服务错误；HTTP {status or 'unknown'}，类型 {type(exc).__name__}。")


@lru_cache(maxsize=1)
def get_llm() -> ChatOpenAI:
    """OpenAI-compatible chat model; works with cloud APIs and compatible local servers."""
    kwargs: dict[str, Any] = {
        "model": settings.llm_model,
        "temperature": settings.llm_temperature,
        "timeout": settings.llm_timeout_seconds,
        "api_key": settings.llm_api_key or ("local-no-key" if settings.llm_base_url else ""),
        "max_retries": 0,  # One bounded retry policy, not SDK retries multiplied by caller retries.
    }
    if settings.llm_base_url:
        try:
            kwargs["base_url"] = validate_base_url(settings.llm_base_url)
        except ValueError as exc:
            raise LLMServiceError("configuration_error", str(exc)) from exc
    if not kwargs["api_key"]:
        raise RuntimeError(
            "未配置模型凭据。请在 .env 中设置 LLM_API_KEY；使用本地 OpenAI 兼容模型时，"
            "同时设置 LLM_BASE_URL 和任意非空 LLM_API_KEY（例如 ollama）。"
        )
    return ChatOpenAI(**kwargs)


def message_text(message: Any) -> str:
    content = getattr(message, "content", message)
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        parts: list[str] = []
        for part in content:
            if isinstance(part, dict) and isinstance(part.get("text"), str):
                parts.append(part["text"])
            else:
                parts.append(str(part))
        return "\n".join(parts)
    return str(content)


def parse_json_response(text: str) -> Any:
    """Parse JSON even when the model adds a Markdown fence or brief lead-in."""
    cleaned = text.strip()
    cleaned = re.sub(r"^```(?:json)?\s*", "", cleaned, flags=re.IGNORECASE)
    cleaned = re.sub(r"\s*```$", "", cleaned)
    try:
        return json.loads(cleaned)
    except json.JSONDecodeError:
        pass
    starts = [p for p in (cleaned.find("{"), cleaned.find("[")) if p >= 0]
    if not starts:
        raise ValueError("模型回复中没有可解析的 JSON")
    start = min(starts)
    for end in range(len(cleaned), start, -1):
        candidate = cleaned[start:end].strip()
        if candidate.endswith("}") or candidate.endswith("]"):
            try:
                return json.loads(candidate)
            except json.JSONDecodeError:
                continue
    raise ValueError("无法从模型回复中解析 JSON")


async def ainvoke_json(system_prompt: str, user_prompt: str, *, retries: int = 0) -> Any:
    llm = get_llm()
    last_error: Exception | None = None
    for attempt in range(retries + 1):
        try:
            response = await llm.ainvoke(
                [SystemMessage(content=system_prompt), HumanMessage(content=user_prompt)]
            )
            return parse_json_response(message_text(response))
        except Exception as exc:  # network/provider errors should be reported by caller
            last_error = describe_llm_error(exc)
            if attempt >= retries or last_error.code in {"configuration_error", "authentication_error", "endpoint_or_model_not_found"}:
                break
            await asyncio.sleep(min(4, 0.5 * 2 ** attempt))
    raise last_error


async def check_llm() -> dict[str, Any]:
    """Check the configured model via a tiny real JSON completion before starting a batch."""
    try:
        result = await asyncio.wait_for(ainvoke_json(
            'Return exactly one JSON object: {"ok": true}.', 'Check JSON availability.', retries=0),
            timeout=min(30, settings.llm_timeout_seconds))
        if not isinstance(result, dict) or result.get("ok") is not True:
            raise LLMServiceError("invalid_json", "模型自检未返回约定 JSON；请确认模型支持指令和 JSON 输出。")
    except Exception as exc:
        raise describe_llm_error(exc) from exc
    return {"status": "ok", "model": settings.llm_model, "base_url": validate_base_url(settings.llm_base_url)}
