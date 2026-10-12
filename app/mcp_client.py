"""MCP transport boundary. Used exclusively by Agent3 workflow nodes."""
from __future__ import annotations

import asyncio
import json
import sys
from pathlib import Path
from typing import Any

from langchain_mcp_adapters.client import MultiServerMCPClient

from .config import settings
from .schemas import LiteratureMetadata


class MCPFailure(RuntimeError):
    def __init__(self, status: str, message: str):
        super().__init__(message)
        self.status = status


def exception_detail(exc: BaseException) -> str:
    nested = getattr(exc, "exceptions", None)
    if nested:
        return "; ".join(exception_detail(child) for child in nested)
    return f"{type(exc).__name__}: {exc}"


def error_status(exc: BaseException) -> str:
    detail = exception_detail(exc).lower()
    return "network_timeout" if "timeout" in detail or "timed out" in detail else "mcp_error"


def decode_tool_result(result: Any) -> dict[str, Any]:
    if getattr(result, "status", None) == "error":
        raise MCPFailure("mcp_error", str(getattr(result, "content", result)))
    if hasattr(result, "model_dump"):
        result = result.model_dump()
    if isinstance(result, str):
        try:
            return decode_tool_result(json.loads(result))
        except (ValueError, TypeError) as exc:
            raise MCPFailure("invalid_response", "MCP tool returned non-JSON text") from exc
    if isinstance(result, dict):
        if result.get("isError") or result.get("is_error") or result.get("status") == "error":
            raise MCPFailure("mcp_error", str(result.get("content", result)))
        for key in ("structuredContent", "structured_content", "artifact"):
            if isinstance(result.get(key), dict) and result[key]:
                return decode_tool_result(result[key])
        if "found" in result or "full_text_available" in result:
            return result
        if "content" in result:
            return decode_tool_result(result["content"])
        if result.get("type") == "text" and "text" in result:
            return decode_tool_result(result["text"])
    if isinstance(result, list):
        for block in result:
            try:
                return decode_tool_result(block)
            except MCPFailure as exc:
                if exc.status != "invalid_response":
                    raise
    raise MCPFailure("invalid_response", "MCP tool response lacks metadata/full-text fields")


def normalize_metadata(raw: dict[str, Any]) -> dict[str, Any]:
    data = {key: value for key, value in raw.items() if key in LiteratureMetadata.model_fields and value is not None}
    if "status" not in data:
        access = raw.get("access_status", "")
        if raw.get("error"):
            data["status"] = "mcp_error"
        elif not raw.get("found"):
            data["status"] = "not_found"
        elif raw.get("is_open_access") is True or access.startswith("open-access"):
            data["status"] = "open_source"
        elif raw.get("is_open_access") is False or access in {"closed-source", "close-source"}:
            data["status"] = "close_source"
        else:
            data["status"] = "metadata_only"
    if data["status"] == "close_source":
        data["access_status"] = "close-source"
    if data["status"] == "open_source":
        data.setdefault("access_status", "open-access")
    return LiteratureMetadata.model_validate(data).model_dump()


class LiteratureClient:
    def __init__(self):
        connection: dict[str, Any]
        if settings.mcp_transport == "stdio":
            connection = {"transport": "stdio", "command": sys.executable,
                          "args": ["-m", "app.mcp_server", "--transport", "stdio"],
                          "cwd": str(Path(__file__).resolve().parent.parent)}
        else:
            connection = {"transport": "http", "url": settings.mcp_url,
                          "timeout": settings.mcp_timeout_seconds}
        self.client = MultiServerMCPClient({"literature": connection})
        self.tools: list[Any] | None = None
        self.lock = asyncio.Lock()

    async def prepare(self):
        async with self.lock:
            if self.tools is None:
                self.tools = await self._retry(self.client.get_tools)

    async def _retry(self, call):
        for attempt in range(max(0, settings.mcp_retries) + 1):
            try:
                return await asyncio.wait_for(call(), timeout=settings.mcp_timeout_seconds)
            except Exception as exc:
                if attempt >= settings.mcp_retries:
                    if isinstance(exc, MCPFailure):
                        raise
                    raise MCPFailure(error_status(exc), exception_detail(exc)) from exc
                await asyncio.sleep(min(4, settings.mcp_retry_delay_seconds * (2 ** attempt)))

    async def call(self, name: str, arguments: dict[str, Any]) -> dict[str, Any]:
        await self.prepare()
        tool = next((t for t in self.tools or [] if t.name == name or t.name.endswith("_" + name)), None)
        if tool is None:
            raise MCPFailure("mcp_error", f"Missing MCP tool: {name}")
        async def invoke():
            result = decode_tool_result(await tool.ainvoke(arguments))
            if result.get("status") in {"network_timeout", "network_error", "mcp_error", "invalid_response"}:
                raise MCPFailure(result["status"], result.get("error") or "; ".join(result.get("provider_errors", [])))
            return result
        return await self._retry(invoke)
