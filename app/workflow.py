from __future__ import annotations

import asyncio
import json
import math
import re
import uuid
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path
from collections.abc import AsyncIterator
from typing import Any, TypedDict

from langgraph.graph import END, START, StateGraph
from langchain_core.callbacks import adispatch_custom_event

from .citation_mapper import extract_docx_references_with_llm, map_citations, normalize_references
from .config import settings
from .document_loader import load_document
from .llm import ainvoke_json, check_llm
from .mcp_client import LiteratureClient, MCPFailure, decode_tool_result, normalize_metadata
from .prompts import DEEP_REVIEW_SYSTEM, INITIAL_REVIEW_SYSTEM
from .schemas import ClaimPair, ReferenceItem, ReportSchema, Review, ReviewPayload

_decode_tool_result = decode_tool_result
LOOKUP_FAILURES = {"mcp_error", "network_timeout", "network_error", "invalid_response", "not_found"}


def _now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _unavailable(reason: str, rationale: str, basis: str, *, error: bool = False) -> dict[str, Any]:
    return Review(status="error" if error else "not_available", rationale=rationale,
                  basis=basis, failure_reason=reason).model_dump()


def _passed(review: dict[str, Any] | None, threshold: float) -> bool:
    return bool(review and review.get("status") == "completed"
                and review.get("confidence") is not None and review["confidence"] > threshold
                and review.get("verdict") in {"supports", "partially_supports"})


def _needs_deep(pair: dict[str, Any], lookup: dict[str, Any], threshold: float) -> bool:
    initial = pair.get("initial_review") or {}
    return (lookup.get("status") not in LOOKUP_FAILURES and not _passed(initial, threshold)
            and initial.get("status") != "error")


class WorkflowState(TypedDict, total=False):
    input_path: str
    threshold: float
    task_id: str
    started_at: str
    source_name: str
    main_text: str
    references: list[dict[str, Any]]
    pairs: list[dict[str, Any]]
    unmatched_references: list[dict[str, Any]]
    lookups: dict[str, dict[str, Any]]
    warnings: list[str]
    errors: list[str]
    deep_reference_ids: list[str]
    report: dict[str, Any]
    retry_log: list[str]


class WorkflowService:
    def __init__(self) -> None:
        self.literature = LiteratureClient()
        self.graph = self._build_graph()

    def _build_graph(self):
        builder = StateGraph(WorkflowState)
        sequence = ["agent6_plan", "agent1_extract", "agent2_map", "agent3_search",
                    "agent4_initial_review", "agent6_route_deep_review"]
        for name in sequence + ["agent3_fetch_fulltext", "agent5_deep_review", "agent6_finalize"]:
            builder.add_node(name, getattr(self, name))
        builder.add_edge(START, sequence[0])
        for first, second in zip(sequence, sequence[1:]):
            builder.add_edge(first, second)
        builder.add_conditional_edges(sequence[-1], self._route_after_initial,
                                      {"deep": "agent3_fetch_fulltext", "finalize": "agent6_finalize"})
        builder.add_edge("agent3_fetch_fulltext", "agent5_deep_review")
        builder.add_edge("agent5_deep_review", "agent6_finalize")
        builder.add_edge("agent6_finalize", END)
        return builder.compile()

    async def run(self, input_path: str | Path, threshold: float | None = None) -> dict[str, Any]:
        path, threshold = self._validate_run(input_path, threshold)
        state = await self.graph.ainvoke({"input_path": str(path), "threshold": threshold})
        return state["report"]

    def _validate_run(self, input_path: str | Path, threshold: float | None) -> tuple[Path, float]:
        path = Path(input_path).resolve()
        if not path.is_file():
            raise FileNotFoundError(f"找不到输入文件：{path}")
        if path.suffix.lower() not in {".docx", ".doc", ".tex", ".latex"}:
            raise ValueError("仅支持 .docx、.doc、.tex、.latex 文件。")
        threshold = settings.confidence_threshold if threshold is None else float(threshold)
        if not math.isfinite(threshold) or not 0 <= threshold <= 1:
            raise ValueError("threshold 必须介于 0 和 1 之间。")
        return path, threshold

    async def stream(self, input_path: str | Path, threshold: float | None = None) -> AsyncIterator[dict[str, Any]]:
        """Yield node start/end events and the final report as the graph runs."""
        path, threshold = self._validate_run(input_path, threshold)
        steps = {"agent6_plan", "agent1_extract", "agent2_map", "agent3_search",
                 "agent4_initial_review", "agent6_route_deep_review", "agent3_fetch_fulltext",
                 "agent5_deep_review", "agent6_finalize"}
        async for event in self.graph.astream_events(
            {"input_path": str(path), "threshold": threshold}, version="v2"
        ):
            metadata = event.get("metadata", {})
            step = metadata.get("langgraph_node")
            is_custom_progress = event.get("event") == "on_custom_event" and event.get("name") == "reference_normalization_progress"
            if is_custom_progress:
                yield {"type": "step_progress", "step": "agent1_extract", "progress": event.get("data", {})}
                continue
            if step not in steps or event.get("name") != step:
                continue
            if event.get("event") == "on_chain_start":
                yield {"type": "step_start", "step": step}
            elif event.get("event") == "on_chain_end":
                output = event.get("data", {}).get("output", {})
                result = output.get(step, output) if isinstance(output, dict) else output
                yield {"type": "step_end", "step": step, "output": result}
                if step == "agent6_finalize" and isinstance(result, dict) and result.get("report"):
                    yield {"type": "final", "report": result["report"]}

    async def agent6_plan(self, state: WorkflowState) -> dict[str, Any]:
        # Fail once before parsing, searching and charging dozens of doomed LLM calls.
        await check_llm()
        return {"task_id": str(uuid.uuid4()), "started_at": _now(), "warnings": [], "errors": [], "lookups": {}}

    async def agent1_extract(self, state: WorkflowState, config=None) -> dict[str, Any]:
        input_path = Path(state["input_path"])
        is_word = input_path.suffix.lower() in {".docx", ".doc"}
        loaded = load_document(input_path, parse_references=not is_word)
        async def report_progress(completed: int, total: int, status: str):
            if config is not None:
                await adispatch_custom_event(
                    "reference_normalization_progress",
                    {"completed": completed, "total": total, "status": status},
                    config=config,
                )
        if is_word:
            refs, warnings = await extract_docx_references_with_llm(loaded.full_text, progress=report_progress)
        else:
            refs, warnings = await normalize_references(loaded.bibliography_entries, progress=report_progress)
        refs = [ReferenceItem.model_validate(ref).model_dump(exclude_none=True) for ref in refs]
        if not refs:
            warnings.append("未识别到参考文献，后续无法进行逐条文献支持性评估。")
        return {"source_name": loaded.source_name, "main_text": loaded.main_text,
                "references": refs, "warnings": loaded.warnings + warnings}

    async def agent2_map(self, state: WorkflowState) -> dict[str, Any]:
        pairs, unmatched, warnings = await map_citations(state["main_text"], state["references"])
        return {"pairs": [ClaimPair.model_validate(p).model_dump() for p in pairs], "unmatched_references": unmatched,
                "warnings": state["warnings"] + warnings}

    async def agent3_search(self, state: WorkflowState) -> dict[str, Any]:
        refs = state.get("references", [])
        warnings = list(state.get("warnings", []))
        if not refs:
            return {"lookups": {}, "warnings": warnings}
        try:
            await self.literature.prepare()
        except MCPFailure as exc:
            warnings.append(f"Agent3 MCP 工具连接失败（已重试）：{exc}")
            return {"lookups": {r["reference_id"]: normalize_metadata(
                {"found": False, "status": exc.status, "error": str(exc)[:1000]}) for r in refs}, "warnings": warnings}
        sem = asyncio.Semaphore(max(1, settings.max_lookup_concurrency))
        async def search(ref):
            async with sem:
                try:
                    raw = await self.literature.call("search_literature", {
                        "query": (ref.get("title") or ref["original_reference"])[:1400], "doi": ref.get("doi", ""),
                        "source_url": ref.get("url", ""), "arxiv_id": ref.get("arxiv_id", ""),
                        "authors": ref.get("authors", []), "year": ref.get("year")})
                    result = normalize_metadata(raw)
                except Exception as exc:
                    result = normalize_metadata({"found": False, "status": getattr(exc, "status", "invalid_response"), "error": str(exc)[:1000]})
                return ref["reference_id"], result
        lookups = dict(await asyncio.gather(*(search(ref) for ref in refs)))
        for ref_id, result in lookups.items():
            if result["status"] in LOOKUP_FAILURES or result.get("provider_errors"):
                warnings.append(f"Agent3 {ref_id}: {result['status']}；{result.get('error') or '; '.join(result.get('provider_errors', []))}")
        return {"lookups": lookups, "warnings": warnings}

    async def _review(self, pair: dict[str, Any], lookup: dict[str, Any], *, deep: bool) -> dict[str, Any]:
        basis = "retrieved_full_text" if deep else "metadata_and_abstract"
        evidence = lookup.get("full_text", "")[:settings.max_fulltext_chars] if deep else lookup.get("abstract", "")
        field = "supporting_passage" if deep else "evidence_from_abstract"
        payload = {"judgmental_statement": pair["statement_original"], "paper_title": lookup.get("title", ""),
                   "doi": lookup.get("doi", ""), "evidence_text": evidence}
        try:
            raw = await ainvoke_json(DEEP_REVIEW_SYSTEM if deep else INITIAL_REVIEW_SYSTEM,
                                     json.dumps(payload, ensure_ascii=False))
        except Exception as exc:
            return _unavailable("model_error", f"模型调用失败，无法判断支持关系：{str(exc)[:400]}", basis, error=True)
        try:
            parsed = ReviewPayload.model_validate(raw)
            passage = getattr(parsed, field)
            compact = lambda text: re.sub(r"\s+", "", text).casefold()
            if passage and compact(passage) not in compact(evidence):
                raise ValueError("模型引用的证据片段不在提供的原文中")
            if parsed.verdict in {"supports", "partially_supports"} and not passage:
                raise ValueError("支持判断缺少可定位的证据片段")
            if parsed.verdict in {"does_not_support", "uncertain"} and parsed.confidence > settings.confidence_threshold:
                raise ValueError("支持置信度与 verdict 不一致")
            return Review(**parsed.model_dump(), status="completed", basis=basis).model_dump()
        except Exception as exc:
            return _unavailable("invalid_response", f"审核 JSON 或证据校验失败：{str(exc)[:400]}", basis, error=True)

    async def _retry_failed_reviews(self, pairs: list[dict[str, Any]], lookups: dict[str, dict[str, Any]],
                                    review_key: str, *, deep: bool) -> tuple[list[dict[str, Any]], list[str]]:
        """Retry only review items that failed; successful items are never rerun."""
        updated = [dict(pair) for pair in pairs]
        retry_log: list[str] = []
        retryable = {"model_error", "invalid_response"}
        for attempt in range(1, max(0, settings.failed_node_retries) + 1):
            targets = [index for index, pair in enumerate(updated)
                       if (pair.get(review_key) or {}).get("failure_reason") in retryable]
            if not targets:
                break
            await asyncio.sleep(min(8, settings.failed_node_retry_delay_seconds * attempt))
            # Keep retries deliberately serial: a provider timeout should not
            # be amplified by retrying all failed requests concurrently.
            for index in targets:
                pair = dict(updated[index])
                result = await self._review(pair, lookups.get(pair["reference_id"], {}), deep=deep)
                pair[review_key] = result
                updated[index] = pair
                retry_log.append(f"{pair['pair_id']} {review_key} retry {attempt}: {result.get('status')}")
        return updated, retry_log

    async def agent4_initial_review(self, state: WorkflowState) -> dict[str, Any]:
        sem = asyncio.Semaphore(max(1, settings.max_concurrent_llm_calls))
        async def review(pair):
            pair = dict(pair)
            lookup = state["lookups"].get(pair["reference_id"], {})
            if lookup.get("status") in LOOKUP_FAILURES:
                result = _unavailable("literature_unavailable", f"文献获取失败（{lookup.get('status')}），无法核验，需人工复核；这不表示文献不支持该判断。", "metadata_and_abstract")
            elif not lookup.get("abstract", "").strip():
                result = _unavailable("abstract_unavailable", "未取得摘要，无法进行初审；将通过 Agent3 尝试获取全文。", "metadata_and_abstract")
            else:
                async with sem:
                    result = await self._review(pair, lookup, deep=False)
            pair["initial_review"] = result
            return pair
        pairs = list(await asyncio.gather(*(review(p) for p in state.get("pairs", []))))
        pairs, retry_log = await self._retry_failed_reviews(pairs, state["lookups"], "initial_review", deep=False)
        return {"pairs": pairs, "retry_log": state.get("retry_log", []) + retry_log}

    async def agent6_route_deep_review(self, state: WorkflowState) -> dict[str, Any]:
        ids = [p["reference_id"] for p in state.get("pairs", [])
               if _needs_deep(p, state["lookups"].get(p["reference_id"], {}), state["threshold"])]
        return {"deep_reference_ids": list(dict.fromkeys(ids))}

    def _route_after_initial(self, state: WorkflowState) -> str:
        return "deep" if state.get("deep_reference_ids") else "finalize"

    async def agent3_fetch_fulltext(self, state: WorkflowState) -> dict[str, Any]:
        lookups = {key: dict(value) for key, value in state["lookups"].items()}
        refs = {ref["reference_id"]: ref for ref in state["references"]}
        warnings = list(state.get("warnings", []))
        sem = asyncio.Semaphore(max(1, settings.max_lookup_concurrency))
        async def fetch(ref_id):
            lookup, ref = lookups[ref_id], refs[ref_id]
            async with sem:
                try:
                    result = await self.literature.call("fetch_full_text", {
                        "doi": lookup.get("doi") or ref.get("doi", ""), "pdf_url": lookup.get("pdf_url", ""),
                        "query": lookup.get("title") or ref.get("title") or ref["original_reference"],
                        "source_url": ref.get("url", ""), "arxiv_id": ref.get("arxiv_id", ""),
                        "authors": ref.get("authors", []), "year": ref.get("year")})
                    if not isinstance(result.get("full_text_available"), bool) or not isinstance(result.get("full_text", ""), str):
                        raise ValueError("Invalid full-text response")
                    for key in ("full_text_available", "full_text_status", "full_text", "word_count", "full_text_truncated", "full_text_error"):
                        if key in result:
                            lookup[key] = result[key]
                    for key in ("pdf_url", "abstract"):
                        if not lookup.get(key) and result.get(key):
                            lookup[key] = result[key]
                except Exception as exc:
                    lookup.update(full_text_available=False, full_text_status=getattr(exc, "status", "invalid_response"),
                                  full_text_error=str(exc)[:1000], full_text="")
                if not lookup.get("full_text_available"):
                    warnings.append(f"Agent3 {ref_id} 全文获取失败：{lookup.get('full_text_status')}；{lookup.get('full_text_error', '')}")
        await asyncio.gather(*(fetch(ref_id) for ref_id in state.get("deep_reference_ids", [])))
        return {"lookups": lookups, "warnings": warnings}

    async def agent5_deep_review(self, state: WorkflowState) -> dict[str, Any]:
        sem = asyncio.Semaphore(max(1, settings.max_concurrent_llm_calls))
        async def review(pair):
            pair = dict(pair)
            lookup = state["lookups"].get(pair["reference_id"], {})
            if not _needs_deep(pair, lookup, state["threshold"]):
                return pair
            if not lookup.get("full_text_available") or len(lookup.get("full_text", "")) < settings.min_fulltext_chars:
                result = _unavailable("full_text_unavailable", f"已通过 Agent3 尝试获取全文，但无足够可解析文本（{lookup.get('full_text_status')}）；无法完成二次核验，需人工复核。", "full_text_unavailable")
            else:
                async with sem:
                    result = await self._review(pair, lookup, deep=True)
            pair["deep_review"] = result
            return pair
        pairs = list(await asyncio.gather(*(review(p) for p in state.get("pairs", []))))
        pairs, retry_log = await self._retry_failed_reviews(pairs, state["lookups"], "deep_review", deep=True)
        return {"pairs": pairs, "retry_log": state.get("retry_log", []) + retry_log}

    async def agent6_finalize(self, state: WorkflowState) -> dict[str, Any]:
        threshold = state["threshold"]
        warnings = list(state.get("warnings", []))
        if state.get("retry_log"):
            warnings.append(f"模型失败项已独立重试 {len(state['retry_log'])} 次；详见步骤输出。")
        references = [{**ref, "literature_metadata": normalize_metadata(state["lookups"].get(ref["reference_id"], {}))}
                      for ref in state.get("references", [])]
        final_pairs = []
        for pair in state.get("pairs", []):
            p = dict(pair)
            lookup = state["lookups"].get(p["reference_id"], {})
            initial, deep = p["initial_review"], p.get("deep_review")
            winning = initial if _passed(initial, threshold) else deep if _passed(deep, threshold) else None
            latest = deep or initial
            manual = winning is None and latest["status"] != "completed"
            selected = winning or latest
            stage = "manual_required" if manual else "deep" if selected is deep else "preliminary"
            p["final_decision"] = {
                "passed": winning is not None,
                "passed_by": "initial_review" if winning is initial else "deep_review" if winning else "none",
                "threshold": threshold, "effective_confidence": selected.get("confidence"),
                "explanation": selected["rationale"], "review_stage": stage,
                "reason_code": "supported" if winning else selected.get("failure_reason") if manual else "below_threshold",
                "rule": "完成的初审或全文复审支持分数 > threshold 且 verdict 为支持/部分支持时通过；无法评估则需人工复核。",
            }
            p["standardized_response"] = {**p["standardized_response"],
                **{key: lookup.get(key, "") for key in ("doi", "abstract", "access_status", "pdf_url")},
                "access_note": "close-source：仅可用摘要，未获取全文时不声称完成全文复核。" if lookup.get("status") == "close_source" else ""}
            if manual:
                warnings.append(f"{p['pair_id']} 需人工复核：{p['final_decision']['reason_code']}")
            final_pairs.append(p)
        passed = sum(p["final_decision"]["passed"] for p in final_pairs)
        manual_count = sum(p["final_decision"]["review_stage"] == "manual_required" for p in final_pairs)
        if not final_pairs:
            warnings.append("未识别到可评估引用对，请检查参考文献及正文引用格式。")
        errors = state.get("errors", [])
        report = ReportSchema.model_validate({
            "schema_version": "1.2.0", "task_id": state["task_id"], "source_file": state["source_name"],
            "status": "failed" if final_pairs and all(p["initial_review"]["status"] != "completed" and (p.get("deep_review") or {}).get("status") != "completed" for p in final_pairs)
                      else "completed_with_warnings" if warnings or errors else "completed", "threshold": threshold,
            "model": settings.llm_model, "started_at": state["started_at"], "completed_at": _now(),
            "statistics": {"references_total": len(references), "pairs_total": len(final_pairs), "passed": passed,
                "failed": len(final_pairs) - passed, "manual_required": manual_count,
                "rejected": len(final_pairs) - passed - manual_count,
                "failure_reasons": dict(Counter(p["final_decision"]["reason_code"] for p in final_pairs if not p["final_decision"]["passed"])),
                "deep_reviewed": sum((p.get("deep_review") or {}).get("status") == "completed" for p in final_pairs),
                "unmatched_references": len(state.get("unmatched_references", []))},
            "references": references, "pairs": final_pairs, "unmatched_references": state.get("unmatched_references", []),
            "warnings": list(dict.fromkeys(warnings)), "errors": errors,
        }).model_dump(mode="json")
        json.dumps(report, ensure_ascii=False, allow_nan=False)
        return {"report": report}
