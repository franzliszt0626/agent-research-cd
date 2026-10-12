"""MCP literature tools. Network access is intentionally isolated in this module.
The LangGraph workflow calls these tools only through Agent 3 / MCP.
"""
from __future__ import annotations

import asyncio
import html
import os
import re
from difflib import SequenceMatcher
from typing import Any
from urllib.parse import quote

import pymupdf as fitz
import httpx
from bs4 import BeautifulSoup
from mcp.server.fastmcp import FastMCP
from dotenv import load_dotenv
from .source_fetch import fetch_source, arxiv_identifier

load_dotenv()

mcp = FastMCP(
    "agent-research-cd-literature",
    host=os.getenv("MCP_HOST", "0.0.0.0"),
    port=int(os.getenv("MCP_PORT", "8001")),
    streamable_http_path=os.getenv("MCP_PATH", "/mcp"),
    stateless_http=True,
    json_response=True,
)
USER_AGENT = "AgentResearchCD/1.0 (academic metadata retrieval)"
MAX_PDF_BYTES = int(os.getenv("MAX_PDF_BYTES", "25000000"))
MAX_FULLTEXT_CHARS = int(os.getenv("MAX_FULLTEXT_CHARS", "40000"))
UNPAYWALL_EMAIL = os.getenv("UNPAYWALL_EMAIL", "")
CROSSREF_MAILTO = os.getenv("CROSSREF_MAILTO", "")
OPENALEX_MAILTO = os.getenv("OPENALEX_MAILTO", "")


def _abstract_from_inverted_index(index: Any) -> str:
    if not isinstance(index, dict):
        return ""
    positioned: list[tuple[int, str]] = []
    for word, positions in index.items():
        if isinstance(positions, list):
            for p in positions:
                if isinstance(p, int):
                    positioned.append((p, word))
    positioned.sort(key=lambda x: x[0])
    return " ".join(word for _, word in positioned).strip()


def _crossref_abstract(value: str) -> str:
    if not value:
        return ""
    return re.sub(r"\s+", " ", BeautifulSoup(value, "html.parser").get_text(" ", strip=True)).strip()


def _doi_from_text(value: str) -> str:
    m = re.search(r"10\.\d{4,9}/[-._;()/:A-Z0-9]+", value or "", re.I)
    return m.group(0).rstrip(".,;)") if m else ""

async def _get_json(client: httpx.AsyncClient, url: str, params: dict[str, Any] | None = None) -> dict[str, Any] | None:
    for attempt in range(3):
        response = await client.get(url, params=params)
        if response.status_code not in {429, 500, 502, 503, 504} or attempt == 2:
            break
        retry_after = response.headers.get("retry-after", "")
        delay = min(10, float(retry_after)) if retry_after.isdigit() else 1.5 * 2 ** attempt
        await asyncio.sleep(delay)
    if response.status_code == 404:
        return None
    response.raise_for_status()
    data = response.json()
    if not isinstance(data, dict):
        raise ValueError("Metadata provider returned non-object JSON")
    return data


def _title_score(query: str, title: str) -> float:
    clean = lambda value: re.sub(r"[^\w]", "", value.casefold())
    left, right = clean(query), clean(title)
    if not left or not right:
        return 0.0
    return SequenceMatcher(None, left, right).ratio()


def _select_candidate(items: list[dict[str, Any]], query: str, doi: str, *, crossref: bool = False,
                      authors: list[str] | None = None, year: int | None = None):
    def title(item):
        value = item.get("title") or ""
        return value[0] if isinstance(value, list) and value else value
    if doi:
        return next((item for item in items if _doi_from_text(str(item.get("DOI" if crossref else "doi") or "")).lower() == doi.lower()), None)
    def identity_matches(item):
        candidate_year = (((item.get("published") or {}).get("date-parts") or [[None]])[0][0]
                          if crossref else item.get("publication_year"))
        if year and candidate_year and abs(int(candidate_year) - year) > 2:
            return False
        if authors:
            if crossref:
                candidate_names = [str(a.get("family", "")) + " " + str(a.get("given", "")) for a in item.get("author", [])]
            else:
                candidate_names = [str((a.get("author") or {}).get("display_name", "")) for a in item.get("authorships", [])]
            tokens = {word for word in re.findall(r"[\w-]+", authors[0].casefold()) if len(word) > 1}
            if tokens and candidate_names and not any(tokens & set(re.findall(r"[\w-]+", name.casefold())) for name in candidate_names):
                return False
        return True
    ranked = sorted((item for item in items if identity_matches(item)), key=lambda item: _title_score(query, title(item)), reverse=True)
    # Similar method names (LoRA vs QA-LoRA) are different papers, not spelling variants.
    return ranked[0] if ranked and _title_score(query, title(ranked[0])) >= 0.94 else None


def _openalex_result(item: dict[str, Any]) -> dict[str, Any]:
    best = item.get("best_oa_location") or {}
    # In some API responses best_oa_location is nested under open_access/locations.
    if not best:
        oa = item.get("open_access") or {}
        best = oa.get("best_oa_location") or {}
    primary = item.get("primary_location") or {}
    if not isinstance(best, dict):
        best = {}
    if not isinstance(primary, dict):
        primary = {}
    source = primary.get("source") or {}
    if not isinstance(source, dict):
        source = {}
    oa_record = item.get("open_access") or {}
    is_oa = oa_record.get("is_oa")  # Unknown is not closed-source.
    if not best:
        for location in item.get("locations", []) or []:
            if not isinstance(location, dict):
                continue
            if location.get("is_oa") or ((location.get("source") or {}).get("is_oa")):
                best = location
                break
    authors = []
    for entry in item.get("authorships", []) or []:
        author = (entry.get("author") or {}).get("display_name")
        if author:
            authors.append(author)
    doi = _doi_from_text(str(item.get("doi") or ""))
    pdf_url = best.get("pdf_url") or ""
    landing = best.get("landing_page_url") or primary.get("landing_page_url") or ""
    abstract = _abstract_from_inverted_index(item.get("abstract_inverted_index"))
    if is_oa:
        status = "open-access" if pdf_url else "open-access-metadata-only"
    elif "is_oa" in oa_record and oa_record.get("is_oa") is False:
        status = "closed-source"
    else:
        status = "access-undetermined"
    return {
        "found": True,
        "title": item.get("title") or "",
        "authors": authors[:30],
        "year": item.get("publication_year"),
        "doi": doi,
        "venue": source.get("display_name") or "",
        "abstract": abstract,
        "is_open_access": is_oa,
        "access_status": status,
        "pdf_url": pdf_url,
        "landing_page_url": landing,
        "source": "OpenAlex",
        "openalex_id": item.get("id") or "",
        "cited_by_count": item.get("cited_by_count"),
    }


async def _search_openalex(query: str, doi: str = "", authors: list[str] | None = None, year: int | None = None) -> dict[str, Any] | None:
    async with httpx.AsyncClient(timeout=25, headers={"User-Agent": USER_AGENT}) as client:
        if doi:
            url = "https://api.openalex.org/works/https://doi.org/" + quote(doi, safe="")
            data = await _get_json(client, url)
            if data and data.get("id"):
                return _openalex_result(data)
        params: dict[str, Any] = {"search": query, "per-page": 5}
        if OPENALEX_MAILTO:
            params["mailto"] = OPENALEX_MAILTO
        data = await _get_json(client, "https://api.openalex.org/works", params)
        results = (data or {}).get("results") or []
        if not results:
            return None
        best = _select_candidate(results, query, doi, authors=authors, year=year)
        return _openalex_result(best) if best else None


async def _search_crossref(query: str, doi: str = "", authors: list[str] | None = None, year: int | None = None) -> dict[str, Any] | None:
    async with httpx.AsyncClient(timeout=25, headers={"User-Agent": USER_AGENT}) as client:
        params: dict[str, Any] = {"rows": 5}
        if CROSSREF_MAILTO:
            params["mailto"] = CROSSREF_MAILTO
        if doi:
            data = await _get_json(client, "https://api.crossref.org/works/" + quote(doi, safe=""), {k: v for k, v in params.items() if k != "rows"})
            item = (data or {}).get("message")
            if isinstance(item, dict):
                items = [item]
            else:
                items = []
        else:
            params["query.bibliographic"] = query
            data = await _get_json(client, "https://api.crossref.org/works", params)
            items = ((data or {}).get("message") or {}).get("items") or []
        if not items:
            return None
        item = _select_candidate(items, query, doi, crossref=True, authors=authors, year=year)
        if item is None:
            return None
        authors = []
        for author in item.get("author", []) or []:
            name = " ".join(x for x in [author.get("given"), author.get("family")] if x)
            if name:
                authors.append(name)
        title_list = item.get("title") or []
        container_list = item.get("container-title") or []
        date_parts = ((item.get("published") or {}).get("date-parts") or [[None]])[0]
        year = date_parts[0] if date_parts else None
        doi_val = _doi_from_text(str(item.get("DOI") or ""))
        abstract = _crossref_abstract(str(item.get("abstract") or ""))
        return {
            "found": True,
            "title": title_list[0] if title_list else "",
            "authors": authors,
            "year": year,
            "doi": doi_val,
            "venue": container_list[0] if container_list else "",
            "abstract": abstract,
            "is_open_access": None,
            "access_status": "access-undetermined",
            "pdf_url": "",
            "landing_page_url": ("https://doi.org/" + doi_val) if doi_val else "",
            "source": "Crossref",
            "cited_by_count": item.get("is-referenced-by-count"),
        }


async def _unpaywall_lookup(doi: str) -> dict[str, Any] | None:
    if not UNPAYWALL_EMAIL or not doi:
        return None
    async with httpx.AsyncClient(timeout=20, headers={"User-Agent": USER_AGENT}) as client:
        data = await _get_json(client, f"https://api.unpaywall.org/v2/{quote(doi, safe='')}?email={quote(UNPAYWALL_EMAIL, safe='@')}")
    if not data:
        return None
    best = data.get("best_oa_location") or {}
    return {
        "is_open_access": bool(data.get("is_oa")),
        "access_status": "open-access" if data.get("is_oa") else "closed-source",
        "pdf_url": (best or {}).get("url_for_pdf") or "",
        "landing_page_url": (best or {}).get("url") or "",
        "oa_status": data.get("oa_status") or "",
    }


@mcp.tool()
async def search_literature(query: str, doi: str = "", source_url: str = "", arxiv_id: str = "",
                            authors: list[str] | None = None, year: int | None = None) -> dict[str, Any]:
    """Search academic metadata using OpenAlex and Crossref. Returns DOI, abstract, OA status, and possible PDF URL."""
    clean_doi = _doi_from_text(doi or query)
    search_query = (query or "").strip()
    if not search_query and clean_doi:
        search_query = clean_doi
    if len(search_query) > 1500:
        search_query = search_query[:1500]
    source_errors = []
    # Exact source identifiers take precedence over fuzzy title search.
    identifier = arxiv_identifier(arxiv_id) or arxiv_identifier(source_url)
    if source_url or identifier:
        try:
            exact = await fetch_source(source_url, identifier)
            if exact:
                exact["query"] = search_query
                return exact
        except Exception as exc:
            source_errors.append(f"Reference URL: {type(exc).__name__}: {exc}")
    # Query Crossref only when OpenAlex lacks a matching paper, abstract, or DOI.
    # This avoids a needless burst of Crossref calls (and 429s) for every bibliography.
    outcomes = []
    try:
        alex = await _search_openalex(search_query, clean_doi, authors, year)
        outcomes.append(alex)
    except Exception as exc:
        alex = None
        outcomes.append(exc)
    cross = None
    if not alex or not alex.get("abstract") or not alex.get("doi"):
        try:
            cross = await _search_crossref(search_query, clean_doi, authors, year)
            outcomes.append(cross)
        except Exception as exc:
            outcomes.append(exc)
    provider_errors = source_errors + [f"{name}: {type(value).__name__}: {value}" for name, value in zip(("OpenAlex", "Crossref"), outcomes) if isinstance(value, Exception)]
    result = alex or cross
    if alex and cross:
        same_work = ((alex.get("doi") and cross.get("doi") and alex["doi"].lower() == cross["doi"].lower())
                     or (not (alex.get("doi") and cross.get("doi")) and _title_score(alex["title"], cross["title"]) >= 0.95))
        if same_work:
            for key in ("abstract", "doi", "venue", "landing_page_url"):
                if not result.get(key) and cross.get(key):
                    result[key] = cross[key]
            result["crossref_found"] = True
    if result is None:
        failure = "network_timeout" if any(isinstance(v, httpx.TimeoutException) for v in outcomes) else "network_error"
        return {"found": False, "status": failure if provider_errors else "not_found", "provider_errors": provider_errors,
                "query": search_query, "doi": clean_doi, "access_status": "not-found", "abstract": "", "pdf_url": "", "source": "OpenAlex/Crossref"}
    doi_val = result.get("doi") or clean_doi
    try:
        unpaywall = await _unpaywall_lookup(doi_val) if doi_val else None
    except Exception as exc:
        unpaywall = None
        provider_errors.append(f"Unpaywall: {type(exc).__name__}: {exc}")
    if unpaywall:
        if unpaywall.get("pdf_url") and not result.get("pdf_url"):
            result["pdf_url"] = unpaywall["pdf_url"]
        if unpaywall.get("landing_page_url") and not result.get("landing_page_url"):
            result["landing_page_url"] = unpaywall["landing_page_url"]
        result["is_open_access"] = unpaywall.get("is_open_access")
        result["access_status"] = unpaywall.get("access_status") or result.get("access_status")
        result["unpaywall_oa_status"] = unpaywall.get("oa_status", "")
    result["query"] = search_query
    result["doi"] = doi_val or ""
    result["provider_errors"] = provider_errors
    if result.get("is_open_access") is False:
        result["access_status"] = "closed-source"
    return result


async def _fulltext_from_pdf(pdf_url: str) -> dict[str, Any]:
    if not pdf_url or not pdf_url.startswith(("http://", "https://")):
        return {"full_text_available": False, "full_text": "", "full_text_status": "no-pdf-url", "word_count": 0}
    try:
        async with httpx.AsyncClient(timeout=40, follow_redirects=True, headers={"User-Agent": USER_AGENT}) as client:
            async with client.stream("GET", pdf_url) as response:
                if response.status_code >= 400:
                    return {"full_text_available": False, "full_text": "", "full_text_status": f"pdf-download-http-{response.status_code}", "word_count": 0}
                content_length = response.headers.get("content-length")
                if content_length and int(content_length) > MAX_PDF_BYTES:
                    return {"full_text_available": False, "full_text": "", "full_text_status": "pdf-too-large", "word_count": 0}
                chunks = bytearray()
                async for chunk in response.aiter_bytes():
                    chunks.extend(chunk)
                    if len(chunks) > MAX_PDF_BYTES:
                        return {"full_text_available": False, "full_text": "", "full_text_status": "pdf-too-large", "word_count": 0}
        if not bytes(chunks).startswith(b"%PDF"):
            return {"full_text_available": False, "full_text": "", "full_text_status": "response-not-pdf", "word_count": 0}
        doc = fitz.open(stream=bytes(chunks), filetype="pdf")
        pages = []
        truncated = False
        for page in doc:
            pages.append(page.get_text("text"))
            if sum(len(x) for x in pages) >= MAX_FULLTEXT_CHARS:
                truncated = True
                break
        doc.close()
        text = re.sub(r"\n{3,}", "\n\n", "\n".join(pages)).strip()[:MAX_FULLTEXT_CHARS]
        words = len(re.findall(r"\b\w+\b", text))
        if words < 250:
            return {"full_text_available": False, "full_text": text, "full_text_status": "pdf-text-too-short-or-scanned", "word_count": words}
        return {"full_text_available": True, "full_text": text, "full_text_status": "retrieved", "word_count": words, "full_text_truncated": truncated}
    except (httpx.HTTPError, ValueError, fitz.FileDataError, RuntimeError) as exc:
        return {"full_text_available": False, "full_text": "", "full_text_status": f"pdf-read-failed: {str(exc)[:180]}", "word_count": 0}


@mcp.tool()
async def fetch_full_text(doi: str = "", pdf_url: str = "", query: str = "", source_url: str = "",
                          arxiv_id: str = "", authors: list[str] | None = None, year: int | None = None) -> dict[str, Any]:
    """Try to retrieve and extract open-access PDF full text. If paywalled/unavailable, return access status and abstract."""
    metadata: dict[str, Any] | None = None
    clean_doi = _doi_from_text(doi or query)
    source_error = ""
    if (source_url or arxiv_id) and not pdf_url:
        try:
            metadata = await fetch_source(source_url, arxiv_id, include_text=True)
            if metadata:
                pdf_url = metadata.get("pdf_url") or pdf_url
                page_text = metadata.pop("_page_text", "")
                # An arXiv abstract landing page is not the paper's full text.
                if (not (arxiv_id or arxiv_identifier(source_url)) and not pdf_url
                        and not metadata.get("doi") and not metadata.get("abstract") and len(page_text) >= 1200):
                    return {**metadata, "full_text_available": True, "full_text": page_text[:MAX_FULLTEXT_CHARS],
                            "full_text_status": "retrieved-webpage", "full_text_truncated": len(page_text) > MAX_FULLTEXT_CHARS,
                            "word_count": len(re.findall(r"\w+", page_text))}
        except Exception as exc:
            source_error = f"Reference URL: {type(exc).__name__}: {exc}"
    if not pdf_url:
        search_query = query or clean_doi or doi
        if search_query:
            metadata = await search_literature(search_query, clean_doi, source_url, arxiv_id, authors, year)
            if not clean_doi:
                clean_doi = metadata.get("doi", "")
            if not pdf_url:
                pdf_url = metadata.get("pdf_url", "")
    metadata = metadata or {"found": False, "doi": clean_doi, "abstract": "", "access_status": "access-undetermined", "source": ""}
    if pdf_url:
        result = await _fulltext_from_pdf(pdf_url)
        return {
            **metadata,
            **result,
            "doi": clean_doi or metadata.get("doi", ""),
            "pdf_url": pdf_url,
            "access_status": "open-access" if result.get("full_text_available") else metadata.get("access_status", "fulltext-unavailable"),
        }
    status = metadata.get("access_status", "access-undetermined")
    if status not in {"closed-source", "not-found"}:
        status = "fulltext-unavailable"
    return {
        **metadata,
        "doi": clean_doi or metadata.get("doi", ""),
        "pdf_url": "",
        "full_text": "",
        "full_text_available": False,
        "full_text_status": "closed-source-no-open-pdf" if status == "closed-source" else "no-open-pdf-found",
        "word_count": 0,
        "access_status": status,
        "full_text_error": source_error,
    }


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--transport", choices=["stdio", "streamable-http"], default="streamable-http")
    mcp.run(transport=parser.parse_args().transport)
