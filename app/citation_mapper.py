from __future__ import annotations

import re
import asyncio
import json
from typing import Any
from collections.abc import Awaitable, Callable

from .config import settings
from .llm import ainvoke_json
from .document_loader import GB_TYPE
from .schemas import ReferenceItem


def _year(raw: str) -> int | None:
    m = re.search(r"\b((?:19|20)\d{2})\b", raw)
    return int(m.group(1)) if m else None



def _bib_field(raw: str, field: str) -> str:
    """Extract a common BibTeX field with balanced braces; handles nested title braces."""
    match = re.search(r"\b" + re.escape(field) + r'\s*=\s*([{"])', raw, re.I)
    if not match:
        bare = re.search(r"\b" + re.escape(field) + r"\s*=\s*(\d+)\s*[,}]", raw, re.I)
        return bare.group(1) if bare else ""
    opening = match.group(1)
    start = match.end()
    if opening == '"':
        result = re.search(r'((?:[^"\\]|\\.)*)"', raw[start:])
        return result.group(1).replace(r'\"', '"').strip() if result else ""
    depth = 1
    i = start
    while i < len(raw) and depth:
        if raw[i] == "{":
            depth += 1
        elif raw[i] == "}":
            depth -= 1
        i += 1
    return raw[start:i - 1].strip() if depth == 0 else ""

def _fallback_reference(raw: str, index: int, bib_key: str = "", label: str = "") -> dict[str, Any]:
    doi_match = re.search(r"10\.\d{4,9}/[-._;()/:A-Z0-9]+", raw, re.I)
    year = _year(raw)
    normalized = re.sub(r"\s+", " ", raw).strip()
    title = ""
    venue = ""
    author_names: list[str] = []
    if raw.lstrip().startswith("@"):
        author_field = _bib_field(raw, "author")
        title = _bib_field(raw, "title")
        venue = _bib_field(raw, "journal") or _bib_field(raw, "booktitle") or _bib_field(raw, "publisher")
        year_field = _bib_field(raw, "year")
        doi_field = _bib_field(raw, "doi")
        year_match = re.search(r"(?:19|20)\d{2}", year_field)
        if year_match:
            year = int(year_match.group(0))
        if doi_field:
            doi_match = re.search(r"10\.\d{4,9}/[-._;()/:A-Z0-9]+", doi_field, re.I) or doi_match
        author_names = [x.strip() for x in re.split(r"\s+and\s+", author_field, flags=re.I) if x.strip()]
        citation_parts = []
        if author_names:
            citation_parts.append(", ".join(author_names))
        if year:
            citation_parts.append(f"({year})")
        if title:
            citation_parts.append(title.rstrip(" .") + ".")
        if venue:
            citation_parts.append(venue.rstrip(" .") + ".")
        doi_value = doi_match.group(0).rstrip(".,)") if doi_match else ""
        if doi_value:
            citation_parts.append(f"doi:{doi_value}")
        if citation_parts:
            normalized = " ".join(citation_parts)
    else:
        gb = re.search(GB_TYPE, raw)
        if gb:
            head = raw[:gb.start()]
            delimiter = re.search(r"[.。]\s+", head)
            if delimiter:
                authors_part = head[:delimiter.start()]
                title = head[delimiter.end():].strip()
                tail = raw[gb.end():].lstrip("./ ")
                # Publication year comes after title, never from the title or access date.
                pub_year = re.search(r"(?<!\d)((?:19|20)\d{2})(?!\d)", tail)
                year = int(pub_year.group(1)) if pub_year else None
                venue = tail[:pub_year.start()].strip(" .,(:") if pub_year else tail
                authors_part = re.sub(r",?\s*et al$|[,，]?\s*等$", "", authors_part, flags=re.I)
                author_names = [a.strip() for a in re.split(r"[,，;；]|\s+and\s+", authors_part) if a.strip()]
        # Heuristic: in author-year references, the title often follows the year delimiter.
        if year and not title:
            year_match = re.search(r"\b(?:19|20)\d{2}[a-z]?\s*[).,]?\s*", raw, re.I)
            if year_match:
                tail = raw[year_match.end():]
                title = re.split(r"\.\s+(?=[A-Z])|\n", tail, maxsplit=1)[0].strip(" .,:;")
        if not author_names:
            authors_part = re.split(r"\b(?:19|20)\d{2}\b", raw, maxsplit=1)[0].strip(" .,(;[]")
            author_names = [x.strip() for x in re.split(r"\s+;\s+|\s+and\s+|,\s+(?=[A-Z][a-z]+\s+[A-Z])", authors_part) if x.strip()]
    label_num = re.search(r"\d+", label)
    return {
        "reference_id": f"REF{index:04d}",
        "original_reference": raw,
        "bib_key": bib_key,
        "label": label,
        "authors": author_names[:12],
        "year": year,
        "title": title[:600],
        "venue": venue[:400],
        "doi": doi_match.group(0).rstrip(".,)") if doi_match else "",
        "arxiv_id": next(iter(re.findall(r"(?:arxiv\s*:\s*|arxiv\.org/(?:abs|pdf)/)(\d{4}\.\d{4,5}(?:v\d+)?)", raw, re.I)), ""),
        "url": (_bib_field(raw, "url") if raw.lstrip().startswith("@") else next(iter(re.findall(r"https?://[^\s{}]+", raw)), "")).rstrip(".,"),
        "access_date": next(iter(re.findall(r"\[((?:19|20)\d{2}-\d{2}-\d{2})\]", raw)), ""),
        "normalized_citation": normalized,
        "citation_number": int(label_num.group(0)) if label_num else index,
    }


def _normalize_reference_shape(item: dict[str, Any], source: dict[str, str], index: int) -> dict[str, Any]:
    fallback = _fallback_reference(source["raw"], index, source.get("bib_key", ""), source.get("label", ""))
    authors = item.get("authors", fallback["authors"])
    if isinstance(authors, str):
        authors = [a.strip() for a in re.split(r";|\band\b", authors) if a.strip()]
    if not isinstance(authors, list):
        authors = fallback["authors"]
    year = item.get("year", fallback["year"])
    try:
        year = int(year) if year not in (None, "", "null") else None
    except (ValueError, TypeError):
        year = fallback["year"]
    doi = str(item.get("doi") or fallback["doi"] or "").strip()
    doi_match = re.search(r"10\.\d{4,9}/[-._;()/:A-Z0-9]+", doi, re.I)
    if doi_match:
        doi = doi_match.group(0).rstrip(".,)")
    if doi and doi.casefold() not in source["raw"].casefold():
        doi = fallback["doi"]
    title = str(item.get("title") or fallback["title"] or "").strip()
    compact = lambda value: re.sub(r"[^\w]", "", value.casefold())
    if title and compact(title) not in compact(source["raw"]):
        title = fallback["title"]
    if year and str(year) not in source["raw"]:
        year = fallback["year"]
    return {
        **fallback,
        "authors": [str(a).strip() for a in authors if str(a).strip()][:20],
        "year": year,
        "title": title,
        "venue": str(item.get("venue") or fallback["venue"] or "").strip(),
        "doi": doi,
        "normalized_citation": str(item.get("normalized_citation") or fallback["normalized_citation"]).strip(),
        "bib_key": source.get("bib_key", ""),
        "label": source.get("label", ""),
        "citation_number": fallback["citation_number"],
    }


async def _normalize_references_serial_legacy(raw_entries: list[dict[str, str]]) -> tuple[list[dict[str, Any]], list[str]]:
    refs: list[dict[str, Any]] = []
    warnings: list[str] = []
    # Batch to avoid sending very long reference lists in one context window.
    for start in range(0, len(raw_entries), 25):
        batch = raw_entries[start:start + 25]
        prompt_entries = [
            {"index": start + i + 1, "raw": entry.get("raw", ""), "bib_key": entry.get("bib_key", ""), "label": entry.get("label", "")}
            for i, entry in enumerate(batch)
        ]
        try:
            result = await ainvoke_json(
                "你是学术参考文献解析器。严格只根据原始条目抽取元数据，不得补造不存在的字段。输出合法 JSON 对象，结构为 {\"references\":[{\"index\":1,\"authors\":[\"\"],\"year\":2022,\"title\":\"\",\"venue\":\"\",\"doi\":\"\",\"normalized_citation\":\"\"}]}。没有的信息用空字符串或 null。",
                "请标准化以下参考文献。保留输入顺序，index 必须原样返回。BibTeX key 和 numeric label 是线索，不是出版元数据。\n" + __import__("json").dumps(prompt_entries, ensure_ascii=False),
                retries=settings.llm_retries,
            )
            items = result.get("references", []) if isinstance(result, dict) else []
            by_index = {int(x.get("index")): x for x in items if isinstance(x, dict) and str(x.get("index", "")).isdigit()}
        except Exception as exc:
            by_index = {}
            warnings.append(f"部分参考文献未能通过模型标准化，已使用规则解析兜底：{exc}")
        for i, entry in enumerate(batch):
            index = start + i + 1
            try:
                ref = _normalize_reference_shape(by_index.get(index, {}), entry, index)
                ReferenceItem.model_validate(ref)
            except (ValueError, TypeError) as exc:
                warnings.append(f"Agent1 第 {index} 条结构校验失败，使用原文规则解析：{exc}")
                ref = _fallback_reference(entry['raw'], index, entry.get('bib_key', ''), entry.get('label', ''))
            refs.append(ref)
    return refs, warnings


async def _normalize_references_concurrent(
    raw_entries: list[dict[str, str]],
    progress: Callable[[int, int, str], Awaitable[None]] | None = None,
) -> tuple[list[dict[str, Any]], list[str]]:
    """Use concurrent LLM batches while preserving deterministic input order."""
    if not raw_entries:
        return [], []
    batch_size = max(1, settings.reference_normalization_batch_size)
    semaphore = asyncio.Semaphore(max(1, settings.reference_normalization_concurrency))
    system_prompt = (
        "Extract bibliographic metadata. Return only JSON with a references array. "
        "Preserve each index exactly. Use only text in that record; never invent. "
        "Missing fields must be null or empty."
    )

    async def run_batch(start: int, batch: list[dict[str, str]]):
        records = [{"index": start + offset + 1, "raw": entry.get("raw", ""),
                    "bib_key": entry.get("bib_key", ""), "label": entry.get("label", "")}
                   for offset, entry in enumerate(batch)]
        try:
            async with semaphore:
                result = await ainvoke_json(
                    system_prompt,
                    "records=" + json.dumps(records, ensure_ascii=False, separators=(",", ":")),
                    retries=settings.llm_retries,
                )
            items = result.get("references", []) if isinstance(result, dict) else []
            return ({int(item["index"]): item for item in items
                     if isinstance(item, dict) and str(item.get("index", "")).isdigit()}, "")
        except Exception as exc:
            return ({}, f"部分参考文献未能通过模型标准化，已使用规则解析兜底：{exc}")

    batches = [(start, raw_entries[start:start + batch_size])
               for start in range(0, len(raw_entries), batch_size)]
    async def completed_batch(start: int, batch: list[dict[str, str]]):
        items, warning = await run_batch(start, batch)
        if progress:
            await progress(start // batch_size + 1, len(batches), "error" if warning else "done")
        return items, warning

    results = await asyncio.gather(*(completed_batch(start, batch) for start, batch in batches))
    model_items: dict[int, dict[str, Any]] = {}
    warnings: list[str] = []
    for items, warning in results:
        model_items.update(items)
        if warning:
            warnings.append(warning)
    refs: list[dict[str, Any]] = []
    for index, entry in enumerate(raw_entries, 1):
        try:
            ref = _normalize_reference_shape(model_items.get(index, {}), entry, index)
            ReferenceItem.model_validate(ref)
        except (ValueError, TypeError) as exc:
            warnings.append(f"Agent1 entry {index} validation failed; using source fields: {exc}")
            ref = _fallback_reference(entry["raw"], index, entry.get("bib_key", ""), entry.get("label", ""))
        refs.append(ref)
    return refs, warnings


async def extract_docx_references_with_llm(
    document_text: str,
    progress: Callable[[int, int, str], Awaitable[None]] | None = None,
) -> tuple[list[dict[str, Any]], list[str]]:
    """Let the model identify Word bibliography boundaries and entries directly."""
    prompt = (
        "Read the complete Word document below and extract every bibliography entry. "
        "Ignore running headers, page numbers, acknowledgements, and body citations. "
        "Do not split or merge entries based on page layout; an entry can wrap across lines. "
        "Return only JSON: {\"references\":[{\"index\":1,\"original_reference\":\"...\","
        "\"authors\":[\"...\"],\"year\":2024,\"title\":\"...\",\"venue\":\"...\","
        "\"doi\":\"...\",\"url\":\"...\",\"normalized_citation\":\"...\"}]}. "
        "Preserve the document order. Use only text present in the document; never invent."
    )
    try:
        result = await ainvoke_json(prompt, "document=" + document_text, retries=settings.llm_retries)
        items = result.get("references", []) if isinstance(result, dict) else []
        refs: list[dict[str, Any]] = []
        for index, item in enumerate(items, 1):
            if not isinstance(item, dict) or not str(item.get("original_reference", "")).strip():
                continue
            raw = str(item["original_reference"]).strip()
            authors = item.get("authors", [])
            if isinstance(authors, str):
                authors = [authors]
            year = item.get("year")
            try:
                year = int(year) if year not in (None, "", "null") else None
            except (TypeError, ValueError):
                year = None
            ref = {
                "reference_id": f"REF{index:04d}", "original_reference": raw,
                "bib_key": "", "label": "", "authors": [str(a).strip() for a in authors if str(a).strip()][:20],
                "year": year, "title": str(item.get("title") or "").strip()[:600],
                "venue": str(item.get("venue") or "").strip()[:400],
                "doi": str(item.get("doi") or "").strip(), "arxiv_id": "",
                "url": str(item.get("url") or "").strip(), "access_date": "",
                "normalized_citation": str(item.get("normalized_citation") or raw).strip(),
                "citation_number": index,
            }
            try:
                ReferenceItem.model_validate(ref)
                refs.append(ref)
            except (ValueError, TypeError):
                continue
        if progress:
            await progress(1, 1, "done")
        return refs, []
    except Exception as exc:
        if progress:
            await progress(1, 1, "error")
        return [], [f"Agent1 Word 文献抽取模型调用失败：{exc}"]


normalize_references = _normalize_references_concurrent


def _clean_sentence(text: str) -> str:
    return re.sub(r"\s+", " ", text).strip(" \t\n;•")


def split_sentences(text: str) -> list[str]:
    text = re.sub(r"\n+", "\n", text)
    # Protect common abbreviations so a citation like "Pradhan et al. 2022" stays intact.
    protected: dict[str, str] = {}
    abbreviations = ["et al.", "e.g.", "i.e.", "Fig.", "Eq.", "Dr.", "Prof.", "vs.", "No."]
    for idx, abbreviation in enumerate(abbreviations):
        token = f"§ABBR{idx}§"
        pattern = re.compile(re.escape(abbreviation), re.I)
        if pattern.search(text):
            text = pattern.sub(token, text)
            protected[token] = abbreviation
    # Protect academic initials, e.g. "A. Smith".
    text = re.sub(r"\b([A-Z])\.\s+(?=[A-Z])", r"\1§INITIAL§ ", text)
    # Sentence-level chunks retain citation markers so the specific citation can be linked.
    chunks = re.split(r"(?<=[。！？])|(?<=[.!?])\s+(?=(?:[A-Z0-9(\[]|[\"']))|\n+", text)
    output: list[str] = []
    for chunk in chunks:
        s = _clean_sentence(chunk)
        for token, value in protected.items():
            s = s.replace(token, value)
        s = s.replace("§INITIAL§", ".")
        if len(s) >= 3 and not re.fullmatch(r"\W+", s):
            output.append(s)
    return output


def _surname(author: str) -> str:
    value = author.strip()
    if "," in value:
        return value.split(",", 1)[0].strip().strip("{}")
    bits = re.sub(r"\s+", " ", value).split()
    if len(bits) >= 2 and len(bits[-1].rstrip(".")) <= 2:
        return bits[0].strip("{}")
    return bits[-1].strip("{}") if bits else ""


def _reference_surnames(ref: dict[str, Any]) -> list[str]:
    result: list[str] = []
    for author in ref.get("authors", []) or []:
        name = _surname(str(author))
        if len(name) >= 3:
            result.append(name)
    if not result:
        raw = str(ref.get("original_reference", ""))
        author_bit = re.split(r"\b(?:19|20)\d{2}\b", raw, maxsplit=1)[0]
        # Typical APA-like first author prefix, before the first comma.
        candidate = author_bit.split(",", 1)[0].strip()
        candidate = re.sub(r"\[[^]]+\]", "", candidate).strip()
        if len(candidate) >= 3 and len(candidate.split()) <= 3:
            result.append(candidate.split()[-1])
    return list(dict.fromkeys(result))


def _citation_numbers(block: str) -> set[int]:
    numbers: set[int] = set()
    for token in re.split(r"[,，;；]", block):
        token = token.strip()
        if not token:
            continue
        range_match = re.fullmatch(r"(\d+)\s*[–-]\s*(\d+)", token)
        if range_match:
            left, right = int(range_match.group(1)), int(range_match.group(2))
            if min(left, right) < 1 or abs(right - left) > 100:
                return set()
            numbers.update(range(min(left, right), max(left, right) + 1))
        elif token.isdigit():
            value = int(token)
            if value < 1:
                return set()
            numbers.add(value)
        else:
            return set()
    return numbers


def _match_citation(sentence: str, ref: dict[str, Any]) -> tuple[bool, str]:
    keys = [k.strip() for k in str(ref.get("bib_key", "")).split(",") if k.strip()]
    for key in keys:
        if any(key in [k.strip() for k in block.split(",")] for block in re.findall(r"\[CITE:([^\]]+)\]", sentence)):
            return True, f"LaTeX/BibTeX citation key: {key}"
    label = str(ref.get("label", "")).strip()
    number = ref.get("citation_number")
    if label and re.search(r"\[\s*" + re.escape(label) + r"\s*\]", sentence):
        return True, f"编号引用: [{label}]"
    if number:
        for block in re.findall(r"[\[［【]([0-9,，;；\s–-]+)[\]］】]", sentence):
            numbers = _citation_numbers(block)
            if int(number) in numbers:
                return True, f"按参考文献序号匹配: [{number}]"
    year = ref.get("year")
    surnames = _reference_surnames(ref)[:1]
    if year:
        for surname in surnames:
            # Bind author and year within one citation; a year elsewhere is not evidence.
            pattern = r"(?<!\w)" + re.escape(surname) + r"(?:\s+et\s+al\.?|等|\s*(?:&|and)\s+[\w-]+)?\s*[,，]?\s*[（(]?\s*" + str(year) + r"[a-z]?(?!\d)"
            if re.search(pattern, sentence, re.I):
                return True, f"作者-年份引用: {surname} ({year})"
    return False, ""


def split_citation_claims(sentence: str) -> list[str]:
    """Split a sentence containing adjacent numeric citations into claims.

    For example, ``A[33]，B[34]`` becomes ``A[33]`` and ``B[34]`` so each
    paper is evaluated against the statement it actually cites.
    """
    matches = [match for match in re.finditer(r"[\[［【]([0-9,，;；\s–-]+)[\]］】]", sentence)
               if _citation_numbers(match.group(1))]
    if len(matches) <= 1:
        return [sentence]
    boundaries: list[int] = []
    for current, following in zip(matches, matches[1:]):
        between = sentence[current.end():following.start()]
        substantive = re.sub(r"^[\s,，;；。！？、]*(?:(?:and|or)\b|[和及与])?\s*", "", between, flags=re.I)
        if substantive.strip():
            boundaries.append(current.end())
    if not boundaries:
        return [sentence]
    pieces: list[str] = []
    start = 0
    for end in boundaries + [len(sentence)]:
        piece = sentence[start:end].strip(" \t,，;；")
        if piece:
            pieces.append(piece)
        start = end
    return pieces or [sentence]


async def map_citations(main_text: str, references: list[dict[str, Any]]) -> tuple[list[dict[str, Any]], list[dict[str, Any]], list[str]]:
    sentences = split_sentences(main_text)
    pairs: list[dict[str, Any]] = []
    matched_ref_ids: set[str] = set()
    warnings: list[str] = []
    for sentence in sentences:
        claims = split_citation_claims(sentence)
        for claim in claims:
            matches: list[tuple[dict[str, Any], str]] = []
            for ref in references:
                ok, method = _match_citation(claim, ref)
                if ok:
                    matches.append((ref, method))
            for ref, method in matches:
                matched_ref_ids.add(ref["reference_id"])
                pairs.append({
                    "pair_id": f"PAIR{len(pairs) + 1:05d}",
                    "statement_original": claim,
                    "statement_summary": claim,
                    "claim_type": "待识别",
                    "reference_id": ref["reference_id"],
                    "mapping_method": method,
                    "standardized_response": {
                        "judgmental_statement": claim,
                        "supporting_paper": ref.get("normalized_citation", ref.get("original_reference", "")),
                        "mapping_basis": method,
                        "verification_scope": "此处仅确认文本中的引用对应关系，不代表文献已被证实支持该判断。",
                    },
                })
    # Ask the LLM only to summarize/classify claims; exact original sentence stays immutable.
    if pairs:
        for start in range(0, len(pairs), 15):
            chunk = pairs[start:start + 15]
            prompt_data = [{"pair_id": p["pair_id"], "sentence": p["statement_original"]} for p in chunk]
            try:
                result = await ainvoke_json(
                    "你是学术引用映射器。对给定原句做轻量分类与摘要；不得改写或替换原句，不得判断引用文献是否真的支持它。输出 JSON：{\"items\":[{\"pair_id\":\"PAIR00001\",\"claim_type\":\"理论主张/经验发现/方法描述/定义/研究动机/其他\",\"statement_summary\":\"忠实简述\"}]}。",
                    "逐条处理以下句子：\n" + __import__("json").dumps(prompt_data, ensure_ascii=False),
                    retries=settings.llm_retries,
                )
                items = result.get("items", []) if isinstance(result, dict) else []
                by_id = {x.get("pair_id"): x for x in items if isinstance(x, dict)}
                for pair in chunk:
                    item = by_id.get(pair["pair_id"], {})
                    pair["claim_type"] = str(item.get("claim_type") or "其他")
                    pair["statement_summary"] = str(item.get("statement_summary") or pair["statement_original"])
            except Exception as exc:
                warnings.append(f"Agent2 的摘要/分类调用失败，已保留原句继续：{exc}")
    unmatched = [r for r in references if r["reference_id"] not in matched_ref_ids]
    if unmatched:
        warnings.append(f"有 {len(unmatched)} 条参考文献未在正文中找到可确定的引用句；已列入 unmatched_references，不会虚构对应句子。")
    return pairs, unmatched, warnings
