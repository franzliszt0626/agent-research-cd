"""Read citation source URLs inside the MCP server only; never invent an abstract."""
from __future__ import annotations

import asyncio
import ipaddress
import re
import socket
from typing import Any
from urllib.parse import urljoin, urlsplit

import httpx
from bs4 import BeautifulSoup


async def public_url(url: str) -> str:
    parts = urlsplit(url)
    if parts.scheme not in {"https", "http"} or not parts.hostname or parts.username or parts.password:
        raise ValueError("Source URL must be a public http(s) URL without credentials")
    addresses = await asyncio.get_running_loop().getaddrinfo(parts.hostname, parts.port or (443 if parts.scheme == "https" else 80), type=socket.SOCK_STREAM)
    if not addresses or any(not ipaddress.ip_address(item[4][0]).is_global for item in addresses):
        raise ValueError("Private/local source URLs are not allowed")
    return url


def arxiv_identifier(value: str) -> str:
    match = re.search(r"(?:arxiv\s*:\s*|arxiv\.org/(?:abs|pdf)/|^)(\d{4}\.\d{4,5}(?:v\d+)?)(?=$|[^\d])", value, re.I)
    return match.group(1) if match else ""


def parse_source_html(content: str, url: str, arxiv_id: str = "") -> dict[str, Any]:
    soup = BeautifulSoup(content, "html.parser")
    def meta(*names):
        for name in names:
            tag = soup.find("meta", attrs={"name": re.compile("^" + re.escape(name) + "$", re.I)}) or soup.find("meta", attrs={"property": name})
            if tag and tag.get("content"):
                return str(tag["content"]).strip()
        return ""
    title = meta("citation_title", "DC.Title", "og:title") or (soup.title.get_text(" ", strip=True) if soup.title else "")
    abstract = meta("citation_abstract", "DC.Description")
    block = soup.select_one("blockquote.abstract, section.abstract, div.abstract, #abstract")
    if block:
        abstract = re.sub(r"^Abstract\s*:?\s*", "", block.get_text(" ", strip=True), flags=re.I)
    # General webpage descriptions are not necessarily abstracts. Keep their body for deep review.
    authors = [str(tag["content"]) for tag in soup.find_all("meta", attrs={"name": "citation_author"}) if tag.get("content")]
    doi = meta("citation_doi", "DC.Identifier")
    match = re.search(r"10\.\d{4,9}/[^\s]+", doi)
    doi = match.group(0).rstrip(".,") if match else ""
    year = re.search(r"(?:19|20)\d{2}", meta("citation_date", "citation_publication_date", "DC.Date"))
    pdf_url = meta("citation_pdf_url")
    if pdf_url:
        pdf_url = urljoin(url, pdf_url)
    if arxiv_id:
        pdf_url = pdf_url or f"https://arxiv.org/pdf/{arxiv_id}"
        # Preserve the repository identifier without claiming a registered DOI.
    for tag in soup(["script", "style", "nav", "footer", "header", "noscript"]):
        tag.decompose()
    body = soup.select_one("article, main, #content, .article, .TRS_Editor, .readme") or soup.body or soup
    text = re.sub(r"[ \t]+", " ", body.get_text("\n", strip=True))
    challenge = re.search(r"^(?:just a moment|access denied|verify you are human|403 forbidden|404 not found)\b", title, re.I)
    if not title or challenge or len(text) < 100:
        raise ValueError("Source page is empty, an error page or a browser challenge")
    return {"found": True, "title": title, "authors": authors, "year": int(year.group()) if year else None,
            "doi": doi, "arxiv_id": arxiv_id, "abstract": abstract, "pdf_url": pdf_url,
            "landing_page_url": url, "source": "arXiv" if arxiv_id else "Reference URL",
            "is_open_access": True if arxiv_id else None,
            "access_status": "open-access" if arxiv_id else "access-undetermined", "_page_text": text}


async def fetch_source(source_url: str = "", arxiv_id: str = "", *, include_text: bool = False) -> dict[str, Any] | None:
    identifier = arxiv_identifier(arxiv_id) or arxiv_identifier(source_url)
    url = f"https://arxiv.org/abs/{identifier}" if identifier else source_url
    if not url:
        return None
    async with httpx.AsyncClient(timeout=25, headers={"User-Agent": "AgentResearchCD/1.2 (citation verification)"}, follow_redirects=False) as client:
        for _ in range(6):
            await public_url(url)
            async with client.stream("GET", url) as response:
                if response.is_redirect:
                    location = response.headers.get("location")
                    if not location:
                        raise ValueError("Source redirect has no destination")
                    url = urljoin(url, location)
                    continue
                response.raise_for_status()
                if "pdf" in response.headers.get("content-type", "").lower():
                    return {"found": True, "landing_page_url": url, "pdf_url": url, "source": "Reference URL", "access_status": "open-access", "is_open_access": True}
                chunks = bytearray()
                async for chunk in response.aiter_bytes():
                    chunks.extend(chunk)
                    if len(chunks) > 3_000_000:
                        raise ValueError("Source page exceeds size limit")
                # Let BeautifulSoup read HTML charset declarations before falling back to UTF-8.
                decoded = BeautifulSoup(bytes(chunks), "html.parser")
                result = parse_source_html(str(decoded), url, identifier)
                if not include_text:
                    result.pop("_page_text", None)
                return result
    raise ValueError("Too many source redirects")
