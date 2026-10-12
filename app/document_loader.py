from __future__ import annotations

import re
import shutil
import subprocess
import tempfile
from dataclasses import dataclass
from pathlib import Path

from docx import Document
from docx.oxml.ns import qn
from docx.text.paragraph import Paragraph
from docx.table import Table


@dataclass
class LoadedDocument:
    source_name: str
    source_ext: str
    full_text: str
    main_text: str
    bibliography_text: str
    bibliography_entries: list[dict[str, str]]
    warnings: list[str]


def _docx_text(path: Path) -> str:
    doc = Document(str(path))
    pieces: list[str] = []
    # Keep tables in document order; appending them at the end pollutes the bibliography.
    for block in doc.element.body:
        if block.tag == qn("w:p"):
            paragraph = Paragraph(block, doc)
            value = ""
            # Word superscript citations often have no literal brackets.
            for run in paragraph.runs:
                part = run.text
                if run.font.superscript and not value.endswith(("[", "［")) and re.fullmatch(r"\d+(?:[,，–-]\d+)*", part):
                    part = f"[{part}]"
                value += part
            # paragraph.text includes hyperlink content that older python-docx omits from runs.
            if any(block.iter(qn("w:hyperlink"))):
                value = paragraph.text
            value = value.strip()
            if value:
                pieces.append(value)
        elif block.tag == qn("w:tbl"):
            for row in Table(block, doc).rows:
                cells = [cell.text.strip().replace("\n", " ") for cell in row.cells]
                if any(cells):
                    pieces.append(" | ".join(cells))
    return "\n".join(pieces)


def _doc_text(path: Path) -> str:
    """Read old .doc files using antiword or LibreOffice as a fallback."""
    if shutil.which("antiword"):
        result = subprocess.run(
            ["antiword", str(path)], capture_output=True, text=True, timeout=60
        )
        if result.returncode == 0 and result.stdout.strip():
            return result.stdout
    if shutil.which("soffice"):
        with tempfile.TemporaryDirectory() as tmp:
            result = subprocess.run(
                ["soffice", "--headless", "--convert-to", "txt:Text", "--outdir", tmp, str(path)],
                capture_output=True, text=True, timeout=90,
            )
            converted = Path(tmp) / f"{path.stem}.txt"
            if result.returncode == 0 and converted.exists():
                return converted.read_text(encoding="utf-8", errors="replace")
    raise RuntimeError("无法读取 .doc 文件。请安装 antiword 或 LibreOffice，或先将文件另存为 .docx。")


def _read_tex(path: Path) -> tuple[str, list[str]]:
    source = path.read_text(encoding="utf-8", errors="replace")
    warnings: list[str] = []
    candidates: list[Path] = []
    bib_decl = re.findall(r"\\bibliography\s*\{([^}]+)\}", source)
    for declaration in bib_decl:
        for raw in declaration.split(","):
            target = path.parent / (raw.strip() + ".bib")
            if target.exists():
                candidates.append(target)
    same_stem = path.with_suffix(".bib")
    if same_stem.exists():
        candidates.append(same_stem)
    if not candidates:
        all_bibs = sorted(path.parent.glob("*.bib"))
        if len(all_bibs) == 1:
            candidates.extend(all_bibs)
    bib_blocks = []
    for bib in dict.fromkeys(candidates):
        try:
            bib_blocks.append(bib.read_text(encoding="utf-8", errors="replace"))
        except OSError as exc:
            warnings.append(f"读取 BibTeX 文件失败 {bib.name}: {exc}")
    if bib_decl and not bib_blocks and r"\begin{thebibliography}" not in source:
        warnings.append("LaTeX 使用了 \\bibliography{}，但未找到对应 .bib 文件；请将 .bib 放在 .tex 同目录。")
    # Include BibTeX entries for parsing, but they are removed from the main-text area later.
    if bib_blocks:
        source += "\n\n% AUTOMATICALLY_ATTACHED_BIB_FILES\n" + "\n".join(bib_blocks)
    return source, warnings


def _strip_tex_commands(text: str) -> str:
    # Preserve citation keys so Agent 2 can map \cite{key} to \bibitem or BibTeX entries.
    text = re.sub(r"(?m)%.*$", "", text)
    def cite_sub(match: re.Match[str]) -> str:
        keys = match.group(1).replace(" ", "")
        return f" [CITE:{keys}] "
    text = re.sub(
        r"\\(?:cite\w*|parencite|textcite|autocite)\*?(?:\[[^\]]*\]){0,2}\{([^}]+)\}",
        cite_sub, text,
    )
    text = re.sub(r"\\(?:ref|eqref|label|pageref)\s*\{[^}]*\}", " ", text)
    text = re.sub(r"\\(?:begin|end)\s*\{[^}]*\}", "\n", text)
    text = re.sub(r"\\(?:documentclass|usepackage|bibliographystyle|bibliography|title|author|date)\s*(?:\[[^]]*\])?\s*\{[^}]*\}", " ", text)
    # Unwrap common formatting commands, one layer at a time.
    for command in ("emph", "textit", "textbf", "texttt", "underline", "mbox", "textrm", "textsf"):
        text = re.sub(r"\\" + command + r"\*?\s*\{([^{}]*)\}", r"\1", text)
    text = re.sub(r"\\url\s*\{([^}]*)\}", r"\1", text)
    text = re.sub(r"\\href\s*\{[^}]*\}\s*\{([^}]*)\}", r"\1", text)
    text = re.sub(r"\\[a-zA-Z]+\*?(?:\[[^]]*\])?", " ", text)
    text = text.replace("\\&", "&").replace("~", " ")
    text = text.replace("{", " ").replace("}", " ")
    text = re.sub(r"\s+", " ", text)
    return text.strip()


REFERENCE_HEADING = r"(?:references|reference list|bibliography|works cited|参\s*考\s*文\s*献)"
SECTION_END = re.compile(
    r"(?im)^\s*(?:(?:第[一二三四五六七八九十\d]+章|\d+[.、]?)\s*)?"
    r"(?:致\s*谢|附\s*录(?:\s*[A-Z一二三\d])?|acknowledg(?:e)?ments?|appendix(?:\s+[A-Z\d])?|appendices)"
    r"(?:\s*[:：].*|\s+.*)?\s*$"
)


def _trim_reference_tail(section: str) -> str:
    end = SECTION_END.search(section)
    if end:
        section = section[:end.start()]
    # Also handle flattened exports (including old reports) with inline Chinese headings.
    return re.split(r"\s+(?:致\s*谢|附\s*录)\s+", section, maxsplit=1)[0]


def _split_reference_section(text: str, is_tex: bool) -> tuple[str, str]:
    if is_tex:
        match = re.search(r"\\begin\s*\{thebibliography\}.*?\\end\s*\{thebibliography\}", text, re.S)
        if match:
            return text[:match.start()], match.group(0)
        # References may be loaded through an attached .bib file.
        auto = text.find("% AUTOMATICALLY_ATTACHED_BIB_FILES")
        if auto >= 0:
            return text[:auto], text[auto:]
    heading = re.compile(
        r"(?im)^\s*(?:(?:第[一二三四五六七八九十\d]+章|\d+[.、]?)\s*)?" + REFERENCE_HEADING + r"\s*[:：]?\s*$"
    )
    match = heading.search(text)
    if match:
        return text[:match.start()], _trim_reference_tail(text[match.end():])
    # Some exports flatten the heading with the first item.
    inline = re.search(r"(?im)(?:^|\n)\s*(References|Bibliography|Works Cited|参考文献)\s*\n", text)
    if inline:
        return text[:inline.start()], text[inline.start():]
    return text, ""


def _extract_bibtex_entries(text: str) -> list[dict[str, str]]:
    entries: list[dict[str, str]] = []
    pos = 0
    while True:
        match = re.search(r"@([A-Za-z]+)\s*([({])\s*([^,\s]+)\s*,", text[pos:])
        if not match:
            break
        start = pos + match.start()
        key = match.group(3).strip()
        open_pos = pos + match.start(2)
        opening = text[open_pos]
        closing = "}" if opening == "{" else ")"
        depth = 1
        i = open_pos + 1
        in_quote = False
        escaped = False
        while i < len(text) and depth > 0:
            ch = text[i]
            if ch == '"' and not escaped:
                in_quote = not in_quote
            if not in_quote:
                if ch == opening:
                    depth += 1
                elif ch == closing:
                    depth -= 1
            escaped = (ch == "\\" and not escaped)
            if ch != "\\":
                escaped = False
            i += 1
        raw = text[start:i].strip()
        entries.append({"raw": raw, "bib_key": key, "label": ""})
        pos = max(i, start + 1)
    return entries


def _extract_thebibliography_entries(section: str) -> list[dict[str, str]]:
    body_match = re.search(r"\\begin\s*\{thebibliography\}(.*?)\\end\s*\{thebibliography\}", section, re.S)
    body = body_match.group(1) if body_match else section
    matches = list(re.finditer(r"\\bibitem(?:\[[^]]*\])?\s*\{([^}]+)\}", body))
    entries: list[dict[str, str]] = []
    for i, match in enumerate(matches):
        end = matches[i + 1].start() if i + 1 < len(matches) else len(body)
        raw = body[match.end():end].strip()
        raw = re.sub(r"\\newblock\s*", " ", raw)
        entries.append({"raw": raw, "bib_key": match.group(1).strip(), "label": ""})
    return entries


GB_TYPE = r"\[(?:J|C|M|R|D|N|S|P|EB|DB|CP|DS|Z)(?:/OL)?\]"
GB_START = re.compile(
    r"(?:^|(?<=[.。])\s+|\n)"
    r"(?P<start>[A-Za-zÀ-ÖØ-öø-ÿ\u4e00-\u9fff][^.。\n\d\[\]:/]{1,180}[.。]\s*"
    r"(?:(?!\.\s)[^\n]){3,500}?" + GB_TYPE + r")", re.M
)


def _extract_plain_entries(section: str) -> list[dict[str, str]]:
    section = _trim_reference_tail(section)
    # Recover separate GB/T entries even when a Word paragraph contains several works.
    starts = [m.start("start") for m in GB_START.finditer(section)]
    for pos in reversed(starts):
        if pos and section[pos - 1] != "\n":
            section = section[:pos] + "\n" + section[pos:]
    section = re.sub(r"(?<=\S)[ \t]+(?=\[\d{1,4}\]\s*[A-Za-z\u4e00-\u9fff])", "\n", section)
    lines = [line.rstrip() for line in section.splitlines()]
    entries: list[dict[str, str]] = []
    # Common numeric styles: [12] ... / 12. ... / 12) ...
    numeric_start = re.compile(r"^\s*(?:\[(\d+)\]|(\d{1,3})[.)])\s*(.*)$")
    # Common author-year styles: Smith, J. (2022). ... / Pradhan R, Lahiri A (2022) ...
    author_year_start = re.compile(
        r"^\s*(?=.*(?:19|20)\d{2})[A-ZÀ-ÖØ-Ý][^\n]{0,180}?(?:\((?:19|20)\d{2}[a-z]?\)|(?:19|20)\d{2}[a-z]?\s*[.,])"
    )
    current = ""
    current_label = ""
    # Conference templates often repeat a running header and page number inside
    # the references section. They are layout artifacts, not bibliography text;
    # ignoring them prevents the next citation from absorbing an entire page.
    page_artifact = re.compile(
        r"^(?:\d{1,3}|(?:Forty[- ]Seventh|Forty[- ]Sixth)\s+International\s+Conference\s+on\s+Information\s+Systems,.*\d{4})$",
        re.I,
    )
    for line in lines:
        stripped = line.strip()
        if not stripped or stripped.lower().rstrip(":") in {"references", "bibliography", "works cited", "参考文献"}:
            continue
        if page_artifact.match(stripped):
            continue
        m = numeric_start.match(line)
        # A wrapped page range may continue as ``651. https://doi.org/...``.
        # It is part of the preceding citation, not bibliography item 651.
        if m and m.group(2) and re.match(r"(?:https?://|doi\b)", m.group(3).strip(), re.I):
            m = None
        author_year_starting = not current_label and bool(author_year_start.match(line))
        starts = bool(m) or author_year_starting or bool(GB_START.match(stripped))
        if starts:
            if current.strip():
                entries.append({"raw": current.strip(), "bib_key": "", "label": current_label})
            if m:
                current_label = m.group(1) or m.group(2) or ""
                current = m.group(3).strip()
            else:
                current_label = ""
                current = stripped
        elif re.match(r"^\s*[-•]\s+", line) and current.strip():
            entries.append({"raw": current.strip(), "bib_key": "", "label": current_label})
            current_label = ""
            current = re.sub(r"^\s*[-•]\s+", "", line).strip()
        elif current:
            current += " " + stripped
        elif stripped:
            current = stripped
    if current.strip():
        entries.append({"raw": current.strip(), "bib_key": "", "label": current_label})
    # Avoid treating an isolated heading or short boilerplate line as a reference.
    # Keep short numbered records too: dropping one silently shifts all later citations.
    return [e for e in entries if e["label"] or (len(e["raw"]) >= 15 and re.search(r"(?:19|20)\d{2}|doi|https?://|journal|proceedings|出版社", e["raw"], re.I))]


def load_document(file_path: str | Path, *, parse_references: bool = True) -> LoadedDocument:
    path = Path(file_path).expanduser().resolve()
    if not path.exists() or not path.is_file():
        raise FileNotFoundError(f"找不到输入文件：{path}")
    ext = path.suffix.lower()
    warnings: list[str] = []
    if ext == ".docx":
        full_text = _docx_text(path)
    elif ext == ".doc":
        full_text = _doc_text(path)
    elif ext in {".tex", ".latex"}:
        full_text, warnings = _read_tex(path)
    else:
        raise ValueError("仅支持 .docx、.doc、.tex、.latex 文件。")

    main, bibliography = _split_reference_section(full_text, is_tex=ext in {".tex", ".latex"})
    entries: list[dict[str, str]] = []
    if ext in {".tex", ".latex"}:
        entries = _extract_thebibliography_entries(bibliography)
        if not entries:
            entries = _extract_bibtex_entries(bibliography or full_text)
    if parse_references and not entries and bibliography:
        entries = _extract_plain_entries(bibliography)
    if parse_references and not entries and ext in {".tex", ".latex"}:
        entries = _extract_bibtex_entries(full_text)
        if entries:
            warnings.append("未识别到独立参考文献标题，已从 LaTeX/BibTeX 条目中提取参考文献。")
    if parse_references and not entries:
        warnings.append("未能自动解析出参考文献条目；请确认文档存在标准参考文献章节，或检查 LaTeX 的 .bib 文件是否与 .tex 放在同一目录。")
    return LoadedDocument(
        source_name=path.name,
        source_ext=ext,
        full_text=full_text,
        main_text=_strip_tex_commands(main) if ext in {".tex", ".latex"} else main,
        bibliography_text=bibliography,
        bibliography_entries=entries,
        warnings=warnings,
    )
