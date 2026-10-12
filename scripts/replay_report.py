"""Offline parser replay of raw bibliography retained in an old report, no LLM/MCP calls."""
import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from app.document_loader import _extract_plain_entries
from app.citation_mapper import _fallback_reference, _match_citation, split_sentences
from app.schemas import ReferenceItem


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("report", type=Path)
    parser.add_argument("--output", type=Path, default=Path("output/reference_replay.json"))
    args = parser.parse_args()
    old = json.loads(args.report.read_text(encoding="utf-8"))
    entries = _extract_plain_entries("\n".join(r["original_reference"] for r in old["references"]))
    refs = [ReferenceItem.model_validate(_fallback_reference(e["raw"], i, e["bib_key"], e["label"])).model_dump(exclude_none=True)
            for i, e in enumerate(entries, 1)]
    sentences = list(dict.fromkeys(s for p in old.get("pairs", []) for s in split_sentences(p["statement_original"])))
    pairs = []
    for sentence in sentences:
        for ref in refs:
            matched, method = _match_citation(sentence, ref)
            if matched:
                pairs.append({"statement_original": sentence, "reference_id": ref["reference_id"], "mapping_method": method})
    result = {"mode": "offline_parser_replay", "source_report": str(args.report),
              "limitations": "仅回放旧报告保留的原始文献和句子；编号按文献顺序推定，原始 Word 编号需重传原文件确认；未联网检索或重新评估支持性。",
              "previous_reference_count": len(old["references"]), "recovered_reference_count": len(refs),
              "recovered_pair_count": len(pairs), "references": refs, "pairs": pairs}
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"References: {len(old['references'])} -> {len(refs)}; mapped pairs: {len(pairs)}; output: {args.output}")


if __name__ == "__main__":
    main()
