from __future__ import annotations

import argparse
import asyncio
import json
from pathlib import Path

from .workflow import WorkflowService


def main() -> None:
    parser = argparse.ArgumentParser(description="Agent Research CD：文献引用证据核验工作流")
    parser.add_argument("input", help="输入 .docx、.doc、.tex 或 .latex 文件路径")
    parser.add_argument("-o", "--output", help="输出 JSON 路径；默认在 output/ 下生成")
    parser.add_argument("--threshold", type=float, default=None, help="置信度通过阈值，0–1；默认读取 .env")
    args = parser.parse_args()
    input_path = Path(args.input).resolve()
    output_path = Path(args.output).resolve() if args.output else Path("output") / f"{input_path.stem}_research_cd.json"
    output_path.parent.mkdir(parents=True, exist_ok=True)
    report = asyncio.run(WorkflowService().run(input_path, threshold=args.threshold))
    output_path.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps({
        "output": str(output_path),
        "status": report.get("status"),
        "references": report.get("statistics", {}).get("references_total", 0),
        "pairs": report.get("statistics", {}).get("pairs_total", 0),
        "passed": report.get("statistics", {}).get("passed", 0),
        "failed": report.get("statistics", {}).get("failed", 0),
        "manual_required": report.get("statistics", {}).get("manual_required", 0),
        "rejected": report.get("statistics", {}).get("rejected", 0),
        "failure_reasons": report.get("statistics", {}).get("failure_reasons", {}),
    }, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
