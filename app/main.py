from __future__ import annotations

import json
import os
import tempfile
from pathlib import Path
from typing import Annotated, Any

from fastapi import FastAPI, File, Form, HTTPException, UploadFile
from fastapi.responses import HTMLResponse, JSONResponse, Response, StreamingResponse

from .config import settings
from .workflow import WorkflowService
from .llm import LLMServiceError, check_llm

app = FastAPI(
    title="Agent Research CD",
    description="LangGraph + LangChain + MCP 的学术引用证据核验工作流",
    version="1.2.0",
)
service = WorkflowService()
ALLOWED_EXTENSIONS = {".docx", ".doc", ".tex", ".latex"}


def _visual_output(value: Any, key: str = "") -> Any:
    """Keep streamed step output readable while preserving the full final report."""
    if isinstance(value, dict):
        return {name: _visual_output(item, name) for name, item in value.items() if name != "report"}
    if isinstance(value, list):
        return [_visual_output(item, key) for item in value]
    if isinstance(value, str) and key in {"main_text", "full_text"}:
        return {"characters": len(value), "preview": value[:800] + ("…" if len(value) > 800 else "")}
    if isinstance(value, str) and key == "abstract" and len(value) > 1500:
        return value[:1500] + "…"
    return value


def _ndjson(event: dict[str, Any]) -> bytes:
    return (json.dumps(event, ensure_ascii=False, allow_nan=False) + "\n").encode("utf-8")

HOME_HTML = r"""<!doctype html>
<html lang="zh-CN"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width, initial-scale=1">
<title>Agent Research CD</title>
<style>
body{font-family:system-ui,-apple-system,"Segoe UI",sans-serif;background:#f5f7fb;color:#1e293b;margin:0}
main{max-width:900px;margin:48px auto;padding:0 20px}h1{font-size:30px;margin-bottom:8px}.muted{color:#64748b;line-height:1.7}
.card{background:#fff;border:1px solid #dbe3ef;border-radius:16px;padding:24px;margin:20px 0;box-shadow:0 6px 22px rgba(15,23,42,.04)}
label{display:block;font-weight:650;margin:15px 0 8px}input,button{font:inherit}input[type=file]{display:block;width:100%;padding:12px;border:1px solid #cbd5e1;border-radius:10px;background:#fff}
input[type=number]{padding:10px;width:110px;border:1px solid #cbd5e1;border-radius:10px}button{background:#1d4ed8;color:#fff;border:0;border-radius:10px;padding:12px 18px;cursor:pointer;font-weight:650}button:disabled{opacity:.5;cursor:wait}
pre{background:#0f172a;color:#e2e8f0;padding:18px;border-radius:12px;overflow:auto;white-space:pre-wrap;word-break:break-word;max-height:620px}.row{display:flex;gap:16px;align-items:end;flex-wrap:wrap}.status{margin-top:12px;font-weight:600}.small{font-size:13px;color:#64748b}.steps{display:grid;gap:10px}.step{border:1px solid #dbe3ef;border-radius:10px;padding:14px;background:#fbfdff}.step-head{display:flex;justify-content:space-between;gap:12px;align-items:center}.step-name{font-weight:700}.step-state{font-size:13px;color:#64748b}.step[data-state=running]{border-color:#2563eb;background:#eff6ff}.step[data-state=done]{border-color:#16a34a}.step[data-state=error]{border-color:#dc2626;background:#fef2f2}.step pre{margin:10px 0 0;max-height:280px;font-size:12px}.summary{display:flex;gap:10px;flex-wrap:wrap;margin:12px 0}.metric{padding:10px 12px;background:#eff6ff;border-radius:8px;font-size:13px}
</style></head><body><main>
<h1>Agent Research CD</h1><p class="muted">科研文献引用证据核验。上传 DOCX / DOC / LaTeX 文件，工作流会提取引用关系、检索 DOI 与摘要、初审并按需尝试全文复核，最后输出 JSON。</p>
<div class="card"><form id="form"><label for="file">论文文件</label><input id="file" type="file" name="file" accept=".docx,.doc,.tex,.latex" required>
<label for="bib_file">配套 BibTeX 文件（可选；仅 LaTeX 需要）</label><input id="bib_file" type="file" name="bib_file" accept=".bib">
<div class="row"><div><label for="threshold">通过阈值（0–1）</label><input id="threshold" name="threshold" type="number" min="0" max="1" step="0.05" value="0.75"></div><div><button id="submit" type="submit">运行工作流</button></div><div><button id="checkModel" type="button">检测模型连接</button></div></div>
<p class="small">说明：运行时间取决于文献数量、模型服务和外部开放获取源的响应情况。文件只在本次服务容器中临时处理。</p></form><div class="status" id="status"></div></div>
<div class="card" id="outputCard" hidden><div class="row" style="justify-content:space-between;align-items:center"><h2>JSON 结果</h2><button id="download" type="button">下载 JSON</button></div><pre id="output"></pre></div>
<div class="card" id="stepsCard" hidden><div class="row" style="justify-content:space-between;align-items:center"><h2>工作流步骤</h2><span class="small" id="stepProgress"></span></div><div class="steps" id="steps"></div></div></main><script>
(()=>{
const form=document.getElementById('form');
const btn=form.querySelector('#submit'),statusEl=document.getElementById('status'),stepsCard=document.getElementById('stepsCard'),stepsEl=document.getElementById('steps'),progress=document.getElementById('stepProgress');
const names={agent6_plan:'模型预检与任务规划',agent1_extract:'提取正文与参考文献',agent2_map:'映射引用与论断',agent3_search:'检索文献元数据',agent4_initial_review:'初审摘要证据',agent6_route_deep_review:'判断是否需要全文复核',agent3_fetch_fulltext:'获取开放全文',agent5_deep_review:'全文证据复核',agent6_finalize:'汇总最终判定'};
const order=Object.keys(names);let latest=null,done=0;const cards=new Map(),show=v=>JSON.stringify(v??{},null,2);
const card=step=>{if(cards.has(step))return cards.get(step);const el=document.createElement('section');el.className='step';el.dataset.state='waiting';el.innerHTML=`<div class="step-head"><span class="step-name">${names[step]||step}</span><span class="step-state">等待</span></div><pre hidden></pre>`;stepsEl.appendChild(el);cards.set(step,el);return el};
const setState=(step,state,out)=>{const el=card(step),was=el.dataset.state;el.dataset.state=state;el.querySelector('.step-state').textContent=state==='running'?'运行中':state==='done'?'已完成':state==='error'?'失败':state==='skipped'?'已跳过':'等待';if(out!==undefined){const pre=el.querySelector('pre');pre.hidden=false;pre.textContent=show(out)}if(state==='done'&&was!=='done')done++;progress.textContent=`${done} 个步骤已完成`};
form.addEventListener('submit',async e=>{e.preventDefault();const f=form.querySelector('#file').files[0];if(!f)return;btn.disabled=true;latest=null;done=0;cards.clear();stepsEl.innerHTML='';order.forEach(card);stepsCard.hidden=false;document.getElementById('outputCard').hidden=true;progress.textContent='0 个步骤已完成';statusEl.textContent='工作流运行中，请保持页面打开……';
try{const data=new FormData(form),res=await fetch('/api/analyze-stream',{method:'POST',body:data});if(!res.ok)throw new Error((await res.json()).detail||'请求失败');const reader=res.body.getReader(),decoder=new TextDecoder();let buffer='';for(;;){const part=await reader.read();if(part.done)break;buffer+=decoder.decode(part.value,{stream:true});const rows=buffer.split('\n');buffer=rows.pop();for(const row of rows){if(!row.trim())continue;const event=JSON.parse(row);if(event.type==='step_start')setState(event.step,'running');else if(event.type==='step_end')setState(event.step,'done',event.output);else if(event.type==='workflow_end')order.forEach(s=>{if(card(s).dataset.state==='waiting')setState(s,'skipped')});else if(event.type==='final'){latest=event.report;document.getElementById('output').textContent=show(latest);document.getElementById('outputCard').hidden=false;const s=latest.statistics||{};statusEl.textContent=`完成：${s.pairs_total??0} 个引用配对，通过 ${s.passed??0} 个，待人工复核 ${s.manual_required??0} 个`;}else if(event.type==='error')throw new Error(event.detail||'工作流失败')}}}
catch(err){statusEl.textContent='运行失败：'+err.message;for(const [step,el] of cards)if(el.dataset.state==='running')setState(step,'error')}
finally{btn.disabled=false}});
form.querySelector('#checkModel').addEventListener('click',async()=>{statusEl.textContent='正在检测模型连接……';try{const res=await fetch('/api/check-model',{method:'POST'}),body=await res.json();if(!res.ok)throw new Error(body.detail||'检测失败');statusEl.textContent=`模型连接正常：${body.model}`}catch(err){statusEl.textContent='模型连接失败：'+err.message}});
document.getElementById('download').addEventListener('click',()=>{if(!latest)return;const blob=new Blob([JSON.stringify(latest,null,2)],{type:'application/json;charset=utf-8'}),url=URL.createObjectURL(blob),a=document.createElement('a');a.href=url;a.download='research_cd_report.json';a.click();URL.revokeObjectURL(url)});
})();
</script></body></html>"""


@app.get("/", response_class=HTMLResponse)
async def home() -> str:
    return HOME_HTML


@app.get("/health")
async def health() -> dict[str, str]:
    return {"status": "ok", "service": "agent-research-cd", "version": "1.2.0"}


@app.post("/api/analyze")
async def analyze(
    file: Annotated[UploadFile, File(...)],
    threshold: Annotated[float, Form()] = settings.confidence_threshold,
    bib_file: Annotated[UploadFile | None, File()] = None,
) -> Response:
    filename = Path(file.filename or "upload").name
    ext = Path(filename).suffix.lower()
    if ext not in ALLOWED_EXTENSIONS:
        raise HTTPException(status_code=400, detail="文件格式不支持。请上传 .docx、.doc、.tex 或 .latex。")
    if not 0 <= threshold <= 1:
        raise HTTPException(status_code=400, detail="threshold 必须介于 0 和 1 之间。")
    content = await file.read()
    if len(content) > settings.max_upload_mb * 1024 * 1024:
        raise HTTPException(status_code=413, detail=f"文件超过 {settings.max_upload_mb} MB 限制。")
    if not content:
        raise HTTPException(status_code=400, detail="上传文件为空。")
    bib_content: bytes | None = None
    bib_name = ""
    if bib_file is not None and bib_file.filename:
        bib_name = Path(bib_file.filename).name
        if Path(bib_name).suffix.lower() != ".bib":
            raise HTTPException(status_code=400, detail="配套文献库文件必须是 .bib 格式。")
        if ext not in {".tex", ".latex"}:
            raise HTTPException(status_code=400, detail="只有上传 LaTeX 文件时才能同时上传 .bib 文件。")
        bib_content = await bib_file.read()
        if len(bib_content) > settings.max_upload_mb * 1024 * 1024:
            raise HTTPException(status_code=413, detail=f"BibTeX 文件超过 {settings.max_upload_mb} MB 限制。")
    # A temporary sibling directory prevents name collisions across concurrent uploads.
    try:
        with tempfile.TemporaryDirectory(prefix="research-cd-") as tmp:
            path = Path(tmp) / filename
            path.write_bytes(content)
            if bib_content is not None:
                (Path(tmp) / bib_name).write_bytes(bib_content)
            report = await service.run(path, threshold=threshold)
            return JSONResponse(content=report)
    except LLMServiceError as exc:
        raise HTTPException(status_code=503, detail=f"模型自检失败，工作流尚未开始。{exc}") from exc
    except (ValueError, FileNotFoundError, RuntimeError) as exc:
        raise HTTPException(status_code=422, detail=str(exc)) from exc
    except Exception as exc:
        raise HTTPException(status_code=500, detail=f"工作流执行失败：{str(exc)[:1000]}") from exc


@app.post("/api/analyze-stream")
async def analyze_stream(
    file: Annotated[UploadFile, File(...)],
    threshold: Annotated[float, Form()] = settings.confidence_threshold,
    bib_file: Annotated[UploadFile | None, File()] = None,
) -> StreamingResponse:
    filename = Path(file.filename or "upload").name
    ext = Path(filename).suffix.lower()
    if ext not in ALLOWED_EXTENSIONS:
        raise HTTPException(status_code=400, detail="文件格式不支持。请上传 .docx、.doc、.tex 或 .latex。")
    if not 0 <= threshold <= 1:
        raise HTTPException(status_code=400, detail="threshold 必须介于 0 和 1 之间。")
    content = await file.read()
    if not content:
        raise HTTPException(status_code=400, detail="上传文件为空。")
    if len(content) > settings.max_upload_mb * 1024 * 1024:
        raise HTTPException(status_code=413, detail=f"文件超过 {settings.max_upload_mb} MB 限制。")
    bib_content: bytes | None = None
    bib_name = ""
    if bib_file is not None and bib_file.filename:
        bib_name = Path(bib_file.filename).name
        if Path(bib_name).suffix.lower() != ".bib":
            raise HTTPException(status_code=400, detail="配套文献库文件必须是 .bib 格式。")
        if ext not in {".tex", ".latex"}:
            raise HTTPException(status_code=400, detail="只有上传 LaTeX 文件时才能同时上传 .bib 文件。")
        bib_content = await bib_file.read()
        if len(bib_content) > settings.max_upload_mb * 1024 * 1024:
            raise HTTPException(status_code=413, detail=f"BibTeX 文件超过 {settings.max_upload_mb} MB 限制。")

    async def events():
        try:
            with tempfile.TemporaryDirectory(prefix="research-cd-") as tmp:
                path = Path(tmp) / filename
                path.write_bytes(content)
                if bib_content is not None:
                    (Path(tmp) / bib_name).write_bytes(bib_content)
                yield _ndjson({"type": "workflow_start"})
                async for event in service.stream(path, threshold=threshold):
                    if event["type"] == "step_end":
                        output = event.get("output", {})
                        if event.get("step") == "agent6_finalize" and isinstance(output, dict) and isinstance(output.get("report"), dict):
                            report = output["report"]
                            event["output"] = {"status": report.get("status"), "statistics": report.get("statistics", {}),
                                                "warnings": report.get("warnings", []), "errors": report.get("errors", [])}
                        else:
                            event["output"] = _visual_output(output)
                    yield _ndjson(event)
                yield _ndjson({"type": "workflow_end"})
        except Exception as exc:
            yield _ndjson({"type": "error", "detail": str(exc)[:1000]})

    return StreamingResponse(events(), media_type="application/x-ndjson; charset=utf-8",
                             headers={"X-Accel-Buffering": "no", "Cache-Control": "no-cache"})


@app.post("/api/check-model")
async def check_model():
    try:
        return await check_llm()
    except LLMServiceError as exc:
        raise HTTPException(status_code=503, detail=str(exc)) from exc
