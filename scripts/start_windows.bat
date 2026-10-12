@echo off
setlocal
if not exist .env copy .env.example .env
python -m venv .venv
call .venv\Scripts\activate
python -m pip install --upgrade pip
pip install -r requirements.txt
rem Default stdio mode starts MCP automatically. For http mode start app.mcp_server separately.
python -m uvicorn app.main:app --host 0.0.0.0 --port 8000
