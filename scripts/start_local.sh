#!/usr/bin/env bash
set -euo pipefail
if [ ! -f .env ]; then cp .env.example .env; echo "已创建 .env；请先填写 LLM_API_KEY / LLM_BASE_URL，再启动。"; fi
python -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
pip install -r requirements.txt
echo "MCP_TRANSPORT=${MCP_TRANSPORT:-stdio}；默认由 Web/CLI 自动启动 MCP 子进程。"
echo "启动 Web 界面："
python -m uvicorn app.main:app --host 0.0.0.0 --port 8000
