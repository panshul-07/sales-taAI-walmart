# Walmart Sales Forecast Dashboard

Production-ready Walmart sales dashboard with a FastAPI backend, React frontend, notebook-linked model artifact, and taAI chatbot.

## Live Deployment

- App: https://sales-panshul-walmart.onrender.com

## Current Stack

- Backend: FastAPI + Uvicorn (`dashboard/backend/server.py`)
- Frontend: React + Vite build served from `dashboard/static/`
- Modeling: ExtraTrees artifact + log-space parametric coefficient diagnostics
- Chatbot LLM: OpenAI or Llama (Ollama) with MCP-style context packets
- Deployment: Render Blueprint (`dashboard/render.yaml`)

## Repo Structure

- `dashboard/backend/server.py`: API + static file serving
- `dashboard/static/index.html`: dashboard UI and chart logic
- `walmart_ai_extension/`: new full-stack AI chatbot platform (FastAPI + React + Postgres + LangChain + Plotly)
- `dashboard/requirements.txt`: runtime dependencies
- `dashboard/render.yaml`: Render deploy config
- `dashboard/README.md`: dashboard-level run/API details
- `walmart_sales_forecasting.ipynb`: historical notebook work

## Local Run

```bash
cd /Users/panshulaj/Documents/front
python3 -m venv .venv
.venv/bin/pip install -r dashboard/requirements.txt
.venv/bin/uvicorn dashboard.backend.server:app --host 127.0.0.1 --port 8000
```

Open: `http://127.0.0.1:8000`

## Deploy

1. Push to GitHub.
2. Create/update Render Blueprint using `dashboard/render.yaml`.
3. Deploy web service.

## Notes

- If a real CSV is available, set `DATA_PATH` to use it.
- Without a CSV, the backend generates demo data automatically.
