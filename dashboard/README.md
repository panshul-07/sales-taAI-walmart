# Dashboard Service (FastAPI + React/Vite Frontend + taAI)

This folder contains the complete dashboard service.

## Live Deployment

- App: https://sales-panshul-walmart.onrender.com

## Executive Summary

`taAI` is a production-grade Walmart financial copilot combining model-driven forecasting, scenario simulation, and multilingual economist QA in one dashboard.

## Key Capabilities

- Deterministic data QA (highest/lowest week, top-N stores, trend and holiday impact)
- Notebook-linked ExtraTrees prediction + coefficient interpretation
- Multi-turn chat with persistent sessions
- Distribution diagnostics (skewness, kurtosis, Jarque-Bera)
- Responsive React frontend with interactive charts and what-if controls

## System Architecture

### Tier 1: Data Layer

- Source dataset: `dashboard/data/walmart_sales.csv`
- Date normalization + feature parsing in backend loader
- Store/date indexed API-ready rows

### Tier 2: AI/ML Processing Layer

- Notebook-style ExtraTrees artifact for predictions
- Coefficient extraction for simulation sensitivity
- Scheduled artifact refresh with cron

### LLM Agent System

- Intent routing + rule-based analytics tools
- Mandatory Llama responses via Ollama (`LLM_PROVIDER=ollama`)
- MCP-style context packet generation for tool grounding (`/api/taai/mcp/context`)
- Persistent chat sessions in SQLite

## What It Includes

- API endpoints for stores, overview, timeseries rows, correlations, coefficients
- Pickled notebook-style ExtraTrees model artifact with scheduled retraining
- `taAI` chatbot endpoints with session memory
- Responsive React + Vite UI with:
  - KPI cards
  - Actual vs baseline vs simulated trend
  - Factor sliders for what-if analysis
  - Scatter plot (Adjusted factor vs Simulated sales)
  - Correlation bars + coefficient table
  - Floating `taAI` assistant widget

## Run Locally

```bash
cd /Users/panshulaj/Documents/front
python3 -m venv .venv
.venv/bin/pip install -r dashboard/requirements.txt
.venv/bin/uvicorn dashboard.backend.server:app --host 127.0.0.1 --port 8000
```

Open: `http://127.0.0.1:8000`

## API Endpoints

- `GET /api/health`
- `GET /api/stores`
- `GET /api/overview?store=all&weeks=160`
- `GET /api/store-data?store=all&weeks=160`
- `GET /api/correlations?store=all&weeks=160`
- `GET /api/coefficients?store=all&weeks=160`
- `GET /api/stats/distribution?store=all&weeks=160`
- `POST /api/taai/chat`
- `GET /api/taai/sessions/{session_id}`
- `GET /api/taai/sessions?limit=10`
- `GET /api/taai/mcp/context`

## Required LLM Config (taAI Product Mode)

Set these env vars on Render:

- `LLM_PROVIDER=ollama`
- `OLLAMA_BASE_URL` (default: `http://127.0.0.1:11434`)
- `OLLAMA_MODEL` (default: `llama3.1:8b-instruct-q4_K_M`)

If Ollama is unreachable, taAI returns a configuration/runtime warning for chat requests.

## Log-Based Parametric Reliability

- Coefficients are learned in log-space (`ln(Weekly_Sales)` vs log features) and exposed with:
  - standard error
  - t-stat
  - p-value
  - 95% confidence interval
- Distribution diagnostics use calibrated log-residuals (Yeo-Johnson + winsorization) to stabilize kurtosis/JB and support parametric interpretation.

## Llama Fine-Tune Scaffold

Generate starter JSONL data for supervised Llama adaptation:

```bash
cd /Users/panshulaj/Documents/front/dashboard
/Users/panshulaj/Documents/front/.venv/bin/python backend/prepare_llama_finetune.py
```

Output file:
- `dashboard/backend/finetune/taai_llama_finetune.jsonl`

## LangChain RAG (Implemented)

- RAG engine module: `dashboard/backend/rag_langchain.py`
- Chat retrieval is now wired through this engine.
- Status endpoint: `GET /api/taai/rag/status`
- Query endpoint: `GET /api/taai/rag/query?q=...`

If you want full LangChain BM25 retriever support (instead of fallback retrieval), install:

```bash
cd /Users/panshulaj/Documents/front
.venv/bin/pip install langchain-core langchain-community rank-bm25
```

## Data Behavior

- If `DATA_PATH` points to a valid Walmart CSV, backend uses that.
- Otherwise backend generates realistic demo data for 45 stores.
- For `store=all`, data is aggregated by date for correct trend math.
- Model uses notebook-style ExtraTrees training and writes pickle to:
  - `dashboard/backend/model_artifacts/extra_trees_notebook.pkl`
  - training script: `dashboard/backend/train_notebook_artifact.py`

## Deploy on Render

`render.yaml` is configured to:
- install dependencies from `requirements.txt`
- start app with `uvicorn backend.server:app --host 0.0.0.0 --port $PORT`
- run a dedicated Render cron job every 6 hours:
  - `python backend/train_notebook_artifact.py`

## Verify Notebook-Driven Runtime

After deploy, check:

- `GET /api/health`:
  - `model_source` should be `extra_trees_notebook_pickle`
- `GET /api/coefficients?store=all&weeks=160`:
  - coefficients should come from notebook-linked artifact payload
- `GET /api/store-data?store=all&weeks=160`:
  - `Predicted_Sales` should be sourced from the ExtraTrees artifact map

Deploy this folder as the service root (`rootDir: dashboard`).
