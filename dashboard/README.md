# Dashboard Service (FastAPI + React/Vite Frontend + taAI)

This folder contains the complete dashboard service.

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
- `POST /api/taai/chat`
- `GET /api/taai/sessions/{session_id}`
- `GET /api/taai/sessions?limit=10`

## Optional LLM Config (taAI Product Mode)

Set these env vars on Render to enable grounded LLM responses:

- `OPENAI_API_KEY`
- `OPENAI_MODEL` (optional, default: `gpt-4o-mini`)

Without API key, taAI falls back to built-in economist rules.

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
