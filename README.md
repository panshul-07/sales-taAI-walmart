# Walmart Sales Forecast Dashboard

Production-ready Walmart sales dashboard with a FastAPI backend and responsive frontend.

## Current Stack

- Backend: FastAPI + Uvicorn (`dashboard/backend/server.py`)
- Frontend: single-page dashboard in vanilla JS/SVG (`dashboard/static/index.html`)
- Modeling: lightweight forecasting + coefficient-based what-if simulation (pure Python)
- Deployment: Render Blueprint (`dashboard/render.yaml`)

## Repo Structure

- `dashboard/backend/server.py`: API + static file serving
- `dashboard/static/index.html`: dashboard UI and chart logic
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

## Live Dashboard Link

- Current live URL: `https://walmart-sales-frontend.onrender.com/`
- This is the persistent Render deployment URL.

## Notes

- If a real CSV is available, set `DATA_PATH` to use it.
- Without a CSV, the backend generates demo data automatically.
