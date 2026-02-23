# Dashboard Service (FastAPI + Interactive Frontend)

This folder contains the complete dashboard service.

## What It Includes

- API endpoints for stores, overview, timeseries rows, correlations, coefficients
- Responsive UI with:
  - KPI cards
  - Actual vs baseline vs simulated trend
  - Factor sliders for what-if analysis
  - Scatter plot (Adjusted factor vs Simulated sales)
  - Correlation bars + coefficient table

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

## Data Behavior

- If `DATA_PATH` points to a valid Walmart CSV, backend uses that.
- Otherwise backend generates realistic demo data for 45 stores.
- For `store=all`, data is aggregated by date for correct trend math.

## Deploy on Render

`render.yaml` is configured to:
- install dependencies from `requirements.txt`
- start app with `uvicorn backend.server:app --host 0.0.0.0 --port $PORT`

Deploy this folder as the service root (`rootDir: dashboard`).
