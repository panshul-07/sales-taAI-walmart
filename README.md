# Walmart Sales Forecasting Dashboard

Interactive 45-store sales forecasting dashboard built with:
- Python backend API (`http.server` + `pandas` + `scikit-learn`)
- React frontend (CDN module) + Plotly visualizations
- Tuned `ExtraTrees` forecasting model
- Stationarity and parametric diagnostics in Jupyter notebook

## Project Structure

- `dashboard/backend/server.py`: API server + model training/inference
- `dashboard/static/index.html`: React + Plotly dashboard UI
- `dashboard/render.yaml`: Render deployment blueprint
- `walmart_sales_forecasting.ipynb`: complete modeling notebook
- `build_notebook.py`: notebook generator script

## Local Run

```bash
cd "/Users/panshulaj/Documents/sale forecasting"
npm run frontend
```

Open: `http://localhost:8080`

## Frontend Code Location

- Main frontend (React + Plotly): `dashboard/static/index.html`
- Backend API server: `dashboard/backend/server.py`

## Notebook

The notebook includes:
- EDA and feature engineering
- Stationarity checks (ADF/KPSS)
- Robust parametric demand equation
- Ensemble ML model comparison

## Deploy

Use Render Blueprint with `dashboard/render.yaml` after pushing this repo to GitHub.
