# Walmart Forecast Dashboard

Interactive 45-store sales intelligence dashboard with:
- Store-level filtering and KPI widgets
- Live Plotly charts (actual vs predicted, feature dependence, correlation heatmap)
- Macro-feature tracking (CPI, Unemployment, Fuel Price, Temperature)
- Darussalam Hyderabad themed background overlay

## Run Locally

```bash
cd "/Users/panshulaj/Documents/sale forecasting/dashboard"
/usr/local/bin/python3 backend/server.py
```

Open: `http://localhost:8080`

## API Endpoints

- `GET /api/health`
- `GET /api/stores`
- `GET /api/overview?store=all&weeks=160`
- `GET /api/timeseries?store=all&weeks=160`
- `GET /api/feature-dependencies?store=all&weeks=160`
- `GET /api/feature-series?store=all&weeks=160&feature=CPI`
- `GET /api/correlation?store=all&weeks=160`

## Notes

- Backend uses a tuned `ExtraTrees` model trained at server startup.
- The logo background URL is set in `static/index.html` as `logoUrl`.
- If you have the official Darussalam Hyderabad logo file, place it under `static/` and switch `logoUrl` to that local path.

## Deploy (Render Blueprint)

`render.yaml` is included. Push this folder to a Git repo, then create a Blueprint service on Render.
