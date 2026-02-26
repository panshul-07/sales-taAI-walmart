# taAI Architecture Notes (Implemented)

## Backend
- FastAPI API server in `/Users/panshulaj/Documents/front/dashboard/backend/server.py`
- Pickled log-linear model artifact: `/Users/panshulaj/Documents/front/dashboard/backend/model_artifacts/sales_log_linear.pkl`
- Scheduled retraining via APScheduler (interval: 6 hours)

## Model
- Natural-log target (`log(Weekly_Sales)`)
- Natural-log macro factors (`CPI`, `Unemployment`, `Fuel_Price`, shifted `Temperature`)
- Lag and seasonal features aligned to notebook intent
- Coefficients exposed as elasticity-style `beta_log`

## Frontend
- React 18 (CDN runtime) + TailwindCSS (CDN)
- Dashboard retains same layout categories:
  - KPI cards
  - Trend chart (actual / baseline / simulated)
  - What-if controls
  - Scatter chart
  - Correlation + coefficient table
- Embedded taAI chat widget with session continuity

## API Additions
- `POST /api/taai/chat`
- `GET /api/taai/sessions/{session_id}`

## Limitations
- Full cross-website assistant behavior requires extension-level architecture.
- "Any language" quality scales significantly with external LLM + translation stack.
