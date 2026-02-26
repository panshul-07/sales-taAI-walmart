# Walmart AI Extension (Add-on to Existing Dashboard)

This folder adds a **new full-stack AI chatbot platform** without replacing your existing `dashboard/` app.

## Stack
- Backend: FastAPI + SQLAlchemy + LangChain tools
- Frontend: React + Tailwind + Plotly
- Database: PostgreSQL
- Vector DB: ChromaDB
- Forecasting: Prophet
- Anomaly Detection: IsolationForest

## Folder Structure
- `backend/app/main.py` - FastAPI entrypoint
- `backend/app/services/*` - SQL, forecasting, anomalies, RAG, visualization, agent logic
- `frontend/src/*` - chat UI, sidebar, chart renderer
- `database/init.sql` - PostgreSQL schema and indexes
- `docker-compose.yml` - local multi-container stack
- `.env.example` - environment config template

## Quick Start (Local Docker)
1. From this folder:
   ```bash
   cd walmart_ai_extension
   cp .env.example .env
   docker compose up --build
   ```
2. API: `http://localhost:8001/api/v1/health`
3. Frontend: `http://localhost:5174`

## Load CSV Data
Your CSV can use columns like:
- `date, store_id, product_id, category, quantity, revenue, cost, profit`

For Walmart CSV variants (`Store`, `Date`, `Weekly_Sales`), loader maps `Weekly_Sales -> revenue`.

Inside backend container:
```bash
python scripts/load_csv.py /data/sales.csv
```

## Sample Queries
- `Why did sales drop in March?`
- `Compare Store 45 vs Store 23`
- `Show me sales trend over last 6 months`
- `Forecast next quarter sales`
- `Were there unusual sales patterns?`
- `Who made this bot?`

## Architecture Flow
User question -> intent/tool routing -> SQL/ML/RAG execution -> LLM synthesis (optional) -> chart/table response.

## Security / Reliability
- SQL read-only guard (SELECT-only)
- Input validation with Pydantic
- Rate limiting middleware (SlowAPI)
- Structured logging
- Containerized deployment

## Extending
- Add new tools in `backend/app/services/`
- Add new intent routes in `AgentService._rule_based`
- Add new chart types in `VisualizationService`

## Production Notes
- Add managed PostgreSQL/Redis/Chroma
- Set real `OPENAI_API_KEY` in environment
- Add auth provider (JWT/OAuth)
- Add background workers for heavy jobs and report generation
