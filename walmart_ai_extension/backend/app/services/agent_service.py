from __future__ import annotations

import json
from statistics import mean
from uuid import uuid4

from sqlalchemy.orm import Session

from app.core.config import settings
from app.db.models import ChatMessage
from app.db.schemas import ChatResponse
from app.services.anomaly_service import AnomalyService
from app.services.forecast_service import ForecastService
from app.services.rag_service import RAGService
from app.services.sql_service import SQLService
from app.services.viz_service import VisualizationService

try:
    from langchain_openai import ChatOpenAI
except Exception:  # pragma: no cover
    ChatOpenAI = None


class AgentService:
    def __init__(self):
        self.sql = SQLService()
        self.forecast = ForecastService()
        self.anomaly = AnomalyService()
        self.viz = VisualizationService()
        self.rag = RAGService(settings.chroma_host, settings.chroma_port)
        self.rag.seed()
        self.llm = None
        if settings.openai_api_key and ChatOpenAI is not None:
            self.llm = ChatOpenAI(model=settings.openai_model, api_key=settings.openai_api_key, temperature=0.1)

    def _save_message(self, db: Session, session_id: str, role: str, content: str, meta: dict | None = None) -> None:
        msg = ChatMessage(session_id=session_id, role=role, content=content, meta_json=json.dumps(meta or {}))
        db.add(msg)
        db.commit()

    def _rule_based(self, question: str, db: Session) -> tuple[str, str, list[dict], list]:
        q = question.lower()
        if "who made" in q or "creator" in q:
            return (
                "taAI was built by Panshulaj Pechetty and Rishabh Gupta.",
                "SELECT 1 as ok",
                [{"ok": 1}],
                [],
            )

        if "best store" in q or "top" in q or "compare" in q:
            sql = (
                "SELECT store_id, ROUND(SUM(revenue)::numeric,2) AS total_revenue "
                "FROM sales GROUP BY store_id ORDER BY total_revenue DESC"
            )
            rows = self.sql.run_query(db, sql, limit=10)
            chart = self.viz.bar_chart(rows, "store_id", "total_revenue", "Top Stores by Revenue")
            answer = "Top stores ranked by total revenue are shown below."
            return answer, sql, rows, [chart]

        if "trend" in q or "last 6 months" in q:
            sql = (
                "SELECT date, ROUND(SUM(revenue)::numeric,2) AS revenue "
                "FROM sales GROUP BY date ORDER BY date"
            )
            rows = self.sql.run_query(db, sql, limit=400)
            chart = self.viz.line_chart(rows, "date", "revenue", "Revenue Trend")
            avg_val = mean([float(r["revenue"]) for r in rows]) if rows else 0
            answer = f"Revenue trend generated. Average daily revenue in selected window is {avg_val:,.2f}."
            return answer, sql, rows, [chart]

        if "forecast" in q or "next quarter" in q or "predict" in q:
            sql = (
                "SELECT date, ROUND(SUM(revenue)::numeric,2) AS revenue "
                "FROM sales GROUP BY date ORDER BY date"
            )
            hist = self.sql.run_query(db, sql, limit=500)
            fc = self.forecast.forecast(hist, periods=13)
            chart = self.viz.line_chart(fc, "ds", "yhat", "Forecast Next Quarter")
            answer = "Forecast generated with base-case projection and uncertainty bounds."
            return answer, sql, fc, [chart]

        if "anomaly" in q or "unusual" in q:
            sql = "SELECT date, store_id, quantity, revenue, profit FROM sales ORDER BY date"
            rows = self.sql.run_query(db, sql, limit=5000)
            anomalies = self.anomaly.detect(rows)
            answer = f"Detected {len(anomalies)} potential anomalies based on multivariate behavior."
            chart = self.viz.scatter_chart(anomalies[:300], "quantity", "revenue", "Anomaly Pattern")
            return answer, sql, anomalies[:200], [chart]

        sql = "SELECT ROUND(SUM(revenue)::numeric,2) AS total_revenue, ROUND(SUM(profit)::numeric,2) AS total_profit FROM sales"
        rows = self.sql.run_query(db, sql, limit=1)
        answer = (
            f"Total revenue is {rows[0]['total_revenue']:,.2f} and total profit is {rows[0]['total_profit']:,.2f}. "
            "Ask for trend, top stores, forecast, anomaly, or causal analysis for deeper insights."
        )
        return answer, sql, rows, []

    def chat(self, db: Session, message: str, session_id: str | None) -> ChatResponse:
        sid = session_id or str(uuid4())
        self._save_message(db, sid, "user", message)

        answer, sql, table, charts = self._rule_based(message, db)

        rag_docs = self.rag.search(message, k=2)
        if rag_docs:
            answer += "\n\nEconomic context:\n- " + "\n- ".join(rag_docs)

        suggestions = [
            "Compare Store 45 vs Store 23 by profit trend",
            "Forecast next quarter revenue with 3 scenarios",
            "Show unusual sales patterns by category",
            "How would inflation affect demand?",
        ]

        self._save_message(db, sid, "assistant", answer, {"sql": sql})
        return ChatResponse(
            session_id=sid,
            answer=answer,
            sql=sql,
            confidence=0.86,
            charts=charts,
            table=table,
            suggestions=suggestions,
        )
